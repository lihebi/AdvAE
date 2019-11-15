using Revise

using Flux, Zygote
using Flux.Data.MNIST
using Flux: @epochs
using Flux: throttle
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using Statistics

using Images: channelview
using Base.Iterators: partition

using CuArrays

using Metalhead

using EmacsREPL

function test_BatchNorm()
    using Flux
    using CuArrays
    bn = BatchNorm(32);
    testimg = randn(Float32, 28, 28, 32, 1);
    @time bn(testimg);
    @time gpu(bn)(gpu(testimg));
end

function test_allowscalar()
    CuArrays.allowscalar(false)
    x = rand(Float32, 10, 3) |> gpu;
    y = Flux.onehotbatch(1:3, 1:10) |> gpu;
    accuracy(x, y) = Flux.onecold(x) .== Flux.onecold(y);
    accuracy(x, y)
    CuArrays.allowscalar(true)
end

function test_CIFAR_ds()
    ds, test_ds = load_CIFAR10_ds(batch_size=64);
    ds
    test_ds
    x, y = next_batch!(ds) |> gpu;
    size(x)
    x[:,:,:,1]

    accuracy_with_logits(model(x), y)
    accuracy_with_logits(resnet20(x), y)

    model = get_CIFAR_CNN_model()[1:end-1]
    resnet20 = resnet(3)[1:end-1] |> gpu;
    # resnet32 = resnet(5) |> gpu;
    # resnet56 = resnet(9) |> gpu;
    # resnet68 = resnet(11) |> gpu;

    opt = ADAM(1e-3)
    @epochs 5 train!(model, opt, ds, print_steps=100)
    # TODO data augmentation
    @epochs 5 train!(resnet20, opt, ds, print_steps=100)
end
