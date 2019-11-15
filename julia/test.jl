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

function evaluate(model, ds; attack_fn=(m,x,y)->x)
    xx,yy = next_batch!(ds) |> gpu

    @info "Sampling BEFORE images .."
    sample_and_view(xx, yy, model)
    @info "Sampling AFTER images .."
    adv = attack_fn(model, xx, yy)
    sample_and_view(adv, yy, model)

    @info "Testing multiple batches .."
    acc_metric = MeanMetric()
    @showprogress 0.1 "Testing..." for step in 1:10
        x,y = next_batch!(ds) |> gpu
        adv = attack_fn(model, x, y)
        acc = accuracy_with_logits(model(adv), y)
        add!(acc_metric, acc)
    end
    @show get!(acc_metric)
end


function test()
    Random.seed!(1234);

    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    model = get_Madry_model()[1:end-1]
    # model = get_LeNet5()[1:end-1]

    # Warm up the model
    model(x)

    # FIXME would this be 0.001 * 0.001?
    # FIXME decay on pleau
    # FIXME print out information when decayed
    # opt = Flux.Optimiser(Flux.ExpDecay(0.001, 0.5, 1000, 1e-4), ADAM(0.001))
    opt = ADAM(1e-4);

    logger = create_logger()
    with_logger(logger) do
        train!(model, opt, ds, print_steps=20)
    end

    @info "Adv training .."
    # custom_train!(model, opt, train_ds)

    with_logger(logger) do
        @epochs 2 advtrain!(model, opt, attack_PGD_k(40), ds, print_steps=20)
    end

    @info "Evaluating clean accuracy .."
    evaluate(model, test_ds)
    evaluate(model, test_ds, attack_fn=attack_FGSM)
    evaluate(model, test_ds, attack_fn=attack_PGD_k(40))
end
