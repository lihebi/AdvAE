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

include("data.jl")
include("model.jl")
include("train.jl")

function test_BatchNorm()
    using Flux
    using CuArrays
    bn = BatchNorm(32);
    testimg = randn(Float32, 28, 28, 32, 1);
    @time bn(testimg);
    @time gpu(bn)(gpu(testimg));

    Flux.testmode!(bn)
    Flux.testmode!(bn, false)
    sum(bn(testimg))

    p = Flux.param(testimg);

    gs = Tracker.gradient(()->sum(bn(p)), Flux.params(bn))
    gs

    function f()
        sum(bn(p))
    end
    Flux.gradient(f, p)
end

function test_allowscalar()
    CuArrays.allowscalar(false)
    x = rand(Float32, 10, 3) |> gpu;
    y = Flux.onehotbatch(1:3, 1:10) |> gpu;
    accuracy(x, y) = Flux.onecold(x) .== Flux.onecold(y);
    accuracy(x, y)
    CuArrays.allowscalar(true)
end

function test_bn_grad()
    x = randn(Float32, 32, 32, 3, 16) |> gpu
    model = Chain(
        myConv((3,3), 3=>16, stride=(1,1), pad=1),
        BatchNorm(16),
        MeanPool((8,8)),
        x -> reshape(x, :, size(x)[end]),
        Dense(256,10)) |> gpu
    model(x)

    Flux.gradient() do
        sum(model(x))
    end

    Flux.testmode!(model)

    # In test mode, it would fail if I don't patch Flux.jl/src/cuda/cudnn.jl
    # cudnnBNBackward!. The root cause is sum(iter; dims) have dims as explicit
    # kwarg, and thus calling without dims=... will call the wrong sum. Also
    # squeeze is deprecated for dropdims. NOTE: this is in Flux v0.9, the master
    # branch currently removed manual testmode, see discussions:
    #
    # - Train/test mode discussion by MikeInnes https://github.com/FluxML/Flux.jl/issues/643
    # - https://github.com/FluxML/Flux.jl/issues/909
    # - https://github.com/FluxML/Flux.jl/issues/232
    Flux.gradient() do
        sum(model(x))
    end
end


# TODO learning rate?
function test_CIFAR10_ds()
    ds, test_ds = load_CIFAR10_ds(batch_size=128);
    x, y = next_batch!(ds) |> gpu;

    # model = get_CIFAR_CNN_model()
    model = resnet(20) |> gpu
    # model = resnet(32)
    # model = resnet(56)
    # model = resnet(68)

    # model = WRN(28, 10) |> gpu;
    model = WRN(16, 4) |> gpu;

    model(x)

    adv = attack_CIFAR10_PGD_k(7)(model, x, y);
    Flux.testmode!(model)
    adv = attack_CIFAR10_PGD_k(7)(model, x, y);

    size(adv)

    augment = Augment()
    augx = augment(cpu(x)) |> gpu;
    # augment(x)
    model(augx)
    accuracy_with_logits(model(augx), y)

    opt = ADAM(1e-3)

    train!(model, opt, ds, print_steps=50)

    accuracy_with_logits(model(x), y)

end



function evaluate_AE(ae, ds; cnn=nothing)
    xx,yy = next_batch!(ds) |> gpu

    @info "Sampling BEFORE images .."
    sample_and_view(xx, yy, cnn)
    @info "Sampling AFTER images .."
    # NOTE: I need .data, currently handled in sample_and_view
    rec = ae(xx)
    sample_and_view(rec, yy, cnn)
end

function test_CIFAR_ae()
    ds, test_ds = load_CIFAR10_ds(batch_size=128);
    x, y = next_batch!(ds) |> gpu;
    x, y = next_batch!(test_ds) |> gpu;

    # get a pretrained WRN
    cnn = CIFAR10_pretrained_fn()
    model = CIFAR10_pretrained_fn()

    accuracy_with_logits(cnn(x), y)
    accuracy_with_logits(model(x), y)
    Flux.testmode!(model)
    Flux.testmode!(model, false)
    Flux.testmode!(cnn)
    Flux.testmode!(cnn, false)

    params(cnn[2].μ)

    cnn(x)

    onecold(cpu(cnn(x)))

    model[2].μ
    cnn[2].μ
    cnn[2].σ²

    cnn

    train!(cnn, opt, ds)

    @save "test.bson" MMM=cpu(cnn) opt
    @load "test.bson" MMM

    MMM=gpu(MMM);

    fieldnames(typeof(cnn[5]))
    cnn[5].norm_layers[1].σ²

    # get the AE

    # ae = cifar10_AE()
    ae = dunet()
    encoder, decoder = cifar10_deep_AE()

    ae = Chain(encoder, decoder)

    rx = randn(Float32, 32, 32, 3, 16) |> gpu;
    size(encoder(rx))
    size(decoder(encoder(rx)))
    size(ae(rx))



    opt = ADAM(1e-3)
    # train the AE
    aetrain!(ae, opt, ds)

    evaluate_AE(ae, test_ds, cnn=cnn)

    param_count(ae)
    param_count(cnn)
end

function test_AE()
    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    cnn = get_Madry_model()
    @show size(cnn(x))

    opt = ADAM(1e-3)

    Flux.testmode!(cnn, false)
    cnn = MNIST_pretrained_fn()

    accuracy_with_logits(cnn(x), y)

    cnn(x)

    size(cnn[1](x))

    typeof(cnn)

    # TODO schedule lr simply by setting field of opt, and maintain the
    # state. FIXME But it seems that the state is maintained by a dict with key
    # Param(x). I'm not sure if this aproach can resume the state.
    #
    # opt.eta = 1e-4

    train!(cnn, opt, ds)

    cnn = maybe_train("trained/pretrain-MNIST.bson", get_Madry_model(), train_fn)


    # ae = dense_AE()
    ae = CNN_AE()
    # ae = CNN2_AE()
    size(ae(x))

    # FIXME opt states?
    opt = ADAM(1e-3)
    opt = ADAM(1e-4)
    # FIXME Flux.mse(logits, x)
    # FIXME sigmoid mse?
    aetrain!(ae, opt, ds)

    evaluate_AE(ae, test_ds, cnn=cnn)

    advae_train!(ae, cnn, opt, attack_PGD_k(40), ds, print_steps=20)

    evaluate(cnn, test_ds)
    evaluate(cnn, test_ds, attack_fn=attack_PGD_k(40))

    evaluate(Chain(ae, cnn), test_ds)
    evaluate(Chain(ae, cnn), test_ds, attack_fn=attack_FGSM)
    evaluate(Chain(ae, cnn), test_ds, attack_fn=attack_PGD_k(40))
end

function test_log_image()
    logger = TBLogger("tensorboard_logs/test/image", tb_append, min_level=Logging.Info)

    ds, test_ds = load_CIFAR10_ds(batch_size=128);
    x, y = next_batch!(ds) |> gpu;
    img = x[:,:,:,1];

    # CAUTION if pass in raw img, the error message contains the whole array,
    # which is very slow
    showable("image/png", MyImage(img))
    log_image(logger, "testimage", MyImage(img));

    display(MyImage(img))
end
