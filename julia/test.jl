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

function test_free_train()
    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    model = get_Madry_model()
    model(x)
    opt = ADAM(1e-3)

    size(gpu(zeros(size(x)...)) + x)

    free_train!(model, opt, ds, 0.3)
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


function sha_arr(x)
    # assuming 3 channel images in a batch
    res = map(collect(1:size(x)[end])) do i
        img = x[:,:,:,i]
        bytes2hex(sha256(repr(img)))
    end
    res
end

function get_wrong_sha(model, ds)
    # evaluate model, and get the wrong sample shas
    res = []
    @showprogress for i in 1:ds.nbatch
        x, y = next_batch!(ds) |> gpu
        logits = model(x)
        wrong_idx = onecold(cpu(logits)) .!= onecold(cpu(y))
        if length(wrong_idx) > 0
            # get sha
            shas = sha_arr(cpu(x)[:,:,:,wrong_idx])
            push!(res, shas...)
        end
    end
    res
end

function load_trained_model(file)
    @load file model
    return gpu(model)
end

# https://github.com/FluxML/Flux.jl/issues/160
function weight_params(m::Chain, ps=Flux.Params())
    map((l)->weight_params(l, ps), m.layers)
    ps
end
weight_params(m::Dense, ps=Flux.Params()) = push!(ps, m.W)
weight_params(m::Conv, ps=Flux.Params()) = push!(ps, m.weight)
weight_params(m::ConvTranspose, ps=Flux.Params()) = push!(ps, m.weight)
weight_params(m, ps=Flux.Params()) = ps

function test_weight_params()
    # get model
    weight_params(model)
end


function test_boundary()
    # test whether all the nail households
    model = CIFAR10_pretrained_fn()
    ds, test_ds = load_CIFAR10_ds(batch_size=128);

    itM = load_trained_model(
        "trained/CIFAR10/itadv-schedule/(0.0004=>40000,4.0e-5=>60000,4.0e-6=>80000).bson")
    dyM = load_trained_model(
        "trained/CIFAR10-dyattack/(0=>2000,1=>3000,2=>4000,3=>5000,4=>6000,7=>8000).bson")
    freeM = load_trained_model("trained/CIFAR10-free/test-2019-11-18T16:57:55.947.bson")

    sha_it = get_wrong_sha(itM, test_ds);
    sha_dy = get_wrong_sha(dyM, test_ds);
    sha_free = get_wrong_sha(freeM, test_ds);

    size(sha_it)
    size(unique(sha_it))
    @show length(sha_it) length(sha_dy) length(sha_free)
    @show length(unique(sha_it)) length(unique(sha_dy)) length(unique(sha_free))
    @show size(intersect(sha_it, sha_dy, sha_free))
    @show size(intersect(sha_it, sha_dy, sha_free))

    ps = params(model);

    model[1].weight in ps


    model

    Flux.mapparams(model) do p
        @show typeof(p)
        p
    end


    model(x)
    typeof(ps)
    typeof(keys(ps.params.dict))
    keytype(ps.params.dict)

    shas[1]
    size(shas)
    size(unique(shas))
    size(unique(shas2))
    # overlap
    size(intersect(shas, shas2))
end

function test()
    ds, test_ds = load_CIFAR10_ds(batch_size=128);
    x, y = next_batch!(ds) |> gpu;
    x, y = next_batch!(test_ds) |> gpu;
    model = load()
    model[2].μ
    # Flux.testmode!(model, false)
    model(x)

    # hash the x
    img = x[:,:,:,1];
    size(img)

    # using SHA
    @time bytes2hex(sha256("test"))

    @time bytes2hex(sha256(repr(img)))

    cat(res..., dims=4)
    size(res)
    res[1]
    size(unique(res))
    res[1]

    ss = map(x) do img
        bytes2hex(sha256(repr(img)))
    end

    for i in 1:10
        i
    end

    accuracy_with_logits(model(x), y)

    sum(onecold(cpu(model(x))) .== onecold(cpu(y)))


    adv = attack_CIFAR10_PGD_k(7)(model, x, y);

    view_preds(model, x, y)
    view_preds(model, adv, y)

    sample_and_view(wrong, cpu(y)[])
    sample_and_view(right)
    displayable("image/png")
end

function view_preds(model, x, y)
    @show accuracy_with_logits(model(x), y)
    right_idx = onecold(cpu(model(x))) .== onecold(cpu(y));
    wrong_idx = onecold(cpu(model(x))) .!= onecold(cpu(y));
    @info "right predictions:"
    sample_and_view(cpu(x)[:,:,:,right_idx], cpu(y)[:,right_idx], model)
    @info "wrong predictions:"
    sample_and_view(cpu(x)[:,:,:,wrong_idx], cpu(y)[:,wrong_idx], model)
    nothing
end
