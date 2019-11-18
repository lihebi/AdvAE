function train_MNIST_model(model, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = my_xent(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(1e-3);
    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end

function advtrain!(model, attack_fn, loss, ps, data, opt; cb = () -> ())
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    step = 0
    @showprogress 0.1 "Training..." for d in data
        # FIXME can I use x,y in for variable?
        x, y = d
        x_adv = attack_fn(model, d...)

        # train xent loss using adv data
        gs = Flux.Tracker.gradient(ps) do
            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))
            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))

            l = adv_loss
            # l = adv_loss + clean_loss
            l
        end
        Flux.Tracker.update!(opt, ps, gs)

        step += 1

        if step % 40 == 0
            println()
            @show get!(m_cleanloss)
            @show get!(m_cleanacc)
            @show get!(m_advloss)
            @show get!(m_advacc)
        end
        # cb(step, total_loss)
        cb()
    end
end

function advtrain(model, attack_fn, trainX, trainY, valX, valY)
    model(trainX[1]);

    # evalcb = throttle(cb_fn , 5);
    evalcb = () -> ()

    # train
    opt = ADAM(1e-4);
    @epochs 3 advtrain!(model, attack_fn, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end

function evaluate_attack(model, attack_fn, trainX, trainY, testX, testY)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println("Clean accuracy:")
    # FIXME this may need to be evaluated on CPU
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(testX[1], testY[1])

    sample_and_view(testX[1], testY[1])

    x = testX[1]
    y = testY[1]

    @info "performing attack .."
    x_adv = attack_fn(model, x, y)
    @info "attack done."

    @show accuracy(x_adv, y)

    sample_and_view(x_adv, model(x_adv))

    # all test data
    m_acc = MeanMetric()
    @showprogress 0.1 "testing all data .." for d in zip(testX, testY)
        x, y = d
        x_adv = attack_fn(model, x, y)
        acc = accuracy(x_adv, y)
        add!(m_acc, acc)
    end
    @show get(m_acc)
    nothing
end

function evaluate_AE(ae, cnn, ds)
    # test clean cnn accuracy
    accuracy(x, y) = mean(onecold(cnn(x)) .== onecold(y))
    println("Computing clean accuracy ..")
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(testX[1], testY[1])
    sample_and_view(testX[1], testY[1])
    # use ae and visualize the images
    println("Visualizing decoded images ..")
    decoded = ae(testX[1]);
    maximum(decoded)
    minimum(decoded)
    # FIXME use .data because decoded is TrackedArray, why???
    sample_and_view(decoded.data, testY[1])
    sample_and_view(testX[1], testY[1])
    # add ae and test all accuracy
    ae_accuracy(x, y) = mean(onecold(cnn(ae(x))) .== onecold(y))
    println("Computing AE accuracy ..")
    @show ae_accuracy(trainX[1], trainY[1])
    @show ae_accuracy(testX[1], testY[1])
end

function upsample(x)
    ratio = (2,2,1,1)
    y = similar(x, (size(x) .* ratio)...)
    for i in Iterators.product(Base.OneTo.(ratio)...)
        loc = map((i,r,s)->range(i, stop = s, step = r), i, ratio, size(y))
        @inbounds y[loc...] = x
    end
    y
end


function AdvAE_train(ae, cnn, attack_fn, trainX, trainY, valX, valY)
    model = Chain(ae, cnn)
    model(trainX[1]);

    loss(x, y) = crossentropy(model(x), y)
    # how about try add regularizer
    # loss(x, y) = crossentropy(cnn(ae(x)), y) + mymse(ae(x), x)

    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    adv_accuracy(x, y) = begin
        x_adv = attack_fn(model, loss, x, y)
        accuracy(x_adv, y)
    end
    adv_loss(x, y) = begin
        x_adv = attack_fn(model, loss, x, y)
        loss(x_adv, y)
    end
    cb_fn() = begin
        # add a new line so that it plays nicely with progress bar
        @time begin
            println("")
            @show loss(valX[1], valY[1])
            @show adv_loss(valX[1], valY[1])
            @show accuracy(valX[1], valY[1])
            @show adv_accuracy(valX[1], valY[1])
            @show accuracy(trainX[1], trainY[1])
            @show adv_accuracy(trainX[1], trainY[1])
        end
    end
    evalcb = throttle(cb_fn, 10)

    # train
    # opt = Flux.Optimiser(Flux.ExpDecay(0.001, 0.5, 1000, 1e-4), ADAM(0.001))
    opt = ADAM(1e-3);
    @epochs 3 advtrain!(model, attack_fn, loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)

    # DEBUG testing gc time
    # @time advtrain!(model, attack_fn, loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)

    # DEBUG testing the trainning of AE using clean image CNN
    # @epochs 3 mytrain!(loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)
end

function AdvAE()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST(batch_size=64);

    # cnn = get_LeNet5()
    cnn = get_Madry_model()

    # FIXME this is in adv.jl
    train_MNIST_model(cnn, trainX, trainY, valX, valY)

    # ae
    ae = CNN_AE()
    train_AE(ae, trainX, trainY, valX, valY)

    # test accuracy
    evaluate_AE(ae, cnn, trainX, trainY, testX, testY)

    # adv train of ae
    AdvAE_train(ae, cnn, attack_PGD_k(7), trainX, trainY, valX, valY)
    AdvAE_train(ae, cnn, attack_PGD_k(20), trainX, trainY, valX, valY)
    AdvAE_train(ae, cnn, attack_PGD_k(40), trainX, trainY, valX, valY)

    # evaluate attack
    evaluate_attack(Chain(ae, cnn), attack_FGSM, trainX, trainY, testX, testY)
    evaluate_attack(Chain(ae, cnn), attack_PGD_k(20), trainX, trainY, testX, testY)
    evaluate_attack(Chain(ae, cnn), attack_PGD_k(40), trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD_k(40), trainX, trainY, testX, testY)
end


# Extend distributions slightly to have a numerically stable logpdf for `p` close to 1 or 0.
logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))

function vae()
    # Latent dimensionality, # hidden units.
    Dz, Dh = 5, 500
    # Components of recognition model / "encoder" MLP.
    A, μ, logσ = Dense(28^2, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)
    g(X) = (h = A(X); (μ(h), logσ(h)))
    z(μ, logσ) = μ + exp(logσ) * randn(Float32)
    # Generative model / "decoder" MLP.
    f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))

    # KL-divergence between approximation posterior and N(0, 1) prior.
    kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))
    # logp(x|z) - conditional probability of data given latents.
    logp_x_z(x, z) = sum(logpdf.(Bernoulli.(f(z)), x))
    # Monte Carlo estimator of mean ELBO using M samples.
    L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // M)

    loss(X) = -L̄(X) + 0.01f0 * sum(x->sum(x.^2), Flux.params(f))

    # Sample from the learned model.
    modelsample() = rand.(Bernoulli.(f(z.(zeros(Dz), zeros(Dz)))))


    evalcb = throttle(() -> @show(-L̄(X[:, rand(1:N, M)])), 30)
    opt = ADAM()
    ps = Flux.params(A, μ, logσ, f)

    @progress for i = 1:20
        @info "Epoch $i"
        Flux.train!(loss, ps, zip(data), opt, cb=evalcb)
    end
end


function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

"""
FIXME batch_size cannot equally divide
FIXME move to GPU on demand, otherwise GPU memory may exaust for large dataset
FIXME shuffle
FIXME minibatch
"""
function load_MNIST(; batch_size=32, val_split=0.1)
    imgs = Flux.Data.MNIST.images();
    labels = Flux.Data.MNIST.labels();

    # I don't want to reshape the data. I'll just add reshape layer
    # X = hcat(float.(reshape.(imgs, :))...);
    # Do I need to do the partition here?
    # N = length(imgs)
    # X = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)];

    function toXY(imgs, labels)
        let N = length(imgs),
            X = float.(imgs),
            Y = Flux.onehotbatch(labels, 0:9)

            # size(X)                     # (60000,)
            # size(X[1])                  # (28, 28)
            # size(Y)                     # (10, 60000)

            X, Y
        end
    end

    # batch_size = 128
    # mb_idxs = partition(1:length(train_imgs), batch_size)
    # train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs];

    # Prepare test set as one giant minibatch:
    # test_imgs = MNIST.images(:test);
    # test_labels = MNIST.labels(:test);
    # test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs));

    X, Y = toXY(Flux.Data.MNIST.images(:train), Flux.Data.MNIST.labels(:train))
    X2, Y2 = toXY(Flux.Data.MNIST.images(:test), Flux.Data.MNIST.labels(:test))

    N = length(X)
    # val_split = 0.1
    mid = round(Int, N * (1 - val_split))
    N2 = length(X2)

    trainX = gpu.([cat(X[i]..., dims = 4) for i in partition(1:mid, batch_size)]);
    trainY = gpu.([Y[:,i] for i in partition(1:mid, batch_size)]);

    valX = [cat(X[i]..., dims = 4) for i in partition(mid+1:N, batch_size)] .|> gpu;
    valY = [Y[:,i] for i in partition(mid+1:N, batch_size)] .|> gpu;

    testX = [cat(X2[i]..., dims = 4) for i in partition(1:N2, batch_size)] .|> gpu;
    testY = [Y2[:,i] for i in partition(1:N2, batch_size)] .|> gpu;

    return (trainX, trainY), (valX, valY), (testX, testY)

end

function test_load_MNIST()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
    sample_and_view(trainX[3])
    size(trainX)
    size(trainX[1])
    size(testX)
end


"""
    load_CIFAR10()

Return trainX, trainY, valX, valY, testX, testY

TODO I'm not going to move it onto GPU yet.

"""
function load_CIFAR10(; batch_size=32, val_split=0.1)
    getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

    function toXY(imgs)
        let N = length(imgs),
            X = [getarray(imgs[i].img) for i in 1:N],
            # FIXME this has to be 1:10, not 0:9. Take care about the
            # consistency
            Y = onehotbatch([imgs[i].ground_truth.class for i in 1:N],1:10)

            X, Y
        end
    end

    X, Y = toXY(trainimgs(Metalhead.CIFAR10));

    N = length(X)
    # TODO use random index
    mid = round(Int, N * (1 - val_split))
    # println("N=$(N), mid=$(mid)")

    # There is no testimgs. the valimgs is the test.
    # imgs3 = testimgs(CIFAR10);
    X2, Y2 = toXY(valimgs(Metalhead.CIFAR10));
    N2 = length(X2);

    trainX = gpu.([cat(X[i]..., dims = 4) for i in partition(1:mid, batch_size)]);
    trainY = gpu.([Y[:,i] for i in partition(1:mid, batch_size)]);

    valX = gpu.([cat(X[i]..., dims = 4) for i in partition(mid+1:N, batch_size)]);
    valY = gpu.([Y[:,i] for i in partition(mid+1:N, batch_size)]);

    testX = gpu.([cat(X2[i]..., dims = 4) for i in partition(1:N2, batch_size)]);
    testY = gpu.([Y2[:,i] for i in partition(1:N2, batch_size)]);

    return (trainX, trainY), (valX, valY), (testX, testY)
end


function test_load_CIFAR10()
    data = load_CIFAR10();
    (trainX, trainY), (valX, valY), (testX, testY) = data;

    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10();
    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10(batch_size=50);
    size(trainX)
    size(trainY)
    size(testY)
end

call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)
"""FIXME replace Tracker with Zygote
"""
function mytrain!(loss, ps, data, opt; cb = () -> ())
    # FIXME maybe this is used to reset the gradients?
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    @showprogress 0.1 "Training..." for d in data
        gs = Flux.Tracker.gradient(ps) do
            loss(d...)
        end
        Flux.Tracker.update!(opt, ps, gs)
        cb()
    end
end

function get_MNIST_FC_model()
    model = Chain(
        # (28,28,N)
        x -> reshape(x, :, size(x, 4)),
        # TODO regularizers?
        # TODO l2 weight decay?
        Dense(28^2, 32, relu),
        Dense(32, 10),
        softmax) |> gpu;
    return model
end

function get_MNIST_CNN_model()
    model = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        # x -> maxpool(x, (2,2)),
        MaxPool((2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        # x -> maxpool(x, (2,2)),
        MaxPool((2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        # x -> maxpool(x, (2,2)),
        MaxPool((2,2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(3*3*32, 10),

        # Finally, softmax to get nice probabilities
        softmax,
    ) |> gpu;
    return model
end

function test_tracker()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
    cnn = get_MNIST_CNN_model()

    size(trainX[1])
    # param(trainX[1]).grad[:,:,:,1]

    model = cnn
    x = trainX[1];
    y = trainY[1];
    # but this will clear it as well
    px = Flux.param(x);

    sum(px.grad)
    sum(px.tracker.grad)

    typeof(px.grad)
    typeof(px.data)

    loss(x, y) = crossentropy(model(x), y)

    grad = Flux.Tracker.extract_grad!(px)

    typeof(grad[1])
    grad.grad

    sign(grad.data)
    px.grad .= 0;

    theta = Flux.params(model);
    typeof(theta)
    typeof(px)

    newtheta = params(theta..., px)

    g = Flux.Tracker.gradient(() -> loss(px, y));
    g = Flux.Tracker.gradient(() -> loss(px, y), theta);
    g = Flux.Tracker.gradient(() -> loss(px, y), newtheta);

    for t in theta
        # @show typeof(t)
        @show sum(t.grad)
        @show sum(g[t])
    end

    sum(g[px])
end

function CNN3_AE()
    encoder = Chain(Conv((3,3), 1=>16, pad=(1,1), relu),
                    MaxPool((2,2)),
                    Conv((3,3), 16=>8, pad=(1,1), relu),
                    MaxPool((2,2)),
                    # FIMXE this is from 7 to 3, not good
                    Conv((3,3), 8=>8, pad=(1,1), relu),
                    MaxPool((2,2)))
    decoder = Chain(Conv((3,3), 8=>8, pad=(1,1), relu),
                    upsample,
                    Conv((3,3), 8=>8, pad=(1,1), relu),
                    upsample,
                    Conv((3,3), 8=>16, pad=(1,1), relu),
                    upsample,
                    Conv((3,3), 16=>1, pad=(1,1)),
                    x -> σ.(x))
    Chain(encoder, decoder) |> gpu
end

function test_MNIST_model(model)
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();

    model(trainX[1]);

    function alternative_loss(x, y)
        # We augment `x` a little bit here, adding in random noise
        x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))
        y_hat = model(x_aug)
        return crossentropy(y_hat, y)
    end

    loss(x, y) = crossentropy(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(0.001);

    # TODO early stopping
    # TODO learning rate decay

    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)

    # print out training details, e.g. accuracy
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(valX[1], valY[1])
    # Test set accuracy
    @show accuracy(testX[1], testY[1])

    sample_and_view(trainX[1], trainY[1])
    sample_and_view(testX[1], testY[1])
end

function test_MNIST()
    fc = get_MNIST_FC_model()
    test_MNIST_model(fc)
    cnn = get_MNIST_CNN_model()
    test_MNIST_model(cnn)
end

"""
CIFAR raw CNN models. Training acc 0.43, val 0.55, testing 0.4
"""
function test_CIFAR_model(model)
    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10();
    model(trainX[1]);

    function loss(x, y)
        # We augment `x` a little bit here, adding in random noise
        x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))
        y_hat = model(x_aug)
        return crossentropy(y_hat, y)
    end

    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(0.001)

    # training
    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)

    # into testing mode
    Flux.testmode!(model)

    # test accuracy
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(valX[1], valY[1])
    @show accuracy(testX[1], testY[1])

    # visualize the dataset
    sample_and_view(trainX[1], trainY[1])
    sample_and_view(testX[1], testY[1])

    # back into training node
    Flux.testmode!(model, false)
end


function test_CIFAR()
    accuracy_with_logits(model(testX[1]), testY[1])

    model = get_CIFAR_CNN_model()
    test_CIFAR_model(model)
    resnet20 = resnet(3) |> gpu;
    # 0.87, 0.74, 0.7
    resnet32 = resnet(5) |> gpu;
    resnet56 = resnet(9) |> gpu;
    resnet68 = resnet(11) |> gpu;
    test_CIFAR_model(resnet32)
end

function test_nested_meter()
    @showprogress 0.1 "Level 0 " for i in 1:10
        sleep(0.5)
        if i % 2 == 0
            @showprogress 0.1 " Level 1 " 1 for i2 in 1:10
                sleep(0.5)
                @showprogress 0.1 "  Level 2 " 2 for i3 in 1:10
                    sleep(0.5)
                end
            end
        end
        sleep(0.5)
    end
end


function test_logger()
    name="$(now())"
    TeeLogger(global_logger(),
              TBLogger("tensorboard_logs/exp-$name",
                       min_level=Logging.Info))
    TBLogger("tensorboard_logs/exp-$name", min_level=Logging.Info)

    # 7937 bytes
    @save "test.bson" weights=Tracker.data.(Flux.params(cpu(model)))
    # 37482 bytes, so I'm going to just save the model
    @save "test.bson" model
    with_logger(ConsoleLogger(stderr, Logging.Debug)) do
        @info "hello"
        @debug "world"
    end
end

"""
Several problems before going forward:

1. the augmentation should happen on-the-fly, so that each batch receives new
kinds of augmentation

2. I need a batched version of this for performance

"""
function test_augment()
    # the padding is easy, but there does not seem to be a random crop, I'll
    # need to write a simple random slicing algorithm
    #
    # In the meanwhile, there are more advanced augmentation techniques
    # using Augmentor
    pl = ElasticDistortion(6, scale=0.3, border=true) |>
        Rotate([10, -5, -3, 0, 3, 5, 10]) |>
        FlipX(0.5) |>
        ShearX(-10:10) * ShearY(-10:10) |>
        CropSize(32, 32) |>
        Zoom(0.9:0.1:1.2)
    augment(img, pl)
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


# FIXME model cannot be easily saved and loaded, weights can, but
# needs to get rid of CuArrays and TrackedArrays
#
@time @save model_file weights=Tracker.data.(Flux.params(cpu(model))) from_steps=step
