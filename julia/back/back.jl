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
