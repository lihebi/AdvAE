using Statistics
using Distributions
import Distributions: logpdf

include("adv.jl")


function dense_AE()
    # FIXME why leakyrelu
    encoder = Chain(x -> reshape(x, :, size(x, 4)),
                    Dense(28 * 28, 32, relu)) |> gpu
    decoder = Chain(Dense(32, 28 * 28),
                    x -> reshape(x, 28, 28, 1, :),
                    x -> σ.(x)) |> gpu
    Chain(encoder, decoder)
end

# FIXME use clamp?
# reshape(clamp.(x, 0, 1), 28, 28)

mycrossentropy(ŷ, y; ϵ=eps()) = -sum(y .* log.(ŷ.+ϵ) .* 1) * 1 // size(y)[end]
mymse(ŷ, y) = sum((ŷ .- y).*(ŷ .- y)) * 1 // length(y)

# maximum(trainX[1])
# minimum(trainX[1])
# maximum(ae(trainX[1]))
# minimum(ae(trainX[1]))

function train_AE(model, trainX, trainY, valX, valY)
    # loss(x) = Flux.mse(model(x), x)
    loss(x) = mymse(model(x), x)

    evalcb = throttle(() -> @show(loss(valX[1])), 1)
    opt = ADAM()

    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX), opt, cb = evalcb)
    # @epochs 10 Flux.train!(loss, Flux.params(model), zip(trainX), opt, cb = evalcb)
end

function evaluate_AE(ae, cnn, trainX, trainY, testX, testY)
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

# function upsample(x)
#     ratio = (2,2,1,1)
#     y = similar(x, (size(x) .* ratio)...)
#     for i in Iterators.product(Base.OneTo.(ratio)...)
#         loc = map((i,r,s)->range(i, stop = s, step = r), i, ratio, size(y))
#         @inbounds y[loc...] = x
#     end
#     y
# end

"""From https://discourse.julialang.org/t/upsampling-in-flux-jl/25919/3
"""
function upsample(x)
    ratio = (2, 2, 1, 1)
    (h, w, c, n) = size(x)
    y = similar(x, (1, ratio[1], 1, ratio[2], 1, 1))
    fill!(y, 1)
    z = reshape(x, (h, 1, w, 1, c, n))  .* y
    reshape(permutedims(z, (2,1,4,3,5,6)), size(x) .* ratio)
end

function CNN_AE()
    # FIXME padding='same'?
    encoder = Chain(Conv((3,3), 1=>16, pad=(1,1), relu),
                    # BatchNorm(16)
                    MaxPool((2,2)))
    decoder = Chain(Conv((3,3), 16=>16, pad=(1,1), relu),
                    # UpSampling((2,2)),
                    upsample,
                    Conv((3,3), 16=>1, pad=(1,1)),
                    x -> σ.(x))
    Chain(encoder, decoder) |> gpu
end

function CNN2_AE()
    encoder = Chain(Conv((3,3), 1=>16, pad=(1,1), relu),
                    MaxPool((2,2)),
                    Conv((3,3), 16=>8, pad=(1,1), relu),
                    MaxPool((2,2)))
    decoder = Chain(Conv((3,3), 8=>8, pad=(1,1), relu),
                    upsample,
                    Conv((3,3), 8=>16, pad=(1,1), relu),
                    upsample,
                    Conv((3,3), 16=>1, pad=(1,1)),
                    x -> σ.(x))
    Chain(encoder, decoder) |> gpu
end

function test_AE()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST(batch_size=128);
    cnn = get_MNIST_CNN_model()
    # FIXME this is in adv.jl
    train_MNIST_model(cnn, trainX, trainY, valX, valY)

    ae = dense_AE()
    ae = CNN_AE()
    ae = CNN2_AE()

    train_AE(ae, trainX, trainY, valX, valY)

    # test accuracy
    evaluate_AE(ae, cnn, trainX, trainY, testX, testY)
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
    opt = ADAM(0.001);
    @epochs 3 advtrain!(model, attack_fn, loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)

    # DEBUG testing gc time
    # @time advtrain!(model, attack_fn, loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)

    # DEBUG testing the trainning of AE using clean image CNN
    # @epochs 3 mytrain!(loss, Flux.params(ae), zip(trainX, trainY), opt, cb=evalcb)
end

function AdvAE()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST(batch_size=64);
    # cnn
    cnn = get_MNIST_CNN_model()
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
