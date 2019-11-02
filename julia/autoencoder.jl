using Statistics
using Distributions
import Distributions: logpdf

include("model.jl")


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
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
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
