using Flux
using Flux.Data.MNIST
using Flux: @epochs
using Flux: throttle
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using Random
using Statistics

using Images: channelview
using Base.Iterators: partition

using CuArrays

using Metalhead

using EmacsREPL

using Revise

include("data.jl")
CuArrays.allowscalar(false)
CuArrays.allowscalar(true)

"""
Sample 10 images and show the image and label
"""
function sample_and_view(X, Y)
    imgs = cpu(hcat([X[1][:,:,:,i] for i in 1:10]...))
    labels = onecold(Y[1][:,1:10], 0:9)
    viewrepl(imgs)
    @show labels
    nothing
end

function test_FC()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();

    # TODO input dimension matters?
    
    model = Chain(
        # (28,28,N)
        x -> reshape(x, :, size(x, 4)),
        # TODO regularizers?
        # TODO l2 weight decay?
        Dense(28^2, 32, relu),
        Dense(32, 10),
        softmax) |> gpu;

    model(trainX[1]);
    
    loss(x, y) = crossentropy(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 1);
    opt = ADAM();

    # TODO early stopping
    # TODO learning rate decay

    @epochs 10 Flux.train!(loss, params(model), zip(trainX, trainY), opt, cb = evalcb)

    # TODO print out training details, e.g. accuracy
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(valX[1], valY[1])
    @show accuracy(testX[1], testY[1])

    # TODO Test set accuracy
    # tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
    # tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu
    # accuracy(tX, tY)

    sample_and_view(trainX, trainY)
    sample_and_view(testX, testY)
end

# x = rand(Float32, 10, 3) |> gpu;
# y = Flux.onehotbatch(1:3, 1:10) |> gpu;
# accuracy(x, y) = Flux.onecold(x) .== Flux.onecold(y);
# accuracy(x, y)


function test_Conv()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
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
        Dense(288, 10),

        # Finally, softmax to get nice probabilities
        softmax,
    ) |> gpu;
    
    model(trainX[1]);
    
    function loss(x, y)
        # We augment `x` a little bit here, adding in random noise
        x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

        y_hat = model(x_aug)
        return crossentropy(y_hat, y)
    end
    
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    
    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 1);
    opt = ADAM(0.001)

    # training
    @epochs 10 Flux.train!(loss, params(model), zip(trainX, trainY), opt, cb=evalcb)

    # test accuracy
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(valX[1], valY[1])
    @show accuracy(testX[1], testY[1])

    # visualize the dataset
    sample_and_view(trainX, trainY)
    sample_and_view(testX, testY)
end


"""
CIFAR raw CNN models
"""
function test_cifar_Conv()
    
end

"""
CIFAR using ResNet models
"""
function test_cifar_Resnet()
end
