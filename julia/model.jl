using Revise

using ProgressMeter

include("data.jl")

"""
Sample up to 10 images and show the image and label. If less than 10, use all.

"""
function sample_and_view(X, Y)
    num = min(size(X)[4], 10)
    @info "Showing $num images .."
    imgs = cpu(hcat([X[:,:,:,i] for i in 1:num]...))
    labels = onecold(Y[:,1:num], 0:9)
    viewrepl(imgs)
    @show labels
    nothing
end

""" Drop-in replacement for Flux.train!. The @showprogress prints on
stderr by default. Thus, I cannot nicely show callbacks.

TODO Maybe output to a file and show that file in Emacs? But I lose
the nice @showprogress macro.

"""
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)
function mytrain!(loss, ps, data, opt; cb = () -> ())
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    @showprogress 0.1 "Training..." for d in data
        try
            gs = Flux.Tracker.gradient(ps) do
                loss(d...)
            end
            Flux.Tracker.update!(opt, ps, gs)
            cb()
        catch ex
            if ex isa Flux.StopException
                break
            else
                rethrow(ex)
            end
        end
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
        Dense(288, 10),

        # Finally, softmax to get nice probabilities
        softmax,
    ) |> gpu;
    return model
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

    @epochs 10 mytrain!(loss, params(model), zip(trainX, trainY), opt, cb=evalcb)

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

function get_CIFAR_CNN_model()
    model = Chain(
        Conv((5,5), 3=>16, relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        # Dense(200, 120),
        # Dense(120, 84),
        # Dense(84, 10),
        Dense(200, 10),
        softmax) |> gpu;
    return model
end

struct ResidualBlock
  conv_layers
  norm_layers
  shortcut
end
Flux.@treelike ResidualBlock

function (block::ResidualBlock)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers)-1
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    relu.(((block.norm_layers[end])((block.conv_layers[end])(value)))
          + block.shortcut(input))
end

"""Identity block, consisting of Conv-BN-Conv-BN + input
"""
function identity_block(kernel_size, filters)
    # conv BN RELU
    local conv_layers = []
    local norm_layers = []
    push!(conv_layers, Conv(kernel_size, filters, pad=1, stride=1))
    push!(conv_layers, Conv(kernel_size, filters, pad=1, stride=1))
    push!(norm_layers, BatchNorm(filters[2]))
    push!(norm_layers, BatchNorm(filters[2]))
    ResidualBlock(Tuple(conv_layers), Tuple(norm_layers), identity)
end

function conv_block(kernel_size, filters, stride)
    local conv_layers = []
    local norm_layers = []
    # half the feature map
    push!(conv_layers, Conv(kernel_size, filters[1]=>filters[2], pad=1, stride=stride))
    push!(conv_layers, Conv(kernel_size, filters[2]=>filters[2], pad=1))
    push!(norm_layers, BatchNorm(filters[2]))
    push!(norm_layers, BatchNorm(filters[2]))
    shortcut = Chain(Conv((1,1), filters,
                          pad = (0,0),
                          stride = stride),
                     BatchNorm(filters[2]))
    return ResidualBlock(Tuple(conv_layers), Tuple(norm_layers), shortcut)
end

function res_block(num_blocks, kernel_size, filters)
    local layers = []
    # conv1 = conv_block(kernel_size, filters, stride=conv_stride)
    # push!(layers, conv1)
    for i = 1:num_blocks
        id_layer = identity_block(kernel_size, filters)
        push!(layers, id_layer)
    end
    return Chain(layers...)
end

function resnet(num_blocks)
    # num_blocks: 3,5,9,11
    # (6*num_blocks + 2)
    # 20, 32, 56, 68
    # USE 9, resnet56
    Chain(
        Conv((3,3), 3=>16, stride=(1,1), pad=1),
        BatchNorm(16),
        # 32,32,16

        # 2n 32x32, 16
        conv_block((3,3), 16=>16, 1),
        res_block(num_blocks-1, (3,3), 16=>16),

        conv_block((3,3), 16=>32, 2),
        res_block(num_blocks-1, (3,3), 32=>32),

        conv_block((3,3), 32=>64, 2),
        res_block(num_blocks-1, (3,3), 64=>64),

        MeanPool((8,8)),
        x -> reshape(x, :, size(x,4)),
        Dense(64,10),
        softmax)
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
    @epochs 10 mytrain!(loss, params(model), zip(trainX, trainY), opt, cb=evalcb)

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
    cnn = get_CIFAR_CNN_model()
    test_CIFAR_model(cnn)
    resnet20 = resnet(3) |> gpu;
    # 0.87, 0.74, 0.7
    resnet32 = resnet(5) |> gpu;
    resnet56 = resnet(9) |> gpu;
    resnet68 = resnet(11) |> gpu;
    test_CIFAR_model(resnet32)
end
