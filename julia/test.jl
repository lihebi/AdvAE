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

# I should probably control the environment explicitly, as julia seems
# to be fragile for it.
#
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# Pkg.status()


function test_Metalhead()
    Metalhead.download(CIFAR10)
    vgg = VGG19()
    load("Elephant.jpg")
end

function test_julia()
    hello="3"
    "1 + 2 = $(hello).png"
    FileIO.save(File(format"PNG", "a.png"), X[1].img)
    FileIO(format"PNG", "a.png")
end

function test_array()
    # these should be enough for understanding ..., cat, and comprehensions
    size(vcat([[1 2 3] for i in  1:20]...))
    size(hcat([[1, 2, 3] for i in  1:20]...))
    size(hcat([[1, 2, 3] for i in  1:20]))

    typeof(collect([[1,2,3] for i in  1:20]))
end


function test()
    function ret()
        # this can refer to outside a .. I need to use let binding
        a
    end
    a = 1
    ret()
end

test()


function test_resnet()
    # load CIFAR10 data
    Metalhead.download(CIFAR10)
    X = trainimgs(CIFAR10);
    # size(X[1])
    X[1].img
    viewrepl(X[3].img)

    getarray(X[1].img)
    imgs = [getarray(X[i].img) for i in 1:50000];
    labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10);
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)]);
    valset = collect(49001:50000);
    valX = cat(imgs[valset]..., dims = 4) |> gpu;
    valY = labels[:, valset] |> gpu;


    # construct resnet
    # run
end


# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

Flux.train!
function test_MNIST()
    # load MNIST data
    imgs = MNIST.images();
    # size(X)
    # FIXME why do I need to hcat? It looks like a transpose


    # FC Model
    m = Chain(
        Dense(28^2, 32, relu),
        Dense(32, 10),
        softmax) |> gpu

    loss(x, y) = crossentropy(m(x), y)
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

    dataset = repeated((X, Y), 1)
    evalcb = () -> @show(loss(X, Y))

    println(" hello hello hello hello hello hello"
            )

    @epochs 2 println("Hello")

    opt = ADAM()

    # @epochs 10 Flux.train!(loss, params(m), (X,Y), opt, cb = Flux.throttle(evalcb, 10))

    Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))
    @epochs 10 Flux.train!(loss, params(m), dataset, opt)

    for j=1:2
        for i=1:10
            println("Hello")
        end
    end

    for i=1:100
        if i % 50 == 0
            @info "Epoch $i"
            ds = [(X,Y)]
            Flux.train!(loss, params(m), ds, opt)
            println("Loss: ", loss(X,Y))
            # println("Accuracy: ", accuracy(X,Y))
        end
    end

    onecold(m(X)) .== onecold(Y)
    accuracy(X,Y)

    accuracy(X, Y)

    # Test set accuracy
    tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
    tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

    accuracy(tX, tY)
end

function test_random()

    tmp = MNIST.images();
    size(tmp)
    typeof(tmp[1][1,1])
    float.(tmp[1]) == float(tmp[1])
    minimum(float.(tmp[1]))
    ?typeof
    ?maximum
    X = float.(hcat(vec.(MNIST.images())...)) .> 0.5;

    typeof(vec.(MNIST.images()))
    size(tmp)
    size(MNIST.images())
    size(vec.(MNIST.images()))
    size(MNIST.images()[1])

    size(vec.(MNIST.images())[1])
    size(hcat(vec.(MNIST.images())...))
    size(hcat(MNIST.images()...))

    # visualize the image?

    N, M = size(X, 2), 100;
    data = [X[:,i] for i in Iterators.partition(1:N,M)];

    # mnist classifier


    # adversarial attacks
    # https://github.com/jaypmorgan/Adversarial.jl

    # adversarial training
    # advAE

end


function test_CIFAR10()
    ## Examing the dataset
    imgs = trainimgs(CIFAR10);
    size(imgs)               # (50000,)
    # CIFAR10 needs to call .img
    size(imgs[1].img)        # (32,32)
    typeof(imgs[1])          # Metalhead.TrainingImage
    viewrepl(imgs[1].img)
    eltype(imgs[1].img)      # ColorTypes.RGB

    ## converting RGB image into float array with 3 channels
    getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))
    size(channelview(imgs[1].img)) # (3,32,32)
    size(permutedims(channelview(imgs[1].img), (2,3,1))) # (32, 32, 3)
    # now can convert to float
    eltype(float.(permutedims(channelview(imgs[1].img), (2,3,1))))
    # but it does not seem to be any difference to use float. or float
    # and the default seems to be Float32

    # reshape and convert to float
    X = [getarray(imgs[i].img) for i in 1:50000];
    # labels and one-hot encoding
    Y = onehotbatch([imgs[i].ground_truth.class for i in 1:50000],1:10);
    size(X)                     # (50000,)
    typeof(X)                   # Array{Array{Float32,3},1}

    # I should concat on dims=4 because there're already 3 dimensions,
    # that is the new dimension.
    #
    # I should have 1:100 explicitly instead of :100, as required by
    # Julia syntax.
    #
    # 1:100 is fine I should not concat a large number, because they
    # are inlined in the function call, and that would create serious
    # performance issue. There does not seem to have a easy way to
    # adjust the shape of the "array of array". The right way seems to
    # split into batch.
    size(cat(X[1:100]..., dims=4))

    # It looks like CHW is the default for implementation purpose.  In
    # addition, Julia uses column-major, and put N at the end,
    # i.e. WHCN.
    size(Y)                     # (10, 50000)

    # cat(A...; dims=dims): Concatenate the input arrays along the
    # specified dimensions in the iterable dims. Seems that dims can
    # also be a integer like in this case.
    train_data = gpu.([(cat(X[i]..., dims = 4), Y[:,i]) for i in partition(1:49000, 100)]);
    size(train_data)                 # (490,)
    size(train_data[1][1])           # (32, 32, 3, 100)
    size(train_data[1][2])           # (10, 100)
    val_indices = collect(49001:50000);
    valX = cat(X[val_indices]..., dims = 4) |> gpu;
    valY = Y[:, val_indices] |> gpu;
    size(valX)                  # (32, 32, 3, 1000)
    size(valY)                  # (10, 1000)
end

function test_MNIST()
    # reading and display images
    # MNIST
    imgs = MNIST.images();
    size(imgs)                  # (60000,)
    eltype(imgs[1])
    size(imgs[1])               # (28,28)
    viewrepl(imgs[1])

    size(reshape(imgs[1], :))  # (784,), i.e. flatten
    eltype(imgs[1])            # ColorTypes.Gray
    eltype(Float32.(imgs[1]))  # Float32
    eltype(float(imgs[1]))     # Float64

    size(float.(reshape.(imgs, :)))     # (60000,)

    # size(float.(reshape.(imgs, :)))

    size(float.(imgs))

    size(cat(float.(imgs)..., dims=3))

    # X = hcat(float.(reshape.(imgs, :))...) |> gpu;

    labels = MNIST.labels();
    Y = Flux.onehotbatch(labels, 0:9) |> gpu;
    size(X)                     # (784, 60000)
    size(Y)                     # (10, 60000)

    # I probably want to use (28,28,60000)
end

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

function test_ResNet()
    testimg = randn(Float32, 32,32, 16,1) |> gpu;
    size(testimg)
    net = conv_block((3,3), 16=>32, 2) |> gpu
    size(net(testimg))

    net = Conv((3,3), 16=>32, pad=1, stride=2) |> gpu


    testimg = randn(Float32, 32,32, 16,1) |> gpu;
    size(testimg)
    # FIXME ResidualBlock does not work on GPU
    conv_block((3,3), 16=>16, 1)
    identity_block((3,3), 16=>16) |> gpu
    net = conv_block((3,3), 16=>16, 1) |> gpu
    net(testimg)
    gpu(net.conv_layers[1])(testimg)
    gpu(net).conv_layers[1](testimg)

    typeof(net.conv_layers[1])
    typeof(gpu(net.conv_layers[1]))

    CuArrays.cu(net)
    net
end

"""
TODO CIFAR using ResNet models
"""
function get_CIFAR_ResNet_model()
    resnet20 = resnet(3) |> gpu;
    resnet32 = resnet(5) |> gpu;
    resnet56 = resnet(9)
    resnet68 = resnet(11)

    test_CIFAR_model(resnet20)

    size(resnet32(trainX[1]))

    m = Chain(
        Conv((3,3), 3=>16, stride=(1,1), pad=1),
        BatchNorm(16),
        conv_block((3,3), 16=>16, 1),
        res_block(5, (3,3), 16=>16),
        # here should be 32,32,16
        conv_block((3,3), 16=>32, 2),
        res_block(5, (3,3), 32=>32),
        conv_block((3,3), 32=>64, 2),
        res_block(5, (3,3), 64=>64),
        MeanPool((8,8)),
        x -> reshape(x, :, size(x,4)),
        Dense(64,10),
        softmax
    ) |> gpu;

    size(trainX[1])
    size(m(trainX[1]))

    MeanPool((64))


    testimg = randn(Float32, 32,32, 16,1) |> gpu;
    size(testimg)
    net = conv_block((3,3), 16=>32, 2) |> gpu
    net(testimg)

    gpu(conv_block((3,3), 16=>16, 1))(m(trainX[1]))

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


function test_MSE()

    # x = trainX[1];
    # size(x)
    # flat = reshape(x, :, size(x,4));
    # size(flat)
    # size(reshape(flat, 28, 28, :))
    # size(ae(x))

    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
    ae = dense_AE()
    model = ae

    # Flux.params(model);

    # size(trainX)

    # FIXME mse broke forward diff on cuarray. Needs CuArrays
    # master. https://github.com/FluxML/Flux.jl/issues/848
    loss(x) = Flux.mse(model(x), x)
    loss(x) = Flux.crossentropy(model(x), x)
    loss(trainX[1]);

    mycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

    mycrossentropy(ŷ, y; ϵ=eps()) = -sum(y .* log.(ŷ.+ϵ) .* 1) * 1 // size(y)[end]
    mycrossentropy(trainX[1], trainX[1])

    size(trainY[1],2)
    size(trainX[1])[end]
    eps(trainX[1])
    eps(typeof(trainY[1]))
    eps(cnn(trainX[1]))
    eps(Float32)
    eps()


    Flux.crossentropy(trainX[1], trainX[1]);
    crossentropy(model(trainX[1]), model(trainX[1]));
    crossentropy(cnn(trainX[1]), trainY[1]);
    crossentropy(trainY[1], trainY[1]);

    -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)

    size(trainX[1], 2)

    -sum(cnn(trainX[1]) .* log.(trainY[1]) .* 1);

    sum(log.(trainX[1]))

    log.(cnn(trainX[1]))
    log.(trainX[1])
    trainX[1]

    sample_and_view(trainX[1], trainY[1])

    -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)

    typeof(cnn(trainX[1]))
    typeof(trainY[1])

    evalcb = throttle(() -> @show(loss(valX[1])), 5)
    opt = ADAM()

    Flux.train!(loss, Flux.params(model), zip(trainX), opt, cb = evalcb)

    # decoder = Dense(32, 28^2, relu)
    # model = Chain(encoder, decoder) |> gpu
    # model(trainX[1]);
    # encoder(trainX[1]);
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
function test_concate()
    # FIXME concatenation works
    nx = cat(d[1], x_adv, dims=4)
    ny = gpu(cat(d[2], d[2], dims=2))
    loss(nx, ny)
end

adv_accuracy(x, y) = begin
    x_adv = attack_fn(model, loss, x, y)
    accuracy(x_adv, y)
end
adv_loss(x, y) = begin
    x_adv = attack_fn(model, loss, x, y)
    loss(x_adv, y)
end
