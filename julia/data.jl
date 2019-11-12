using Flux
# Zygote
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

export load_MNIST, load_CIFAR10


using MLDatasets

function test()
    train_x, train_y = MLDatasets.MNIST.traindata();
    test_x,  test_y  = MLDatasets.MNIST.testdata();
    train_x[:,:,1]
    train_y[1]
    train_x, train_y = MLDatasets.CIFAR10.traindata();
    test_x,  test_y  = MLDatasets.CIFAR10.testdata();
    train_x[:,:,:,1]
    train_y[1]
end

mutable struct DataSetIterator
    raw_x::AbstractArray
    raw_y::AbstractArray
    index::UInt64
    batch_size::UInt64
    DataSetIterator(x,y,batch_size) = new(x,y,0,batch_size)
end

function next_batch!(ds::DataSetIterator)
    if (ds.index + 1) * ds.batch_size > size(ds.raw_x)[end]
        ds.index = 0
    end
    if ds.index == 0
        indices = shuffle(1:size(ds.raw_x)[end])
        ds.raw_x = ds.raw_x[:,:,:,indices]
        ds.raw_y = ds.raw_y[:,indices]
    end
    a = ds.index * ds.batch_size + 1
    b = (ds.index + 1) * ds.batch_size

    ds.index += 1
    # FIXME use view instead of copy
    # FIXME properly slice with channel-last format
    return ds.raw_x[:,:,:,a:b], ds.raw_y[:,a:b]
end

function my_convert2image(x)
    # DEPRECATED
    #
    # FIXME the digits are horizontal-major instead of vertical-major, which is
    # very weird, create problem for my visualization. I have to use
    # MLDatasets.MNIST.convert2image, which turns floating array into white
    # background digits. WTF. I would want to just rotate it.
    reshape(MLDatasets.MNIST.convert2image(reshape(x, 28,28,:)), 28,28,1,:)
end
function load_MNIST_ds(;batch_size)
    # Ideally, it takes a batch size, and should return an iterator that repeats
    # infinitely, and shuffle before each epoch. The data should be moved to GPU
    # on demand. I prefer to just move online during training.
    train_x, train_y = MLDatasets.MNIST.traindata();
    test_x,  test_y  = MLDatasets.MNIST.testdata();
    # reshape to add channel, and onehot y encoding
    #
    # I'll just permute dims to make it column major
    train_ds = DataSetIterator(reshape(permutedims(train_x, [2,1,3]), 28,28,1,:),
                               onehotbatch(train_y, 0:9), batch_size);
    # FIXME test_ds should repeat only once
    # TODO add iterator interface, for x,y in ds end
    test_ds = DataSetIterator(reshape(permutedims(test_x, [2,1,3]), 28,28,1,:),
                              onehotbatch(test_y, 0:9), batch_size);
    # x, y = next_batch!(train_ds);
    return train_ds, test_ds
end

function test()
    train_ds, test_ds = load_MNIST_ds(128)
    x, y = next_batch!(train_ds)
    size(x)
    size(y)
    train_ds.raw_y[1:10]
    for i=1:1000
        next_batch!(train_ds)
        @show train_ds.index
    end
    60000 / 32
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
