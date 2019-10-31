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

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function load_MNIST(; use_batch=true, val_split=0.1)
    imgs = MNIST.images();
    labels = MNIST.labels();
    # size(imgs)

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

    # println("3")
    # batch_size = 128
    # mb_idxs = partition(1:length(train_imgs), batch_size)
    # train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs];

    # Prepare test set as one giant minibatch:
    # test_imgs = MNIST.images(:test);
    # test_labels = MNIST.labels(:test);
    # test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs));

    X, Y = toXY(MNIST.images(:train), MNIST.labels(:train))
    X2, Y2 = toXY(MNIST.images(:test), MNIST.labels(:test))

    N = length(X)
    # val_split = 0.1
    mid = round(Int, N * (1 - val_split))
    N2 = length(X2)

    if use_batch
        # TODO fixed batch_num 100 here
        trainX = gpu.([cat(X[i]..., dims = 4) for i in partition(1:mid, 100)]);
        trainY = gpu.([Y[:,i] for i in partition(1:mid, 100)]);
        
        valX = [cat(X[i]..., dims = 4) for i in partition(mid+1:N, 100)] .|> gpu;
        valY = [Y[:,i] for i in partition(mid+1:N, 100)] .|> gpu;

        testX = [cat(X2[i]..., dims = 4) for i in partition(1:N2, 100)] .|> gpu;
        testY = [Y2[:,i] for i in partition(1:N2, 100)] .|> gpu;
    else
        @warn("Warning: should use batch.")
        trainX = X[1:mid]
        trainY = Y[1:mid]
        valX = X[mid+1:N]
        valY = Y[mid+1:N]
        testX = X2
        testY = Y2
    end
    
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

Each trainX is batch of 100.

TODO I'm not going to move it onto GPU yet.

"""
function load_CIFAR10(; use_batch=true, val_split=0.1)
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

    X, Y = toXY(trainimgs(CIFAR10));

    N = length(X)
    # TODO use random index
    mid = round(Int, N * (1 - val_split))
    # println("N=$(N), mid=$(mid)")

    # There is no testimgs. the valimgs is the test.
    # imgs3 = testimgs(CIFAR10);
    X2, Y2 = toXY(valimgs(CIFAR10));
    N2 = length(X2);
    
    if use_batch
        # TODO fixed batch_num 100 here
        trainX = gpu.([cat(X[i]..., dims = 4) for i in partition(1:mid, 100)]);
        trainY = gpu.([Y[:,i] for i in partition(1:mid, 100)]);

        # size(trainX)            # (490,)
        # size(trainX[1])         # (32, 32, 3, 100)
        # size(trainY[1])         # (10, 100)
        
        valX = gpu.([cat(X[i]..., dims = 4) for i in partition(mid+1:N, 100)]);
        valY = gpu.([Y[:,i] for i in partition(mid+1:N, 100)]);

        testX = gpu.([cat(X2[i]..., dims = 4) for i in partition(1:N2, 100)]);
        testY = gpu.([Y2[:,i] for i in partition(1:N2, 100)]);
    else
        trainX = X[1:mid]
        trainY = Y[1:mid]
        valX = X[mid+1:N]
        valY = Y[mid+1:N]
        testX = X2
        testY = Y2
        # FIXME ??
        # val_indices = collect(mid+1:N);
        # valX = cat(X[val_indices]..., dims = 4);
        # valY = Y[:, val_indices];
    end
    
    return (trainX, trainY), (valX, valY), (testX, testY)
end


function test_load_CIFAR10()
    data = load_CIFAR10();
    (trainX, trainY), (valX, valY), (testX, testY) = data;

    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10();
    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10(use_batch=true);
    size(trainX)
    size(trainY)
    size(testY)
end

