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
    index::Int
    batch_size::Int
    nbatch::Int
    DataSetIterator(x,y,batch_size) = new(x,y,0,batch_size, convert(Int, floor(size(x)[end]/batch_size)))
end

Base.show(io::IO, x::DataSetIterator) = begin
    println(io, "DataSetIterator:")
    println(io, "  batch size: ", x.batch_size)
    println(io, "  number of batches: ", x.nbatch)
    println(io, "  x data shape: ", size(x.raw_x))
    println(io, "  y data shape: ", size(x.raw_y))
    print(io, "  current index: ", x.index)
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

function load_CIFAR10_ds(;batch_size)
    train_x, train_y = MLDatasets.CIFAR10.traindata();
    test_x,  test_y  = MLDatasets.CIFAR10.testdata();
    train_ds = DataSetIterator(permutedims(train_x, [2,1,3,4]),
                               onehotbatch(train_y, 0:9), batch_size);
    test_ds = DataSetIterator(permutedims(test_x, [2,1,3,4]),
                              onehotbatch(test_y, 0:9), batch_size);
    return train_ds, test_ds
end

function test()
    train_ds, test_ds = load_MNIST_ds(batch_size=128);
    train_ds.nbatch
    test_ds.nbatch
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

"""
Sample up to 10 images and show the image and label. If less than 10, use all.

"""
function sample_and_view(x, y=nothing, model=nothing)
    if length(size(x)) < 4
        x = x[:,:,:,:]
    end
    if typeof(x) <: TrackedArray
        x = x.data
    end
    size(x)[1] in [28,32,56] ||
        error("Image size $(size(X)[1]) not correct size. Currently support 28 or 32.")
    num = min(size(x)[4], 10)
    @info "Showing $num images .."
    imgs = cpu(hcat([x[:,:,:,i] for i in 1:num]...))
    viewrepl(imgs)
    if y != nothing
        labels = onecold(cpu(y[:,1:num]), 0:9)
        @show labels
    end
    if model != nothing
        preds = onecold(cpu(model(x)[:,1:num]), 0:9)
        @show preds
    end
    nothing
end


