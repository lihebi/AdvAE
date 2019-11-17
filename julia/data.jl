# DEBUG I'm trying to set this env before any possible using of CuArrays
#
# integer in bytes
# 7G = 7 * 1000 * 1000 * 1000
# FIXME does not work
#
# For 1070
# g=7.5
# For 2080 Ti
g=9.5
ENV["CUARRAYS_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

# this manually set works
# CuArrays.usage_limit[] = parse(Int, ENV["CUARRAYS_MEMORY_LIMIT"])
# Or better call the init function
# CuArrays.__init_memory__()
# Or
# CuArrays.__init__()
# check status
# CuArrays.memory_status()

using CuArrays

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

# for padarray and Fill
using Images

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

# There is a data loader PR: https://github.com/FluxML/Flux.jl/pull/450
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

function compute_mean_var()
    ds, test_ds = load_CIFAR10_ds(batch_size=128)
    # FIXME do I need to consider test_ds?
    train_x, train_y = MLDatasets.CIFAR10.traindata();
    test_x,  test_y  = MLDatasets.CIFAR10.testdata();
    # dim=3 is the color channel, it will return a value for each channel

    # FIXME
    # 1. the mean var of MLDatasets and Metalhead.CIFAR10 are different
    # 2. when subtracting mean, half of values will be negative?
    mean(train_x)
    mean(train_x, dims=(1,2,4))
    std(train_x, dims=(1,2,4))
    minimum(train_x)

    # NOTE: This is in back.jl
    (trainX, trainY), (valX, valY), (testX, testY) = load_CIFAR10();
    trainX = cpu.(trainX);
    size(trainX)
    size(trainX[1])
    typeof(trainX)
    mean(cat(trainX..., dims=4), dims=(1,2,4))
    std(cat(trainX..., dims=4), dims=(1,2,4))
    mean(trainX[1])
end

function flipx(p)
    function f(img)
        if rand() < p
            reverse(img, dims=2)
        else
            img
        end
    end
end

function pad_and_crop(img)
    w = size(img)[1]
    h = size(img)[2]
    # FIXME this returns OffsetArrays, with possibly negative or 0 index
    padded = padarray(img, Fill(0,(4,4,0), (4,4,0)))
    a = rand(-3:4)
    b = rand(-3:4)
    # @show a
    # @show b
    # convert OffsetArrays to normal array with parent()
    parent(padded[a:a+w-1, b:b+h-1, :])
end

function augment_one(img)
    pad_and_crop(flipx(0.5)(img))
end

struct Augment end
Flux.@treelike Augment

# - will this apply different random op to images in a batch? Yes.
# - can this apply to batch automatically? NO!
# FIXME performance, and also on CPU/GPU?
function (a::Augment)(x::AbstractArray)
    length(size(x)) == 4 || error("Dim of input must be 4, got $(length(size(x)))")
    res = map(collect(1:size(x)[end])) do i
        augment_one(x[:,:,:,i])
    end
    cat(res..., dims=4)
end

function Base.show(io::IO, l::Augment)
  print(io, "Augment()")
end

function test_augment()
    # padarray

    ds, test_ds = load_CIFAR10_ds(batch_size=16);
    x,y = next_batch!(ds);

    # some testing
    img = x[:,:,:,1];
    sample_and_view(img)
    sample_and_view(reverse(img, dims=2))
    sample_and_view(pad_and_crop(flipx(0.5)(img)))

    # testing augment layer
    aug = Augment()
    sample_and_view(x)
    sample_and_view(aug(x))
    size(aug(x))
end

function test()
    ds, test_ds = load_MNIST_ds(batch_size=128);
    ds.nbatch
    test_ds.nbatch
    x, y = next_batch!(ds);
    size(x)
    size(y)
    sample_and_view(x)
    # size(x)
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
        error("Image size $(size(x)[1]) not correct size. Currently support 28 or 32.")
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


