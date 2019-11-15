using ProgressMeter

include("data.jl")

# glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
# glorot_normal(dims...) = randn(Float32, dims...) .* sqrt(2.0f0/sum(dims))
#
# https://github.com/FluxML/Flux.jl/issues/442

_nfan(dims...) = prod(dims[1:end-2]) * sum(dims[end-1:end])
my_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/_nfan(dims...))


function get_LeNet5()
    model = Chain(
        Conv((3,3), 1=>32, relu, pad=(1,1), init=my_glorot_uniform),
        MaxPool((2,2)),
        Conv((3,3), 32=>64, relu, pad=(1,1), init=my_glorot_uniform),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(7 * 7 * 64, 200, relu),
        Dense(200, 10),
        softmax,
    ) |> gpu;
    return model
end

function get_Madry_model()
    model = Chain(
        Conv((5,5), 1=>32, pad=(2,2), relu, init=my_glorot_uniform),
        # Conv((5,5), 1=>32, pad=(2,2), relu),
        MaxPool((2,2)),
        Conv((5,5), 32=>64, pad=(2,2), relu, init=my_glorot_uniform),
        # Conv((5,5), 32=>64, pad=(2,2), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(7*7*64, 1024),
        Dense(1024, 10),
        softmax,
    ) |> gpu;
    return model
end

myConv(args...; kwargs...) = Conv(args..., init=my_glorot_uniform; kwargs...)

function get_CIFAR_CNN_model()
    model = Chain(
        myConv((5,5), 3=>16, relu),
        MaxPool((2,2)),
        myConv((5,5), 16=>8, relu),
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
    push!(conv_layers, myConv(kernel_size, filters, pad=1, stride=1))
    push!(conv_layers, myConv(kernel_size, filters, pad=1, stride=1))
    push!(norm_layers, BatchNorm(filters[2]))
    push!(norm_layers, BatchNorm(filters[2]))
    ResidualBlock(Tuple(conv_layers), Tuple(norm_layers), identity)
end

function conv_block(kernel_size, filters, stride)
    local conv_layers = []
    local norm_layers = []
    # half the feature map
    push!(conv_layers, myConv(kernel_size, filters[1]=>filters[2], pad=1, stride=stride))
    push!(conv_layers, myConv(kernel_size, filters[2]=>filters[2], pad=1))
    push!(norm_layers, BatchNorm(filters[2]))
    push!(norm_layers, BatchNorm(filters[2]))
    shortcut = Chain(myConv((1,1), filters,
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

# TODO wide resnet
function resnet(num_blocks)
    # num_blocks: 3,5,9,11
    # (6*num_blocks + 2)
    # 20, 32, 56, 68
    # USE 9, resnet56
    Chain(
        myConv((3,3), 3=>16, stride=(1,1), pad=1),
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
