using ProgressMeter

# glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
# glorot_normal(dims...) = randn(Float32, dims...) .* sqrt(2.0f0/sum(dims))
#
# https://github.com/FluxML/Flux.jl/issues/442

_nfan(dims...) = prod(dims[1:end-2]) * sum(dims[end-1:end])
my_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/_nfan(dims...))
myConv(args...; kwargs...) = Conv(args..., init=my_glorot_uniform; kwargs...)
myConvTranspose(args...; kwargs...) = ConvTranspose(args..., init=my_glorot_uniform; kwargs...)

function param_count(model)
    ps = Flux.params(model)
    res = 0
    for p in keys(ps.params.dict)
        res += prod(size(p))
    end
    res / 1000 / 1000
end

struct Flatten end

function (l::Flatten)(input)
    reshape(input, :, size(input, 4))
end

struct Sigmoid end
function (l::Sigmoid)(input)
    σ.(input)
end

struct ReLU end

function (l::ReLU)(input)
    relu.(input)
end

##############################
## MNIST models
##############################

function get_LeNet5()
    Chain(myConv((3,3), 1=>32, relu, pad=(1,1)),
          MaxPool((2,2)),
          myConv((3,3), 32=>64, relu, pad=(1,1)),
          MaxPool((2,2)),
          Flatten(),
          Dense(7 * 7 * 64, 200, relu),
          Dense(200, 10),
          # softmax,
          ) |> gpu
end


function get_Madry_model()
    Chain(myConv((5,5), 1=>32, pad=(2,2), relu),
          MaxPool((2,2)),
          myConv((5,5), 32=>64, pad=(2,2), relu),
          MaxPool((2,2)),
          Flatten(),
          Dense(7*7*64, 1024),
          Dense(1024, 10),
          # softmax,
          ) |> gpu
end


##############################
## MNIST AE models
##############################

function dense_AE()
    # FIXME why leakyrelu
    error("deprecated. replace anonymous functions")
    encoder = Chain(Flatten(),
                    Dense(28 * 28, 32, relu)) |> gpu
    decoder = Chain(Dense(32, 28 * 28),
                    x -> reshape(x, 28, 28, 1, :),
                    # FIXME use clamp?
                    # reshape(clamp.(x, 0, 1), 28, 28)
                    Sigmoid()) |> gpu
    Chain(encoder, decoder)
end

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
    encoder = Chain(myConv((3,3), 1=>16, pad=(1,1), relu),
                    # BatchNorm(16)
                    MaxPool((2,2)))
    decoder = Chain(myConv((3,3), 16=>16, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 16=>1, pad=(1,1)),
                    Sigmoid())
    Chain(encoder, decoder) |> gpu
end

function CNN2_AE()
    encoder = Chain(myConv((3,3), 1=>16, pad=(1,1), relu),
                    MaxPool((2,2)),
                    myConv((3,3), 16=>8, pad=(1,1), relu),
                    MaxPool((2,2)))
    decoder = Chain(myConv((3,3), 8=>8, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 8=>16, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 16=>1, pad=(1,1)),
                    Sigmoid())
    Chain(encoder, decoder) |> gpu
end



##############################
## CIFAR models
##############################

function get_CIFAR10_CNN_model()
    # DEPRECATED
    Chain(myConv((5,5), 3=>16, relu),
          MaxPool((2,2)),
          myConv((5,5), 16=>8, relu),
          MaxPool((2,2)),
          Flatten(),
          Dense(200, 10),
          # softmax
          ) |> gpu
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

# TODO resnet v2 with pre-activations. But it does not seem to change the result
# much, so not a priority
function resnet(depth)
    # num_blocks: 3,5,9,11
    # (6*num_blocks + 2)
    # 20, 32, 56, 68
    # USE 9, resnet56
    (depth - 2) % 6 == 0 || error("resnet error")
    n = convert(Int, (depth - 2) / 6)

    Chain(
        myConv((3,3), 3=>16, stride=(1,1), pad=1),
        BatchNorm(16),
        # 32,32,16

        # 2n 32x32, 16
        conv_block((3,3), 16=>16, 1),
        res_block(n-1, (3,3), 16=>16),

        conv_block((3,3), 16=>32, 2),
        res_block(n-1, (3,3), 32=>32),

        conv_block((3,3), 32=>64, 2),
        res_block(n-1, (3,3), 64=>64),

        MeanPool((8,8)),
        Flatten(),
        Dense(64,10),
        # softmax
    ) |> gpu
end

function WRN(depth, k)
    # depth = n * 6 + 4 (conv layers, not including dense)
    # w28-10: depth 28, widen factor 10
    # TODO wide resnet w28-10 (https://arxiv.org/abs/1605.07146)
    #
    # use filters = [16, 16, 32, 64] for a non-wide version
    # filters = [16, 160, 320, 640]
    (depth - 4) % 6 == 0 || error("WRN error")
    n = convert(Int, (depth - 4) / 6)
    filters = [16, 16*k, 32*k, 64*k]
    Chain(
        myConv((3,3), 3=>16, stride=(1,1), pad=1),
        BatchNorm(16),

        conv_block((3,3), 16=>filters[2], 1),
        res_block(n, (3,3), filters[2]=>filters[2]),

        conv_block((3,3), filters[2]=>filters[3], 2),
        res_block(n, (3,3), filters[3]=>filters[3]),

        conv_block((3,3), filters[3]=>filters[4], 2),
        res_block(n, (3,3), filters[4]=>filters[4]),

        MeanPool((8,8)),
        Flatten(),
        Dense(filters[4],10),
        # softmax
    ) |> gpu
end

##############################
## CIFAR10 AE models
##############################

function cifar10_AE()
    encoder = Chain(myConv((3,3), 3=>64, pad=(1,1)),
                    BatchNorm(64),
                    ReLU(),
                    MaxPool((2,2)),
                    myConv((3,3), 64=>32, pad=(1,1)),
                    BatchNorm(32),
                    ReLU(),
                    MaxPool((2,2)),
                    myConv((3,3), 32=>16, pad=(1,1)),
                    BatchNorm(16),
                    ReLU(),
                    MaxPool((2,2)))
    decoder = Chain(myConv((3,3), 16=>16, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 16=>32, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 32=>64, pad=(1,1), relu),
                    upsample,
                    myConv((3,3), 64=>3, pad=(1,1)),
                    Sigmoid())
    Chain(encoder, decoder) |> gpu
end

function cifar10_deep_AE()
    encoder = Chain(myConv((3,3), 3=>32, pad=(1,1), relu),
                    myConv((3,3), 32=>32, pad=(1,1), relu),

                    # DEBUG using conv with stride 2 instead of maxpool
                    # MaxPool((2,2)),
                    # myConv((3,3), 32=>32, stride=2, relu)
                    myConv((1,1), 32=>32, pad=0, stride=2, relu),

                    myConv((3,3), 32=>64, pad=(1,1), relu),
                    myConv((3,3), 64=>64, pad=(1,1), relu),

                    # MaxPool((2,2)),
                    # myConv((3,3), 64=>64, stride=2, relu)
                    myConv((1,1), 64=>64, pad=0, stride=2, relu),

                    myConv((3,3), 64=>128, pad=(1,1), relu),
                    myConv((3,3), 128=>128, pad=(1,1), relu),
                    # MaxPool((2,2)),
                    # myConv((3,3), 128=>256, pad=(1,1), relu),
                    # myConv((3,3), 256=>256, pad=(1,1), relu)
                    )

    decoder = Chain(
        # myConvTranspose((2,2), 256=>128, stride=2, pad=0),
        # myConv((3,3), 128=>128, pad=1, relu),

        # DEBUG using upsample instead of ConvTrans
        myConvTranspose((2,2), 128=>64, stride=2, pad=0),
        # upsample,
        myConv((3,3), 64=>64, pad=1, relu),

        myConvTranspose((2,2), 64=>32, stride=2, pad=0),
        # upsample,

        myConv((3,3), 32=>32, pad=1, relu),
        myConv((3,3), 32=>3, pad=1),
        Sigmoid())

    Chain(encoder, decoder) |> gpu
end

struct Dunet
    down_layers
    up_layers
    final_layers
end
Flux.@treelike Dunet

function (l::Dunet)(input)
    local value = copy.(input)
    local saved = []
    for i in 1:length(l.down_layers)
        value = (l.down_layers[i])(value)
        push!(saved, value)
    end
    pop!(saved)
    for i in 1:length(l.up_layers)
        prev = pop!(saved)
        value = (l.up_layers[i])(value)
        value = cat(value, prev, dims=3)
    end
    # finally add this to original input
    value = l.final_layers(value)
    value + input
end



function dunet_downlayers()
    Tuple([Chain(myConv((3,3), 3=>32, pad=(1,1), relu),
                 myConv((3,3), 32=>32, pad=(1,1), relu)),
           Chain(MaxPool((2,2)),
                 myConv((3,3), 32=>64, pad=(1,1), relu),
                 myConv((3,3), 64=>64, pad=(1,1), relu)),
           Chain(MaxPool((2,2)),
                 myConv((3,3), 64=>128, pad=(1,1), relu),
                 myConv((3,3), 128=>128, pad=(1,1), relu)),
           Chain(MaxPool((2,2)),
                 myConv((3,3), 128=>256, pad=(1,1), relu),
                 myConv((3,3), 256=>256, pad=(1,1), relu)),
           Chain(MaxPool((2,2)),
                 myConv((3,3), 256=>512, pad=(1,1), relu),
                 myConv((3,3), 512=>512, pad=(1,1), relu))])
end

function dunet_uplayers()
    # NOTE: for pad=same I should use pad=0 here
    Tuple([Chain(myConvTranspose((2,2), 512=>256, stride=2, pad=0)),
           Chain(myConv((3,3), 512=>256, pad=1, relu),
                 myConv((3,3), 256=>256, pad=1, relu),
                 myConvTranspose((2,2), 256=>128, stride=2, pad=0)),
           Chain(myConv((3,3), 256=>128, pad=1, relu),
                 myConv((3,3), 128=>128, pad=1, relu),
                 myConvTranspose((2,2), 128=>64, stride=2, pad=0)),
           Chain(myConv((3,3), 128=>64, pad=1, relu),
                 myConv((3,3), 64=>64, pad=1, relu),
                 myConvTranspose((2,2), 64=>32, stride=2, pad=0))])
end

function dunet_finallayer()
    Chain(myConv((3,3), 64=>32, pad=1, relu),
          myConv((3,3), 32=>32, pad=1, relu),
          myConv((3,3), 32=>3, pad=1),
          Sigmoid())
end

function dunet()
    down = dunet_downlayers()
    up = dunet_uplayers()
    final = dunet_finallayer()
    Dunet(down, up, final) |> gpu
end

