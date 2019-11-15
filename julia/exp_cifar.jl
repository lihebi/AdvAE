include("data.jl")
include("model.jl")
include("train.jl")
include("exp.jl")

function param_count(model)
    ps = Flux.params(model)
    res = 0
    for p in keys(ps.params.dict)
        res += prod(size(p))
    end
    res / 1000 / 1000
end

function test_param_count()
    param_count(get_CIFAR_CNN_model())
    param_count(resnet(20))
    param_count(resnet(32))
    param_count(resnet(44))
    param_count(resnet(56))
    param_count(resnet(110))
    param_count(resnet(1202))
    # The WRN paper has the wrong param count, thus probably the wrong implementation
    param_count(WRN(16,1))
    param_count(WRN(16,4))
    param_count(WRN(16,8))
    param_count(WRN(28,4))
    # This is too wide and consume too much memory (even batch size 32 won't work)
    param_count(WRN(28,10))
    param_count(WRN(40,1))
    param_count(WRN(40,8))
end

# TODO learning rate?
function test_CIFAR_ds()
    ds, test_ds = load_CIFAR10_ds(batch_size=128);
    x, y = next_batch!(ds) |> gpu;

    model = get_CIFAR_CNN_model()[1:end-1]
    model = resnet(20)[1:end-1]
    # model = resnet(32)[1:end-1]
    # model = resnet(56)
    # model = resnet(68)

    # model = WRN(28, 10)[1:end-1] |> gpu;
    # model = WRN(16, 2)[1:end-1] |> gpu;

    # model parameters
    param_count(model)

    model(x)

    opt = ADAM(1e-3)

    @epochs 5 train!(model, opt, ds, print_steps=50)
    # TODO data augmentation
    # TODO per_image_standardization

    accuracy_with_logits(model(x), y)

end

function exp_res_model(expID, model_fn, total_steps)
    # 390 batch / epoch
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)
    expID = "CIFAR/model-$expID"
    nat_exp_helper(expID, 1e-3, total_steps, model_fn, ds_fn)
end

# TODO test different resnet models
function test_res_model(model)
    exp_res_model("resnet20", () -> resnet(20), 3000)
    exp_res_model("resnet32", () -> resnet(32), 1000)
    exp_res_model("resnet56", () -> resnet(56), 1000)
    exp_res_model("resnet110", () -> resnet(110), 1000)
    exp_res_model("WRN-16-2", () -> WRN(16,2), 1000)
    exp_res_model("WRN-16-4", () -> WRN(16,4), 1000)
    exp_res_model("WRN-16-8", () -> WRN(16,8), 1000)
    exp_res_model("WRN-28-4", () -> WRN(28,4), 1000)
end

