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
    param_count(get_CIFAR10_CNN_model())
    param_count(resnet(20))
    param_count(resnet(32))
    param_count(resnet(44))
    param_count(resnet(56))
    param_count(resnet(110))
    param_count(resnet(1202))
    # The WRN paper has the wrong param count, thus probably the wrong implementation
    param_count(WRN(16,1))
    # (16,4) is probably the best for me, 88% acc
    param_count(WRN(16,4))
    param_count(WRN(28,4))
    param_count(WRN(16,8))
    # This is too wide and consume too much memory (even batch size 32 won't work)
    param_count(WRN(28,10))
    param_count(WRN(40,1))
    param_count(WRN(40,8))
end

function exp_res_model(expID, model_fn, total_steps; lr=1e-3)
    # 390 batch / epoch
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)
    expID = "CIFAR10/model-$expID"
    nat_exp_helper(expID, lr, total_steps, model_fn, ds_fn)
end

function exp_res_model_with_lr(expID, model_fn)
    # Implementing learing rate schedule: the optmizer scheduler uses the
    # implicit opt.apply steps. I do not want to save the optimier, thus I'll
    # need to advance opt steps, which is not clean. Thus, the lr schedule won't
    # be implemented inside opt. Instead, I'm implementing it in experiment
    # level script, i.e. here.
    #
    # Original paper: start with lr 0.1, divide it by 10 at 32k and 48k
    # iterations, and terminate training at 64k iterations. But I think the data
    # is [0,255], so the starting rate is 0.1/255 = 4e-4
    exp_res_model(expID, model_fn, 2000, lr=1e-3)
    exp_res_model(expID, model_fn, 4000, lr=1e-4)
    # FIXME This is not changing anything, is it too small?
    exp_res_model(expID, model_fn, 6000, lr=5e-6)
    # exp_res_model(expID, model_fn, 8000, lr=1e-6)
end

# TODO per_image_standardization, Flux has LayerNorm layer, but might not be
# useful when using BN layers.
#
# TODO data augmentation
#
# Original resnet paper: We follow the simple data augmentation in [24] for
# training: 4 pixels are padded on each side, and a 32×32 crop is randomly
# sampled from the padded image or its horizontal flip

# testing different resnet models, and choose one
function test_res_model()
    exp_res_model_with_lr("resnet20", () -> resnet(20))
    exp_res_model_with_lr("resnet32", () -> resnet(32))
    exp_res_model_with_lr("resnet56", () -> resnet(56))
    exp_res_model_with_lr("resnet110", () -> resnet(110))
    exp_res_model_with_lr("WRN-16-2", () -> WRN(16,2))
    exp_res_model_with_lr("WRN-16-4", () -> WRN(16,4))
    exp_res_model_with_lr("WRN-28-4", () -> WRN(28,4))
    exp_res_model_with_lr("WRN-16-8", () -> WRN(16,8))
end




function exp_itadv(lr, total_steps)
    expID = "f0-$lr"
    CIFAR10_exp_helper(expID, lr, total_steps, 0)
end

function exp_pretrain(lr, total_steps)
    expID = "pretrain-$lr"
    CIFAR10_exp_helper(expID, lr, total_steps, 0, pretrain=true)
end

function exp_f1(lr, total_steps)
    expID = "f1-$lr"
    CIFAR10_exp_helper(expID, lr, total_steps, 1)
end

function exp_itadv()
    # 200 steps costs 15 min
    # 2000 steps costs 2.5h
    exp_itadv(5e-2, 100)
    exp_itadv(1e-2, 100)
    exp_itadv(5e-3, 200)
    # this works
    exp_itadv(1e-3, 500)
    exp_itadv(5e-4, 500)
    exp_itadv(1e-4, 500)
    # exp_itadv(5e-5, 1000)
    # exp_itadv(1e-5, 1000)
end

function exp_tmp()
    exp_f1(5e-3, 500)
    exp_f1(3e-3, 2000)
    exp_f1(1e-3, 2000)
    exp_itadv(1e-3, 2000)
    exp_itadv(5e-4, 2000)
    exp_pretrain(1e-3, 500)
end

function exp_pretrain()
    exp_pretrain(5e-2, 100)
    exp_pretrain(1e-2, 100)
    exp_pretrain(5e-3, 200)
    exp_pretrain(1e-3, 500)
    exp_pretrain(5e-4, 500)
    exp_pretrain(1e-4, 500)
end

function exp_f1()
    exp_f1(5e-2, 100)
    exp_f1(1e-2, 100)
    exp_f1(5e-3, 200)

    exp_f1(1e-3, 500)
    exp_f1(5e-4, 500)
    exp_f1(1e-4, 500)
end

function exp()
    exp_itadv()
    exp_pretrain()
    exp_f1()
    exp_tmp()
end
