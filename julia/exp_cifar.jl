include("data.jl")
include("model.jl")
include("train.jl")
include("exp.jl")



function CIFAR10_pretrain_fn(m)
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)
    function train_fn(model)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        # FIXME record the pretrain as well?
        # FIXME I might want to print the test acc at the end
        #
        # I actually want to record the pretrain process
        # TODO test this
        #
        # FIXME Remove if exist?
        logger = TBLogger("tensorboard_logs/$expID/train", tb_append, min_level=Logging.Info)
        test_logger = TBLogger("tensorboard_logs/$expID/test", tb_append, min_level=Logging.Info)
        test_cb = create_test_cb(model, test_ds,
                                 logger=test_logger,
                                 test_per_steps=50,
                                 test_run_steps=4)
        opt = ADAM(1e-3);
        train!(model, opt, ds,
               train_steps=2000, print_steps=20,
               logger=logger, test_cb=test_cb)
        opt = ADAM(1e-4);
        train!(model, opt, ds,
               train_steps=4000, from_steps=2000, print_steps=20,
               logger=logger, test_cb=test_cb)
    end
    maybe_train("trained/pretrain-CIFAR10.bson", m, train_fn)
end

function CIFAR10_pretrained_fn()
    # test WRN(28,10) on 2080 Ti
    model_fn = () -> WRN(16,4)
    m = model_fn()
    CIFAR10_pretrain_fn(m)
end


function CIFAR10_AE_pretrain_fn(m)
    ds_fn = () -> load_CIFAR10_ds(batch_size=50)
    function train_fn(model)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        aetrain!(model, opt, ds, train_steps=5000, print_steps=100)
    end
    maybe_train("trained/pretrain-CIFAR10-ae.bson", m, train_fn)
end

function CIFAR10_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    expID = "CIFAR10/" * expID

    model_fn = () -> WRN(16,4)
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)

    adv_exp_helper(expID, lr, total_steps, λ,
                   model_fn, ds_fn,
                   if pretrain CIFAR10_pretrain_fn else (a)->a end,
                   print_steps=2, save_steps=20,
                   test_per_steps=20, test_run_steps=2,
                   attack_fn=attack_CIFAR10_PGD_k(7))
end

function CIFAR10_free_exp_helper(expID, lr, total_steps)
    expID = "CIFAR10-free/" * expID
    @show expID

    model_fn = () -> WRN(16,4)
    ds_fn = () -> load_CIFAR10_ds(batch_size=50)
    free_exp_helper(expID, lr, total_steps, 8/255,
                    model_fn, ds_fn,
                    (a)->a,
                    print_steps=2, save_steps=20,
                    test_per_steps=20, test_run_steps=2,
                    test_attack_fn=attack_CIFAR10_PGD_k(7))
end

function exp_warmup()
    expID="warmup/test-$(now())"
    CIFAR10_exp_helper(expID, 1e-3, 1, 0)
end

function exp_free()
    # TODO add schedule
    CIFAR10_free_exp_helper("test-$(now())", 1e-3, 5000)
end



function CIFAR10_dyattack_exp_helper(expID, lr, attack_fn, total_steps; pretrain)
    # This put log and saved model into MNIST subfolder
    if pretrain
        expID = "CIFAR10-dyattack-pretrain/" * expID
    else
        expID = "CIFAR10-dyattack/" * expID
    end

    model_fn = () -> WRN(16,4)
    ds_fn = () -> load_CIFAR10_ds(batch_size=50)

    adv_exp_helper(expID, lr, total_steps, 0,
                   model_fn, ds_fn,
                   if pretrain CIFAR10_pretrain_fn else (a)->a end,
                   print_steps=2, save_steps=20,
                   test_per_steps=20, test_run_steps=2,
                   attack_fn=attack_fn,
                   test_attack_fn=attack_CIFAR10_PGD_k(7))
end

function exp_dyattack(schedule; pretrain=false)
    # TODO IMPORTANT I'll definitely need to (HEBI: record the training time)
    expID = replace("$schedule", " "=>"")
    @show expID
    for m in schedule
        @show m
        # TODO not only k, but (HEBI: also ε and η)
        CIFAR10_dyattack_exp_helper(expID, 1e-3,
                                    attack_CIFAR10_PGD_k(m[1]),
                                    m[2],
                                    pretrain=pretrain)
    end
end

function exp_dymix(expID)
    expID = replace("$schedule", " "=>"")
    @show expID
    for m in schedule
        @show m
        steps = m[2]
        λ = m[1]
        # FIXME when λ is large, reduce lr
        CIFAR10_exp_helper(expID, 1e-3, steps, λ)
    end
end

function exp_lrdecay(schedule)
    expID = replace("$schedule", " "=>"")
    @show expID
    for m in schedule
        lr = m[1]
        steps = m[2]
        CIFAR10_exp_helper("itadv-schedule/"*expID, lr, steps, 0)
    end
end

function tmp()
    # dyattack
    exp_dyattack((0=>200, 1=>400, 2=>600, 3=>800, 4=>1000, 5=>1200, 6=>1400, 7=>2000))
    exp_dyattack((0=>200, 1=>400, 2=>600, 3=>800, 4=>1000, 5=>1200, 6=>1400, 7=>2000), pretrain=true)
    exp_dyattack((1=>500, 2=>1000, 3=>1500, 4=>2000, 5=>2500, 6=>3000, 7=>4000), pretrain=true)
    # TODO (HEBI: add lr schedule)
    exp_dyattack((1=>1000, 2=>2000, 3=>3000, 4=>4000, 5=>5000, 6=>6000, 7=>7000), pretrain=true)
    # dymix
    exp_dymix((5=>1000, 4=>2000, 3=>3000, 2=>4000, 1=>5000, 0=>6000))
    # itadv baseline
    #
    # CIFAR10_exp_helper("baseline:itadv-1e-3", 1e-3, 2000, 0)
    # CIFAR10_exp_helper("baseline:f1-1e-3", 1e-3, 2000, 1)
    #
    # Maybe reuse the first part?
    exp_lrdecay((1e-3=>2000, 2e-4=>3000, 5e-5=>4000))

    # TODO run this OVERNIGHT
    exp_lrdecay((1e-3=>10000, 2e-4=>15000, 5e-5=>20000))
end


# TODO log decoded images to tensorboard
# TODO record rec loss
function CIFAR10_advae_exp_helper(expID, lr, total_steps; λ=0, γ=0, β=1, pretrain=false)
    # This put log and saved model into MNIST subfolder
    expID = "CIFAR10-advae/" * expID

    # ae_model_fn = cifar10_AE
    ae_model_fn = dunet
    # ae_model_fn = cifar10_deep_AE
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)

    advae_exp_helper(expID, lr, total_steps,
                     ae_model_fn, ds_fn,
                     if pretrain CIFAR10_AE_pretrain_fn else (a)->a end,
                     λ=λ, γ=γ, β=β,
                     CIFAR10_pretrained_fn,
                     print_steps=2, save_steps=4,
                     test_per_steps=20, test_run_steps=2,
                     attack_fn=attack_CIFAR10_PGD_k(7))
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


##############################
## AdvAE experiments
##############################

function exp_advae_f0(lr, total_steps)
    expID = "f0-$lr"
    CIFAR10_advae_exp_helper(expID, lr, total_steps, λ=0)
end

function exp_advae_pretrain(lr, total_steps)
    expID = "pretrain-$lr"
    CIFAR10_advae_exp_helper(expID, lr, total_steps, λ=0, pretrain=true)
end

function exp_advae_f1(lr, total_steps)
    expID = "f1-$lr"
    CIFAR10_advae_exp_helper(expID, lr, total_steps, λ=1)
end

function exp_advae_f01(lr, total_steps)
    expID = "f01-$lr"
    CIFAR10_advae_exp_helper(expID, lr, total_steps, λ=0, γ=1)
end

function exp_advae_test(lr, total_steps; expID="test-$(now())")
    @show expID
    CIFAR10_advae_exp_helper(expID, lr, total_steps, λ=3, γ=3, β=1)
end


# FIXME parametrize cnn_model_fn and ae_model_fn
function test()
    exp_advae_f0(1e-3, 2000)
    exp_advae_f0(1e-4, 2000)
    exp_advae_pretrain(1e-4, 2000)
    exp_advae_pretrain(1e-3, 3000)
    exp_advae_f1(1e-3, 2000)
    exp_advae_f1(2e-3, 2000)
    exp_advae_f01(1e-3, 2000)

    exp_advae_test(1e-3, 1000, expID="test-2019-11-17T13:28:29.278")
    exp_advae_test(4e-4, 2000, expID="test-2019-11-18T00:15:19.833")
    exp_advae_test(4e-4, 2000)
end
