using Logging
using LoggingExtras: TeeLogger
using TensorBoardLogger
using Dates

# I'm going to have one and only one MNIST and CIFAR model, so pretrained model
# only takes dataset name.
function maybe_train(model_file, model, train_fn)
    if isfile(model_file)
        @info "Already trained. Loading .."
        # load the model
        @load model_file weights
        @info "loading weights into model"
        Flux.loadparams!(model, weights)
        return model
    else
        @info "Pre-training .."
        train_fn(model)
        @info "saving .."
        @save model_file weights=Tracker.data.(Flux.params(cpu(model)))
        return model
    end
end

function maybe_load(model_file, model_fn)
    if isfile(model_file)
        @info "Loading from $model_file .."
        # During load, there is no key=value, but the variable name shall match
        # the key when saving, thus the order is not important. During saving,
        # it can be key=value pair, or just the variable, but cannot be x.y
        @load model_file weights from_steps
        @info "loading weights"
        # NOTE: add 1 as starting step
        from_steps += 1
        model = model_fn()
        Flux.loadparams!(model, weights)
    else
        @info "Starting from scratch .."
        # FIXME load a pretrained model here
        model = model_fn()
        from_steps = 1
    end
    return model, from_steps
end

function MNIST_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    # This put log and saved model into MNIST subfolder
    expID = "MNIST/" * expID

    model_fn = () -> get_Madry_model()[1:end-1]
    ds_fn = () -> load_MNIST_ds(batch_size=50)

    function train_fn(model)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        train!(model, opt, ds, train_steps=5000, print_steps=100)
    end
    pretrain_fn(model) = maybe_train(pretrained_model_file, model, train_fn)

    adv_exp_helper(expID, lr, total_steps, λ,
                   model_fn, ds_fn,
                   if pretrain pretrain_fn else (a)->a end,
                   print_steps=20, save_steps=40,
                   test_per_steps=100, test_run_steps=20,
                   attack_fn=attack_PGD_k(40))
end

function CIFAR10_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    expID = "CIFAR10/" * expID

    model_fn = () -> WRN(16,4)[1:end-1]
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)

    function train_fn(model)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        # FIXME I might want to print the test acc at the end
        # FIXME record the pretrain as well?
        opt = ADAM(1e-3);
        train!(model, opt, ds, train_steps=2000, print_steps=20)
        opt = ADAM(1e-4);
        train!(model, opt, ds, train_steps=4000, from_steps=2000, print_steps=20)
    end

    pretrain_fn(model) = maybe_train(pretrained_model_file, model, train_fn)

    adv_exp_helper(expID, lr, total_steps, λ,
                   model_fn, ds_fn,
                   if pretrain pretrain_fn else (a)->a end,
                   print_steps=2, save_steps=4,
                   test_per_steps=20, test_run_steps=2,
                   attack_fn=CIFAR10_PGD_7)
end

function adv_exp_helper(expID, lr, total_steps, λ,
                        model_fn, ds_fn, pretrained_fn;
                        attack_fn,
                        print_steps, save_steps,
                        test_per_steps, test_run_steps)
    model_file = "trained/$expID.bson"
    mkpath(dirname(model_file))

    # TODO record training @time? I need to use the average time instead of
    # accumulated time, because accumulated time introduces states that needs to
    # be saved. This is not very urgent.
    #
    # FIXME what should be my batch_size?
    ds, test_ds = ds_fn();
    x, y = next_batch!(ds) |> gpu;

    model, from_steps = maybe_load(model_file, model_fn)
    if from_steps == 1
        model = pretrained_fn(model)
    end

    @info "Progress" total_steps from_steps
    # stops here to avoid model warming up overhead
    if from_steps > total_steps return end

    logger = TBLogger("tensorboard_logs/$expID/train", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/$expID/test", tb_append, min_level=Logging.Info)

    # warm up the model
    model(x)

    # FIXME opt states
    opt = ADAM(lr);

    save_cb = create_save_cb(model_file, model, save_steps=save_steps)
    test_cb = create_adv_test_cb(model, test_ds,
                                 logger=test_logger,
                                 attack_fn=attack_fn,
                                 test_per_steps=test_per_steps, test_run_steps=test_run_steps)

    advtrain!(model, opt, attack_fn, ds,
              train_steps=total_steps,
              from_steps=from_steps,
              print_steps=print_steps,
              λ=λ,
              logger=logger,
              save_cb=save_cb,
              test_cb=test_cb)

end

function nat_exp_helper(expID, lr, total_steps,
                        model_fn, ds_fn)
    model_file = "trained/$expID.bson"
    mkpath(dirname(model_file))

    ds, test_ds = ds_fn();
    x, y = next_batch!(ds) |> gpu;

    model, from_steps = maybe_load(model_file, model_fn)

    @info "Progress" total_steps from_steps
    # stops here to avoid model warming up overhead
    if from_steps > total_steps return end

    logger = TBLogger("tensorboard_logs/$expID/train", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/$expID/test", tb_append, min_level=Logging.Info)

    # warm up the model
    model(x)

    # FIXME opt states
    opt = ADAM(lr);
    #
    # TODO WRN used SGD with Nesterov momentum 0.9 and weight decay 0.0005 start
    # lr=0.1, drop by 0.2 at 60,120,160 epoch
    #
    # FIXME weight decay?
    #
    # Nesterov(lr, 0.9)
    # Momentum(lr, 0.9)

    save_cb = create_save_cb(model_file, model, save_steps=40)
    test_cb = create_test_cb(model, test_ds, logger=test_logger,
                             test_per_steps=100, test_run_steps=20)

    train!(model, opt, ds,
           # DEBUG adding cifar augment. It does not seem to provide any improvement
           # augment=Augment(),
           train_steps=total_steps,
           from_steps=from_steps,
           print_steps=20,
           logger=logger,
           save_cb=save_cb,
           test_cb=test_cb)
end
