using Logging
using LoggingExtras: TeeLogger
using TensorBoardLogger
using Dates

# I'm going to have one and only one MNIST and CIFAR model, so pretrained model
# only takes dataset name.
function maybe_train(model_file, model, train_fn)
    if isfile(model_file)
        @info "Already trained. Loading .." model_file
        @load model_file model
        return gpu(model)
    else
        @info "Pre-training .."
        train_fn(model)
        @info "saving .."
        @save model_file model=cpu(model)
        return model
    end
end

function maybe_load(model_file, model_fn)
    if isfile(model_file)
        @info "Loading from $model_file .."
        @load model_file model from_steps
        model = gpu(model)
        # NOTE: add 1 as starting step
        from_steps += 1
    else
        @info "Starting from scratch .."
        model = model_fn()
        from_steps = 1
    end
    return gpu(model), from_steps
end


function MNIST_pretrain_fn(m)
    function train_fn(model)
        ds_fn = () -> load_MNIST_ds(batch_size=50)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        train!(model, opt, ds, train_steps=2000, print_steps=100)
        model
    end
    maybe_train("trained/pretrain-MNIST.bson", m, train_fn)
end

function MNIST_pretrained_fn()
    m = get_Madry_model()
    MNIST_pretrain_fn(m)
end

function MNIST_AE_pretrain_fn(m)
    ds_fn = () -> load_MNIST_ds(batch_size=50)
    function train_fn(model)
        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        aetrain!(model, opt, ds, train_steps=2000, print_steps=100)
    end
    maybe_train("trained/pretrain-MNIST-ae.bson", m, train_fn)
end

function CIFAR10_pretrain_fn(m)
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
    maybe_train("trained/pretrain-CIFAR10.bson", m, train_fn)
end

function CIFAR10_pretrained_fn()
    model_fn = () -> WRN(16,4)
    m = model_fn()
    CIFAR10_pretrain_fn(m)
end


function MNIST_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    # This put log and saved model into MNIST subfolder
    expID = "MNIST/" * expID

    model_fn = () -> get_Madry_model()
    ds_fn = () -> load_MNIST_ds(batch_size=50)

    adv_exp_helper(expID, lr, total_steps, λ,
                   model_fn, ds_fn,
                   if pretrain MNIST_pretrain_fn else (a)->a end,
                   print_steps=20, save_steps=40,
                   test_per_steps=100, test_run_steps=20,
                   attack_fn=attack_PGD_k(40))
end

function CIFAR10_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    expID = "CIFAR10/" * expID

    model_fn = () -> WRN(16,4)
    ds_fn = () -> load_CIFAR10_ds(batch_size=128)

    adv_exp_helper(expID, lr, total_steps, λ,
                   model_fn, ds_fn,
                   if pretrain CIFAR10_pretrain_fn else (a)->a end,
                   print_steps=2, save_steps=4,
                   test_per_steps=20, test_run_steps=2,
                   attack_fn=CIFAR10_PGD_7)
end

function adv_exp_helper(expID, lr, total_steps, λ,
                        model_fn, ds_fn, pretrain_fn;
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
        model = pretrain_fn(model)
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

function advae_exp_helper(expID, lr, total_steps,
                          ae_model_fn, ds_fn,
                          pretrain_fn, pretrained_cnn_fn;
                          λ, γ, β,
                          attack_fn,
                          print_steps, save_steps,
                          test_per_steps, test_run_steps)
    model_file = "trained/$expID.bson"
    @show model_file
    mkpath(dirname(model_file))

    ds, test_ds = ds_fn();
    x, y = next_batch!(ds) |> gpu;

    # here we have two models, AE and CNN
    ae, from_steps = maybe_load(model_file, ae_model_fn)
    if from_steps == 1
        ae = pretrain_fn(ae)
    end
    # load pretrained CNN
    cnn = pretrained_cnn_fn()
    # set test mode
    Flux.testmode!(cnn)
    # warm up the model
    cnn(ae(x))

    @info "Progress" total_steps from_steps
    if from_steps > total_steps return end

    logger = TBLogger("tensorboard_logs/$expID/train", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/$expID/test", tb_append, min_level=Logging.Info)

    opt = ADAM(lr);

    save_cb = create_save_cb(model_file, ae, save_steps=save_steps)
    test_cb = create_advae_test_cb(ae, cnn, test_ds,
                                   logger=test_logger,
                                   attack_fn=attack_fn,
                                   test_per_steps=test_per_steps, test_run_steps=test_run_steps)

    advae_train!(ae, cnn, opt, attack_fn, ds,
                 train_steps=total_steps,
                 from_steps=from_steps,
                 print_steps=print_steps,
                 λ=λ,
                 γ=γ,
                 logger=logger,
                 save_cb=save_cb,
                 test_cb=test_cb)

end


function MNIST_advae_exp_helper(expID, lr, total_steps; λ=0, γ=0, β=1, pretrain=false)
    # This put log and saved model into MNIST subfolder
    expID = "MNIST-advae/" * expID

    ae_model_fn = CNN_AE
    ds_fn = () -> load_MNIST_ds(batch_size=50)

    advae_exp_helper(expID, lr, total_steps,
                     ae_model_fn, ds_fn,
                     if pretrain MNIST_AE_pretrain_fn else (a)->a end,
                     MNIST_pretrained_fn,
                     λ=λ, γ=γ, β=β,
                     print_steps=20, save_steps=40,
                     test_per_steps=100, test_run_steps=20,
                     attack_fn=attack_PGD_k(40))
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
                     attack_fn=CIFAR10_PGD_7)
end
