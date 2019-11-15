using Logging
using LoggingExtras: TeeLogger
using TensorBoardLogger
using Dates

MNIST_pretrained_model_file = "trained/pretrain-MNIST.bson"
MNIST_model_fn = get_Madry_model
MNIST_ds_fn = () -> load_MNIST_ds(batch_size=50)
MNIST_train_fn = (args...) -> train!(args...)

CIFAR10_pretrained_model_file = "trained/pretrain-CIFAR10.bson"
CIFAR10_model_fn = () -> resnet(20)
CIFAR10_ds_fn = () -> load_CIFAR10_ds(batch_size=128)
CIFAR10_train_fn = (args...) -> @epochs 3 train!(args...)

function get_pretrained_MNIST_model()
    get_pretrained_model(MNIST_pretrained_model_file, MNIST_model_fn, MNIST_ds_fn, MNIST_train_fn)
end

function get_pretrained_CIFAR10_model()
    get_pretrained_model(CIFAR10_pretrained_model_file, CIFAR10_model_fn, CIFAR10_ds_fn, CIFAR10_train_fn)
end

# I'm going to have one and only one MNIST and CIFAR model, so pretrained model
# only takes dataset name.
function get_pretrained_model(model_file, model_fn, ds_fn, train_fn)
    if isfile(model_file)
        @info "Already trained. Loading .."
        # load the model
        @load model_file weights
        @info "loading weights into model"
        model = model_fn()[1:end-1]
        Flux.loadparams!(model, weights)
        return model
    else
        @info "Pre-training CNN model .."
        model = model_fn()[1:end-1]

        ds, test_ds = ds_fn();
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        @info "trainig .."
        train_fn(model, opt, ds)
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
        model = model_fn()[1:end-1]
        Flux.loadparams!(model, weights)
    else
        @info "Starting from scratch .."
        # FIXME load a pretrained model here
        model = model_fn()[1:end-1]
        from_steps = 1
    end
    return model, from_steps
end

function MNIST_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    # This put log and saved model into MNIST subfolder
    expID = "MNIST/" * expID
    adv_exp_helper(expID, lr, total_steps, λ,
                   MNIST_model_fn, MNIST_ds_fn, get_pretrained_MNIST_model,
                   pretrain=pretrain)
end

function CIFAR10_exp_helper(expID, lr, total_steps, λ; pretrain=false)
    expID = "CIFAR10/" * expID
    adv_exp_helper(expID, lr, total_steps, λ,
                   CIFAR10_model_fn, CIFAR10_ds_fn, get_pretrained_CIFAR10_model,
                   pretrain=pretrain)
end

function adv_exp_helper(expID, lr, total_steps, λ,
                        model_fn, ds_fn, pretrained_fn;
                        pretrain=false)
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
    if from_steps == 1 & pretrain
        model = pretrained_fn()
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

    save_cb = create_save_cb(model_file, model)
    test_cb = create_adv_test_cb(model, test_ds, logger=test_logger)

    advtrain!(model, opt, attack_PGD_k(40), ds,
              train_steps=total_steps,
              from_steps=from_steps,
              print_steps=20,
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

    save_cb = create_save_cb(model_file, model)
    test_cb = create_test_cb(model, test_ds, logger=test_logger)

    train!(model, opt, ds,
           train_steps=total_steps,
           from_steps=from_steps,
           print_steps=20,
           logger=logger,
           save_cb=save_cb,
           test_cb=test_cb)
end

