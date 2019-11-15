using Images

using Logging
using LoggingExtras: TeeLogger
using TensorBoardLogger
using Dates

include("train.jl")

function get_pretrained_model()
    model_file = "trained/pretrain.bson"
    if isfile(model_file)
        @info "Already trained. Loading .."
        # load the model
        @load model_file weights
        @info "loading weights into model"
        model = get_Madry_model()[1:end-1]
        Flux.loadparams!(model, weights)
        return model
    else
        @info "Pre-training CNN model .."
        model = get_Madry_model()[1:end-1]
        ds, test_ds = load_MNIST_ds(batch_size=50);
        x, y = next_batch!(ds) |> gpu;
        model(x)

        opt = ADAM(1e-3);
        @info "trainig .."
        train!(model, opt, ds)
        @info "saving .."
        @save model_file weights=Tracker.data.(Flux.params(cpu(model)))
        return model
    end
end

function maybe_load(model_file)
    if isfile(model_file)
        @info "Loading from $model_file .."
        # During load, there is no key=value, but the variable name shall match
        # the key when saving, thus the order is not important. During saving,
        # it can be key=value pair, or just the variable, but cannot be x.y
        @load model_file weights from_steps
        @info "loading weights"
        # NOTE: add 1 as starting step
        from_steps += 1
        model = get_Madry_model()[1:end-1]
        Flux.loadparams!(model, weights)
    else
        @info "Starting from scratch .."
        # FIXME load a pretrained model here
        model = get_Madry_model()[1:end-1]
        from_steps = 1
    end
    return model, from_steps
end

function exp_itadv(lr, total_steps)
    expID = "itadv-$lr"
    exp_helper(expID, lr, total_steps, 0)
end

function exp_pretrain(lr, total_steps)
    expID = "pretrain-$lr"
    exp_helper(expID, lr, total_steps, 0, pretrain=true)
end

function exp_f1(lr, total_steps)
    expID = "f1-$lr"
    exp_helper(expID, lr, total_steps, 1)
end

function exp_helper(expID, lr, total_steps, λ; pretrain=false)
    model_file = "trained/$expID.bson"
    mkpath(dirname(model_file))

    # TODO record training @time? I need to use the average time instead of
    # accumulated time, because accumulated time introduces states that needs to
    # be saved. This is not very urgent.
    #
    # FIXME what should be my batch_size?
    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    model, from_steps = maybe_load(model_file)
    if from_steps == 1 & pretrain
        model = get_pretrained_model()
    end
    @info "Progress" total_steps from_steps
    # stops here to avoid model warming up overhead
    if from_steps > total_steps return end

    logger = TBLogger("tensorboard_logs/exp-$expID/train", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/exp-$expID/test", tb_append, min_level=Logging.Info)

    # warm up the model
    model(x)

    # FIXME opt states
    opt = ADAM(lr);

    save_cb = create_save_cb(model_file, model)
    test_cb = create_test_cb(model, test_ds, logger=test_logger)

    advtrain!(model, opt, attack_PGD_k(40), ds,
              train_steps=total_steps,
              from_steps=from_steps,
              print_steps=20,
              λ=λ,
              logger=logger,
              save_cb=save_cb,
              test_cb=test_cb)

end

function evaluate(model, ds; attack_fn=(m,x,y)->x)
    xx,yy = next_batch!(ds) |> gpu

    @info "Sampling BEFORE images .."
    sample_and_view(xx, yy, model)
    @info "Sampling AFTER images .."
    adv = attack_fn(model, xx, yy)
    sample_and_view(adv, yy, model)

    @info "Testing multiple batches .."
    acc_metric = MeanMetric()
    @showprogress 0.1 "Testing..." for step in 1:10
        x,y = next_batch!(ds) |> gpu
        adv = attack_fn(model, x, y)
        acc = accuracy_with_logits(model(adv), y)
        add!(acc_metric, acc)
    end
    @show get!(acc_metric)
end


function test()
    Random.seed!(1234);

    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    model = get_Madry_model()[1:end-1]
    # model = get_LeNet5()[1:end-1]

    # Warm up the model
    model(x)

    # FIXME would this be 0.001 * 0.001?
    # FIXME decay on pleau
    # FIXME print out information when decayed
    # opt = Flux.Optimiser(Flux.ExpDecay(0.001, 0.5, 1000, 1e-4), ADAM(0.001))
    opt = ADAM(1e-4);

    logger = create_logger()
    with_logger(logger) do
        train!(model, opt, ds, print_steps=20)
    end

    @info "Adv training .."
    # custom_train!(model, opt, train_ds)

    with_logger(logger) do
        @epochs 2 advtrain!(model, opt, attack_PGD_k(40), ds, print_steps=20)
    end

    @info "Evaluating clean accuracy .."
    evaluate(model, test_ds)
    evaluate(model, test_ds, attack_fn=attack_FGSM)
    evaluate(model, test_ds, attack_fn=attack_PGD_k(40))
end

# integer in bytes
# 7G = 7 * 1000 * 1000 * 1000
# FIXME does not work
ENV["CUARRAYS_MEMORY_LIMIT"] = 7 * 1000 * 1000 * 1000

function exp_itadv()
    # FIXME should I use learning rate decay at the same time?
    #
    # NOTE: the steps must devide all metric steps, especially save steps,
    # otherwise it won't be saved correctly.
    exp_itadv(1e-1, 600)
    exp_itadv(5e-2, 600)
    exp_itadv(1e-2, 600)
    exp_itadv(5e-3, 600)
    exp_itadv(1e-3, 600)
    exp_itadv(5e-4, 1000)
    exp_itadv(4e-4, 1000)

    # converging from 3e-4, (HEBI: this is the border line)
    exp_itadv(3e-4, 6000)
    exp_itadv(1e-4, 8000)
    exp_itadv(5e-5, 4000)

    exp_itadv(1e-5, 3000)
end

function exp_pretrain()
    exp_pretrain(1e-2, 1000)
    # This does not converge, and I would expect nat acc to graduallly reduce 0.1
    exp_pretrain(1e-3, 1000)
    exp_pretrain(8e-4, 2000)
    # TODO and this is important because it is next to border-line. FIXME It
    # also does work
    exp_pretrain(7e-4, 2000)

    # This converges, and (HEBI: this is the border line)
    exp_pretrain(6e-4, 5000)
    # it is working here, but struggled
    exp_pretrain(5e-4, 5000)
    # TODO and I want to show how the worked one perform with a pretrained start
    exp_pretrain(3e-4, 3000)
end

function exp_f1()
    # TODO what about starting from pretrained?
    #
    # (HEBI: I hope this not to reach high accuracy)
    #
    # FIXME what if we just use a simple lr decay? The key point should be, no
    # matter how the lr change, the accuracy should still not reach the high
    # value. This might make more sense on CIFAR10 than MNIST. I'll need to stop
    # here and (HEBI: move to CIFAR NOW).
    #
    # this defnintely does not converge, this lr may not converge even for clean
    # train, I didn't try though.
    exp_f1(1e-2, 2000)
    # this converges, but end acc is not high, as expected
    exp_f1(8e-3, 3000)
    exp_f1(5e-3, 3000)
    # TODO what is the borderline of fast+acc
    exp_f1(3e-3, 5000)
    exp_f1(2e-3, 5000)
    # this should be the most promising results for this exp setting
    exp_f1(1e-3, 5000)
    exp_f1(5e-4, 1000)
end

function exp()
    # itadv train with different learning rate
    exp_itadv()
    # TODO pretrain CNN with different learning rate
    exp_pretrain()
    # TODO nat+acc 1:1 with different learning rate
    exp_f1()
    # TODO mixing data with schedule
    # TODO dynamic attacking strength
    # FIXME I will need to monitor the acc of which attack?
end

function main()
    test()
    exp()
end
