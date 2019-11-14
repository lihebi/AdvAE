# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
# using Adversarial: FGSM
import Adversarial
using Images

using Logging
using LoggingExtras: TeeLogger
using TensorBoardLogger
using Dates

using BSON: @save, @load

include("model.jl")

# CAUTION: A bug in CuArrays, https://github.com/FluxML/Flux.jl/issues/839
#
# onecold(cpu([1,-2])) # => 1
# onecold(gpu([1,-2])) # => 2
#
# The only scalar operation is the "mapreduce" in onecold. It also has bug that
# causes argmax to give incorrect results, thus, I'm disablng scalar operations,
# and force using CPU to compute onecold. CAUTION I should also check whether
# argmax bug affects my code elsewhere.
CuArrays.allowscalar(false)
# FIXME do I need .data for y?
# FIXME logits.data?
accuracy_with_logits(logits, y) = mean(onecold(cpu(logits)) .== onecold(cpu(y)))
# CuArrays.allowscalar(true)
# accuracy_with_logits(logits, y) = mean(onecold(logits) .== onecold(y))

my_xent(logits, y) = Flux.logitcrossentropy(logits, y)

"""Not using, the same as Adversarial.jl's FGSM.
"""
function myFGSM(model, x, y; ϵ = 0.3, clamp_range = (0, 1))
    # FIXME what if I just don't create Flux.params(model)?
    px, θ = Flux.param(x), Flux.params(model)
    Flux.Tracker.gradient(() -> my_xent(model(px), y), θ)
    x_adv = clamp.(x + (Float32(ϵ) * sign.(px.grad)), clamp_range...)
end


"""
My modifications:

- remove (δ < ϵ) condition. This seems to fix PGD performance problem. This
  condition also has potential problem when attacking in batch: you should
  really not stop when one image in the batch reached the epsilon-ball

- clipped into valid range of epsilon, i.e. clip_eta in cleverhans
"""
function myPGD(model, x, y;
               ϵ = 0.3, step_size = 0.01,
               iters = 40, clamp_range = (0, 1))
    # start from the random point
    # eta = gpu(randn(Float32, size(x)...))
    # eta = clamp.(eta, -ϵ, ϵ)
    # DEBUG uniform sample
    eta = (gpu(rand(Float32, size(x)...)) .- 0.5) * 2 * ϵ
    # FIXME this clamp should not be necessary
    eta = clamp.(eta, -ϵ, ϵ)
    x_adv = x + Float32.(eta)
    x_adv = clamp.(x_adv, clamp_range...)
    # x_adv = clamp.(x + (r * Float32(step_size)), clamp_range...)
    for iter = 1:iters
        x_adv = myFGSM(model, x_adv, y;
                       ϵ = step_size, clamp_range = clamp_range)
        eta = x_adv - x
        eta = clamp.(eta, -ϵ, ϵ)
        x_adv = x + Float32.(eta)
        x_adv = clamp.(x_adv, clamp_range...)
    end
    return x_adv
end

function attack_PGD(model, x, y)
    x_adv = myPGD(model, x, y;
                  ϵ = 0.3,
                  step_size = 0.01,
                  iters = 40)
end

function attack_PGD_k(k)
    (model, x, y) -> begin
        x_adv = myPGD(model, x, y;
                      ϵ = 0.3,
                      step_size = 0.01,
                      iters = k)
    end
end


function attack_FGSM(model, x, y)
    x_adv = myFGSM(model, x, y; ϵ = 0.3)
end

# something like tf.keras.metrics.Mean
# a good reference https://github.com/apache/incubator-mxnet/blob/master/julia/src/metric.jl
mutable struct MeanMetric
    sum::Float64
    n::Int
    MeanMetric() = new(0.0, 0)
end
function add!(m::MeanMetric, v)
    m.sum += v
    m.n += 1
end
get(m::MeanMetric) = m.sum / m.n
function get!(m::MeanMetric)
    res = m.sum / m.n
    reset!(m)
    res
end
function reset!(m::MeanMetric)
    m.sum = 0.0
    m.n = 0
end

function train!(model, opt, ds;
                train_steps=ds.nbatch, print_steps=50)
    ps=Flux.params(model)

    loss_metric = MeanMetric()
    acc_metric = MeanMetric()
    step = 0


    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds) |> gpu
        gs = Flux.Tracker.gradient(ps) do
            logits = model(x)
            loss = my_xent(logits, y)
            add!(loss_metric, loss.data)
            add!(acc_metric, accuracy_with_logits(logits.data, y))
            loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            @info "data" loss=get!(loss_metric) acc=get!(acc_metric) log_step_increment=print_steps
        end
    end
end

function advtrain!(model, opt, attack_fn, ds;
                   train_steps=ds.nbatch, print_steps=50,
                   logger=global_logger(),
                   save_cb=(i)->nothing)
    ps=Flux.params(model)

    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds) |> gpu
        # this computation won't affect model parameter gradients
        x_adv = attack_fn(model, x, y)

        gs = Flux.Tracker.gradient(ps) do
            # FIXME this will slow down the model twice
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)

            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))
            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))

            l = adv_loss
            # l = clean_loss
            # l = adv_loss + clean_loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        save_cb(step)

        if step % print_steps == 0
            println()
            @info "data" get(m_cleanloss) get(m_advloss) get(m_cleanacc) get(m_advacc)
            # TODO log training time
            @show typeof(logger)
            with_logger(logger) do
                @info "loss" nat_loss=get!(m_cleanloss) adv_loss=get!(m_advloss) log_step_increment=0
                @info "acc" nat_acc=get!(m_cleanacc) adv_acc=get!(m_advacc) log_step_increment=print_steps
            end
        end
    end
end

"""
TODO test data and results
"""
function exp_itadv(lr, total_steps)
    model_file = "trained/itadv-$lr.bson"
    mkpath(dirname(model_file))

    # FIXME should I record @time?
    # FIXME what should be my batch_size?
    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    if isfile(model_file)
        @info "Loading from $model_file .."
        # During load, there is no key=value, but the variable name shall match
        # the key when saving, thus the order is not important. During saving,
        # it can be key=value pair, or just the variable, but cannot be x.y
        @load model_file weights from_steps
        @info "loading weights"
        model = get_Madry_model()[1:end-1]
        Flux.loadparams!(model, weights)
    else
        @info "Starting from scratch .."
        model = get_Madry_model()[1:end-1]
        from_steps = 0
    end

    # advancing logger
    logger = TBLogger("tensorboard_logs/exp-itadv-$lr", tb_append, min_level=Logging.Info)

    to_steps = total_steps - from_steps
    @info "Progress:" total_steps from_steps to_steps

    @info "Advancing log step .." log_step_increment=from_steps
    with_logger(logger) do
        @info "Advancing log step .." log_step_increment=from_steps
    end

    # warm up the model
    model(x)

    # FIXME opt states
    opt = ADAM(lr);

    function save_cb(step)
        # FIXME as config, should be Integer multiple of print_steps
        save_steps = 20
        if step % save_steps == 0
            println()
            @info "saving .."
            # FIXME opt cannot be saved
            # FIXME logger cannot be saved
            #
            # FIXME model cannot be easily saved and loaded, weights can, but
            # needs to get rid of CuArrays and TrackedArrays
            @save model_file weights=Tracker.data.(Flux.params(cpu(model))) from_steps=step
        end
    end

    advtrain!(model, opt, attack_PGD_k(40), ds,
              train_steps=to_steps,
              print_steps=20,
              logger=logger,
              save_cb=save_cb)

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

function exp()
    # itadv train with different learning rate
    # FIXME should I use learning rate decay at the same time?
    exp_itadv(1e-1, 500)
    exp_itadv(5e-2, 500)
    exp_itadv(1e-2, 500)
    exp_itadv(5e-3, 500)
    exp_itadv(1e-3, 500)
    exp_itadv(5e-4, 1000)


    exp_itadv(1e-4, 5000)
    exp_itadv(5e-5, 5000)
    exp_itadv(1e-5, 5000)

    # TODO pretrain CNN with different learning rate
    # TODO nat+acc 1:1 with different learning rate
    # TODO mixing data with schedule
    # TODO dynamic attacking strength
end

function main()
    test()
    exp_itadv(1e-4, 500)
end
