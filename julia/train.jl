# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
# using Adversarial: FGSM
# import Adversarial

using BSON: @save, @load

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

function attack_FGSM(model, x, y)
    x_adv = myFGSM(model, x, y; ϵ = 0.3)
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

function attack_CIFAR10_PGD_k(k)
    # FIXME the model contains BN layers, and is set to test mode during
    # testing. However, I still need to perform attack during testing, which
    # requires gradient. But testing mode model cannot take gradient?
    function attack(model, x, y)
        # cifar: 8.0/255, 7, 2./255
        myPGD(model, x, y;
              ϵ = 8/255,
              step_size = 2/255,
              iters = k)
    end
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
                train_steps=ds.nbatch,
                from_steps=1,
                print_steps=100,
                augment=(x)->x,
                logger=nothing,
                save_cb=(i)->nothing,
                test_cb=(i)->nothing)
    ps=Flux.params(model)

    loss_metric = MeanMetric()
    acc_metric = MeanMetric()


    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in from_steps:train_steps
        x, y = next_batch!(ds)
        # TODO add data augment here
        x = augment(x) |> gpu
        y = gpu(y)

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
            loss=get!(loss_metric)
            acc=get!(acc_metric)
            @info "data"  loss acc
            if typeof(logger) <: TBLogger
                log_value(logger, "loss", loss, step=step)
                log_value(logger, "acc", acc, step=step)
            end
        end
        test_cb(step)
        save_cb(step)
    end
end

function advtrain!(model, opt, attack_fn, ds;
                   train_steps=ds.nbatch,
                   from_steps=1,
                   print_steps=50,
                   logger=nothing,
                   λ = 0,
                   save_cb=(i)->nothing,
                   test_cb=(i)->nothing)
    ps=Flux.params(model)

    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in from_steps:train_steps
        x, y = next_batch!(ds) |> gpu
        # this computation won't affect model parameter gradients
        x_adv = attack_fn(model, x, y)

        gs = Flux.Tracker.gradient(ps) do
            # FIXME this will slow down the model twice
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            # TODO add l2 regularizers
            #
            # l1(x) = sum(abs.(x))
            # l2(x) = sum(x.^2)
            # l2_decay = sum(l2, weight_params(model))

            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)

            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))
            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))

            l = adv_loss + λ * clean_loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            natloss = get!(m_cleanloss)
            advloss = get!(m_advloss)
            natacc = get!(m_cleanacc)
            advacc = get!(m_advacc)
            @info "data" natloss advloss natacc advacc
            # TODO log training time
            if typeof(logger) <: TBLogger
                log_value(logger, "loss/nat_loss", natloss, step=step)
                log_value(logger, "loss/adv_loss", advloss, step=step)
                log_value(logger, "acc/nat_acc", natacc, step=step)
                log_value(logger, "acc/adv_acc", advacc, step=step)
            end
        end
        test_cb(step)
        save_cb(step)
    end
end

function free_train!(model, opt, ds, ε;
                     train_steps=ds.nbatch,
                     from_steps=1,
                     print_steps=50,
                     logger=nothing,
                     λ = 0,
                     replay_m=8,
                     save_cb=(i)->nothing,
                     test_cb=(i)->nothing)
    ps=Flux.params(model)

    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in from_steps:train_steps
        x, y = next_batch!(ds) |> gpu

        # comment out to use "warm up"
        δ = gpu(zeros(size(x)...))
        # FIXME which is one step?
        for i in 1:replay_m
            xx = x + δ
            px = Flux.param(xx)
            gs = Flux.Tracker.gradient(ps) do
                logits = model(px)
                loss = my_xent(logits, y)
                # record statistics
                add!(m_advloss, loss.data)
                add!(m_advacc, accuracy_with_logits(logits.data, y))
                loss
            end
            Flux.Tracker.update!(opt, ps, gs)
            δ += Float32(ε) * sign.(px.grad)
            δ = clamp.(δ, -Float32(ε), Float32(ε))
        end

        if step % print_steps == 0
            println()
            natloss = get!(m_cleanloss)
            advloss = get!(m_advloss)
            natacc = get!(m_cleanacc)
            advacc = get!(m_advacc)
            @info "data" natloss advloss natacc advacc
            # TODO log training time
            if typeof(logger) <: TBLogger
                log_value(logger, "loss/nat_loss", natloss, step=step)
                log_value(logger, "loss/adv_loss", advloss, step=step)
                log_value(logger, "acc/nat_acc", natacc, step=step)
                log_value(logger, "acc/adv_acc", advacc, step=step)
            end
        end
        test_cb(step)
        save_cb(step)
    end
end

function create_save_cb(model_file, model; save_steps)
    function save_cb(step)
        if step % save_steps == 0
            @info "saving .."
            # FIXME opt cannot be saved
            # FIXME logger cannot be saved
            #
            # FIXME sometimes user interrupt would crash the file, I'm saving it
            # as xxx-tmp, and perform move afterwards, hopefully fix this.
            @time @save model_file*".tmp" model=cpu(model) from_steps=step
            mv(model_file*".tmp", model_file, force=true)
        end
    end
end

function create_advae_test_cb(ae, cnn, test_ds; kwargs...)
    testmode_fn() = Flux.testmode!(ae)
    testmode_exit_fn() = Flux.testmode!(ae, false)
    create_adv_test_cb_impl(Chain(ae, cnn), test_ds,
                            testmode_fn, testmode_exit_fn;
                            kwargs...)
end

function create_adv_test_cb(model, test_ds; kwargs...)
    testmode_fn() = Flux.testmode!(model)
    testmode_exit_fn() = Flux.testmode!(model, false)
    create_adv_test_cb_impl(model, test_ds,
                            testmode_fn, testmode_exit_fn;
                            kwargs...)
end

function create_adv_test_cb_impl(model, test_ds, testmode_fn, testmode_exit_fn;
                                 test_per_steps, test_run_steps, attack_fn, logger=nothing)
    function test_cb(step)
        if step % test_per_steps == 0
            println()
            @info "testing for $test_run_steps steps .."
            m_cleanloss = MeanMetric()
            m_cleanacc = MeanMetric()
            m_advloss = MeanMetric()
            m_advacc = MeanMetric()

            testmode_fn()
            @showprogress 0.1 "Inner testing..." for i in 1:test_run_steps
                x, y = next_batch!(test_ds) |> gpu
                # this computation won't affect model parameter gradients
                x_adv = attack_fn(model, x, y)
                adv_logits = model(x_adv)
                adv_loss = my_xent(adv_logits, y)
                clean_logits = model(x)
                clean_loss = my_xent(clean_logits, y)
                add!(m_advloss, adv_loss.data)
                add!(m_advacc, accuracy_with_logits(adv_logits.data, y))
                add!(m_cleanloss, clean_loss.data)
                add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))
            end
            testmode_exit_fn()

            natloss = get!(m_cleanloss)
            advloss = get!(m_advloss)
            natacc = get!(m_cleanacc)
            advacc = get!(m_advacc)
            @info "test data" natloss advloss natacc advacc

            # use explicit API to avoid manipulating log_step_increment
            if typeof(logger) <: TBLogger
                @info "logging results .."
                log_value(logger, "loss/nat_loss", natloss, step=step)
                log_value(logger, "loss/adv_loss", advloss, step=step)
                log_value(logger, "acc/nat_acc", natacc, step=step)
                log_value(logger, "acc/adv_acc", advacc, step=step)
            end
        end
    end
end

function create_test_cb(model, test_ds; logger=nothing)
    function test_cb(step)
        test_per_steps = 100
        test_run_steps = 20

        if step % test_per_steps == 0
            println()
            @info "testing for $test_run_steps steps .."
            m_cleanloss = MeanMetric()
            m_cleanacc = MeanMetric()

            # into testing mode
            Flux.testmode!(model)

            @showprogress 0.1 "Inner testing..." for i in 1:test_run_steps
                x, y = next_batch!(test_ds) |> gpu
                # this computation won't affect model parameter gradients
                logits = model(x)
                loss = my_xent(logits, y)
                add!(m_cleanloss, loss.data)
                add!(m_cleanacc, accuracy_with_logits(logits.data, y))
            end

            # back into training node
            Flux.testmode!(model, false)

            loss = get!(m_cleanloss)
            acc = get!(m_cleanacc)
            @info "test data" loss acc

            # use explicit API to avoid manipulating log_step_increment
            if typeof(logger) <: TBLogger
                @info "logging results .."
                log_value(logger, "loss", loss, step=step)
                log_value(logger, "acc", acc, step=step)
            end
        end
    end
end

##############################
## AE train
##############################

my_mse(ŷ, y) = sum((ŷ .- y).*(ŷ .- y)) * 1 // length(y)

function aetrain!(model, opt, ds;
                  train_steps=ds.nbatch,
                  from_steps=1,
                  print_steps=50)
    ps=Flux.params(model)

    loss_metric = MeanMetric()

    @info "Training for $train_steps steps, printing even $print_steps steps .."
    @showprogress 0.1 "Training..." for step in from_steps:train_steps
        x, _ = next_batch!(ds) |> gpu
        gs = Flux.Tracker.gradient(ps) do
            logits = model(x)
            loss = my_mse(logits, x)
            add!(loss_metric, loss.data)
            loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            @info "data" get!(loss_metric)
        end
    end
end

function advae_train!(ae, cnn, opt, attack_fn, ds;
                      train_steps=ds.nbatch,
                      from_steps=1,
                      β=1,
                      λ=0,
                      γ=0,
                      print_steps=50,
                      logger=nothing,
                      save_cb=(i)->nothing,
                      test_cb=(i)->nothing)
    # update only ae parameter
    ps=Flux.params(ae)

    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    m_recloss = MeanMetric()

    full_model = Chain(ae, cnn)

    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in from_steps:train_steps
        x, y = next_batch!(ds) |> gpu
        # this computation won't affect model parameter gradients
        x_adv = attack_fn(full_model, x, y)

        gs = Flux.Tracker.gradient(ps) do
            adv_logits = full_model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            clean_logits = full_model(x)
            clean_loss = my_xent(clean_logits, y)

            rec_loss = my_mse(ae(x), x)

            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))
            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))
            add!(m_recloss, rec_loss.data)

            l = adv_loss + λ * clean_loss + γ * rec_loss
        end
        # FIXME clean up cnn parameter gradients?
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            natloss = get!(m_cleanloss)
            advloss = get!(m_advloss)
            recloss = get!(m_recloss)
            natacc = get!(m_cleanacc)
            advacc = get!(m_advacc)
            @info "data" natloss advloss natacc advacc recloss
            if typeof(logger) <: TBLogger
                log_value(logger, "loss/nat_loss", natloss, step=step)
                log_value(logger, "loss/adv_loss", advloss, step=step)
                log_value(logger, "loss/rec_loss", recloss, step=step)
                log_value(logger, "acc/nat_acc", natacc, step=step)
                log_value(logger, "acc/adv_acc", advacc, step=step)
            end
        end
        test_cb(step)
        save_cb(step)
    end
end
