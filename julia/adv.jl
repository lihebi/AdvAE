# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
# using Adversarial: FGSM
import Adversarial
using Images

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
                loss_fn=my_xent, ps=Flux.params(model),
                acc_fn=accuracy_with_logits,
                train_steps=ds.nbatch, print_steps=50)
    loss_metric = MeanMetric()
    acc_metric = MeanMetric()
    step = 0

    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds) |> gpu
        gs = Flux.Tracker.gradient(ps) do
            logits = model(x)
            loss = loss_fn(logits, y)
            add!(loss_metric, loss.data)
            add!(acc_metric, acc_fn(logits.data, y))
            loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            @show get!(loss_metric)
            @show get!(acc_metric)
        end
    end
end

function advtrain!(model, opt, attack_fn, ds;
                   loss_fn=my_xent, ps=Flux.params(model),
                   train_steps=ds.nbatch, print_steps=50)
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

        if step % print_steps == 0
            println()
            @show get!(m_cleanloss)
            @show get!(m_cleanacc)
            @show get!(m_advloss)
            @show get!(m_advacc)
        end
    end
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

    @info "Adv training .."
    # custom_train!(model, opt, train_ds)
    @epochs 2 advtrain!(model, opt, attack_PGD_k(40), ds,
                        print_steps=20)

    @info "Evaluating clean accuracy .."
    evaluate(model, test_ds)
    evaluate(model, test_ds, attack_fn=attack_FGSM)
    evaluate(model, test_ds, attack_fn=attack_PGD_k(40))
end

function main()
    test()
end
