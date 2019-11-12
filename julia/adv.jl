# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
using Adversarial: FGSM
using Images

include("model.jl")

function my_xent(logits, y)
    # maybe sum?
    Flux.logitcrossentropy(logits, y)
end

function train_MNIST_model(model, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = my_xent(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(1e-3);
    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end

"""Not using, the same as Adversarial.jl's FGSM.
"""
function myFGSM(model, loss, x, y; ϵ = 0.1, clamp_range = (0, 1))
    px, θ = Flux.param(x), Flux.params(model)
    Flux.Tracker.gradient(() -> loss(px, y), θ)
    x_adv = clamp.(x + (Float32(ϵ) * sign.(px.grad)), clamp_range...)
end


"""
My modifications:

- remove (δ < ϵ) condition. This seems to fix PGD performance problem. This
  condition also has potential problem when attacking in batch: you should
  really not stop when one image in the batch reached the epsilon-ball

- clipped into valid range of epsilon, i.e. clip_eta in cleverhans
"""
function myPGD(model, loss, x, y;
               ϵ = 10, step_size = 0.001,
               iters = 100, clamp_range = (0, 1))
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
        x_adv = FGSM(model, loss, x_adv, y;
                     ϵ = step_size, clamp_range = clamp_range)
        eta = x_adv - x
        eta = clamp.(eta, -ϵ, ϵ)
        x_adv = x + Float32.(eta)
        x_adv = clamp.(x_adv, clamp_range...)
    end
    return x_adv
end

function attack_PGD(model, loss, x, y)
    x_adv = myPGD(model, loss, x, y;
                  ϵ = 0.3,
                  step_size = 0.01,
                  iters = 40)
end

function attack_PGD_k(k)
    (model, loss, x, y) -> begin
        x_adv = myPGD(model, loss, x, y;
                      ϵ = 0.3,
                      step_size = 0.01,
                      iters = k)
    end
end


function attack_FGSM(model, loss, x, y)
    x_adv = FGSM(model, loss, x, y; ϵ = 0.3)
end

"""Attack the model, get adversarial inputs, and evaluate accuracy

TODO different attack methods
FIXME use all test data to test
"""
function evaluate_attack(model, attack_fn, trainX, trainY, testX, testY)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println("Clean accuracy:")
    # FIXME this may need to be evaluated on CPU
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(testX[1], testY[1])

    sample_and_view(testX[1], testY[1])

    # using only one
    # x = testX[1][:,:,:,1:1]
    # y = testY[1][:,1]
    # use 10
    # x = testX[1][:,:,:,1:10]
    # y = testY[1][:,1:10]
    # use entire batch
    x = testX[1]
    y = testY[1]

    loss(x, y) = my_xent(model(x), y)
    @info "performing attack .."
    x_adv = attack_fn(model, loss, x, y)
    @info "attack done."

    # we can see that the predicted labels are different
    # adversarial_pred = model(x_adv) |> Flux.onecold
    # original_pred = model(x) |> Flux.onecold

    @show accuracy(x_adv, y)

    sample_and_view(x_adv, model(x_adv))

    # all test data
    m_acc = MeanMetric()
    @showprogress 0.1 "testing all data .." for d in zip(testX, testY)
        x, y = d
        loss(x, y) = my_xent(model(x), y)
        x_adv = attack_fn(model, loss, x, y)
        acc = accuracy(x_adv, y)
        add!(m_acc, acc)
    end
    @show get(m_acc)
    nothing
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

function test()
    m = MeanMetric()
    get(m)
    add!(m, 1)
    add!(m, 2)
    get(m)
    reset!(m)

    m.sum
    m.n
end

# FIXME this is scalar operation, but anyway
# FIXME do I need .data for y?
# FIXME logits.data?
accuracy_with_logits(logits, y) = mean(onecold(logits) .== onecold(y))

"""TODO use model and ps in attack? This follows the flux train tradition, but
is this better? I think using model and loss is better, where loss should accept
model, not x and y.

TODO full adv evaluation at the end of each epoch

"""
function advtrain!(model, attack_fn, loss, ps, data, opt; cb = () -> ())
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    # FIXME how to efficiently track metrics?
    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    step = 0
    @showprogress 0.1 "Training..." for d in data
        # FIXME can I use x,y in for variable?
        x, y = d
        x_adv = attack_fn(model, loss, d...)

        # train xent loss using adv data
        gs = Flux.Tracker.gradient(ps) do
            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))
            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))

            l = adv_loss
            # DEBUG add clean loss
            # l = adv_loss + clean_loss

            # NOTE: the last expression is returned, but I just want to
            # explicitly return l to have the correct loss calculation.
            # return l
            #
            # FIXME return keyword seem to break showprogress. Thus I need to
            # make sure this is the last expression.
            l
        end
        Flux.Tracker.update!(opt, ps, gs)

        step += 1

        if step % 40 == 0
            println()
            @show get!(m_cleanloss)
            @show get!(m_cleanacc)
            @show get!(m_advloss)
            @show get!(m_advacc)
        end
        # cb(step, total_loss)
        cb()
    end
end

function advtrain(model, attack_fn, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss_fn(x, y) = my_xent(model(x), y)
    # evalcb = throttle(cb_fn , 5);
    evalcb = () -> ()

    # train
    # FIXME would this be 0.001 * 0.001?
    # FIXME decay on pleau
    # FIXME print out information when decayed
    # opt = Flux.Optimiser(Flux.ExpDecay(0.001, 0.5, 1000, 1e-4), ADAM(0.001))
    opt = ADAM(1e-4);
    # TODO use steps instead of epoch
    # TODO shuffle dataset, using data loader instead of moving all data to GPU at once
    # TODO make it faster
    @epochs 3 advtrain!(model, attack_fn, loss_fn, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end


function test_attack()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST(batch_size=32);

    model = get_Madry_model()[1:end-1]
    # FIXME LeNet5 is working, Madry is not working
    # cnn = get_LeNet5()

    train_MNIST_model(model, trainX, trainY, valX, valY)

    # FIXME it does not seem to have the same level of security
    advtrain(model, attack_PGD_k(7), trainX, trainY, valX, valY)
    advtrain(model, attack_PGD_k(20), trainX, trainY, valX, valY)
    advtrain(model, attack_PGD_k(40), trainX, trainY, valX, valY)

    evaluate_attack(model, attack_FGSM, trainX, trainY, testX, testY)
    evaluate_attack(model, attack_PGD, trainX, trainY, testX, testY)
    evaluate_attack(model, attack_PGD_k(7), trainX, trainY, testX, testY)
    evaluate_attack(model, attack_PGD_k(20), trainX, trainY, testX, testY)
    evaluate_attack(model, attack_PGD_k(40), trainX, trainY, testX, testY)
end

function custom_train!(model, opt, ds)
    x, y = next_batch!(ds) |> gpu
    model(x);

    ps = Flux.params(model)
    loss_metric = MeanMetric()
    acc_metric = MeanMetric()
    step = 0
    @showprogress 0.1 "Training..." for step in 1:1000
        x, y = next_batch!(ds) |> gpu
        gs = Flux.Tracker.gradient(ps) do
            # FIXME this will slow down the model twice
            logits = model(x)
            loss = my_xent(logits, y)
            add!(loss_metric, loss.data)
            add!(acc_metric, accuracy_with_logits(logits.data, y))
            loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % 40 == 0
            println()
            @show get!(loss_metric)
            @show get!(acc_metric)
        end
    end
end

function custom_evaluate(model, ds; attack_fn=nothing)
    # first, sample clean image
    xx,yy = next_batch!(ds) |> gpu
    sample_and_view(xx, model(xx))

    # then, run clean image accuracy
    acc_metric = MeanMetric()
    @showprogress 0.1 "Testing..." for step in 1:10
        x,y = next_batch!(ds) |> gpu
        acc = accuracy_with_logits(model(x), y)
        add!(acc_metric, acc)
    end
    @show get!(acc_metric)

    # then, run attack
    if attack_fn != nothing
        loss_fn(x, y) = my_xent(model(x), y)

        adv = attack_fn(model, loss_fn, xx, yy)
        # FIXME showing prediction, should show both label/pred
        sample_and_view(adv, model(adv))

        advacc_metric = MeanMetric()
        @showprogress 0.1 "Testing adv..." for step in 1:10
            x,y = next_batch!(ds) |> gpu
            adv = attack_fn(model, loss_fn, x, y)
            acc = accuracy_with_logits(model(adv), y)
            add!(advacc_metric, acc)
        end
        @show get!(advacc_metric)
    end
    nothing
end

function custom_advtrain!(model, opt, attack_fn, ds)
    loss_fn(x, y) = my_xent(model(x), y)
    ps = Flux.params(model)

    # FIXME how to efficiently track metrics?
    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    @showprogress 0.1 "Training..." for step in 1:1000
        x, y = next_batch!(ds) |> gpu
        x_adv = attack_fn(model, loss_fn, x,y)
        gs = Flux.Tracker.gradient(ps) do
            # FIXME this will slow down the model twice
            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            # I should be able to compute any value, or even get data, and save
            # to metrics
            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))
            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))

            l = adv_loss
            # DEBUG add clean loss
            # l = adv_loss + clean_loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % 40 == 0
            println()
            @show get!(m_cleanloss)
            @show get!(m_cleanacc)
            @show get!(m_advloss)
            @show get!(m_advacc)
        end
    end
end


function test()
    train_ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(train_ds) |> gpu;

    # FIXME NOW why Madry model does not work, while LeNet5 works (on
    # adv+clean, but still not on adv alone)?
    model = get_Madry_model()[1:end-1]
    model = get_LeNet5()[1:end-1]

    model(x)

    opt = ADAM(1e-4);

    custom_train!(model, opt, train_ds)
    custom_advtrain!(model, opt, attack_PGD_k(40), train_ds)

    # FIXME arbitrary model has 1.0 adv accuracy?
    custom_evaluate(model, test_ds, attack_fn=attack_PGD_k(40))
    accuracy_with_logits(model(x), y)
end
