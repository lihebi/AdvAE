# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
using Adversarial: FGSM
using Images

include("model.jl")

function train_MNIST_model(model, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = crossentropy(model(x), y)
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
    r = clamp.(gpu(randn(Float32, size(x)...)), clamp_range...)
    # DEBUG uniform sample
    # r = gpu(rand(Float32, size(x)...)) .- 0.5 .* 2
    x_adv = clamp.(x + (r * Float32(step_size)), clamp_range...)
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

    loss(x, y) = crossentropy(model(x), y)
    @info "performing attack .."
    x_adv = attack_fn(model, loss, x, y)
    @info "attack done."

    # we can see that the predicted labels are different
    # adversarial_pred = model(x_adv) |> Flux.onecold
    # original_pred = model(x) |> Flux.onecold

    @show accuracy(x_adv, y)

    sample_and_view(x_adv, model(x_adv))

    # all test data
    accs = []
    @showprogress 0.1 "testing all data .." for d in zip(testX, testY)
        x, y = d
        loss(x, y) = crossentropy(model(x), y)
        x_adv = attack_fn(model, loss, x, y)
        acc = accuracy(x_adv, y)
        push!(accs, acc)
    end
    @show mean(accs)
    nothing
end


"""TODO use model and ps in attack? This follows the flux train tradition, but
is this better? I think using model and loss is better, where loss should accept
model, not x and y.

TODO full adv evaluation at the end of each epoch

"""
function advtrain!(model, attack_fn, loss, ps, data, opt; cb = () -> ())
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    @showprogress 0.1 "Training..." for d in data
        x_adv = attack_fn(model, loss, d...)

        # train xent loss using adv data
        gs = Flux.Tracker.gradient(ps) do
            # addition works
            # loss(x_adv, d[2]) + loss(d...)
            loss(x_adv, d[2])

            # FIXME concatenation works
            # nx = cat(d[1], x_adv, dims=4)
            # ny = gpu(cat(d[2], d[2], dims=2))
            # loss(nx, ny)

            # FIXME cannot run two loss sequentially. The last one seems to be in effect
            # loss(d...)
            # loss(x_adv, d[2])
        end
        Flux.Tracker.update!(opt, ps, gs)

        cb()
    end
end

function advtrain(model, attack_fn, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = crossentropy(model(x), y)
    # evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    #
    # I should probably monitor the adv accuracy
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    adv_accuracy(x, y) = begin
        x_adv = attack_fn(model, loss, x, y)
        accuracy(x_adv, y)
    end
    adv_loss(x, y) = begin
        x_adv = attack_fn(model, loss, x, y)
        loss(x_adv, y)
    end
    cb_fn() = begin
        # add a new line so that it plays nicely with progress bar
        @time begin
            println("")
            @show loss(valX[1], valY[1])
            @show adv_loss(valX[1], valY[1])
            @show accuracy(valX[1], valY[1])
            @show adv_accuracy(valX[1], valY[1])
            @show accuracy(trainX[1], trainY[1])
            @show adv_accuracy(trainX[1], trainY[1])
        end
    end
    evalcb = throttle(cb_fn , 10);

    # train
    # FIXME would this be 0.001 * 0.001?
    # FIXME decay on pleau
    # FIXME print out information when decayed
    # opt = Flux.Optimiser(Flux.ExpDecay(0.001, 0.5, 1000, 1e-4), ADAM(0.001))
    opt = ADAM(1e-4);
    @epochs 3 advtrain!(model, attack_fn, loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end


function test_attack()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST(batch_size=50);

    cnn = get_Madry_model()
    # FIXME LeNet5 is working, Madry is not working
    # cnn = get_LeNet5()

    train_MNIST_model(cnn, trainX, trainY, valX, valY)

    # FIXME it does not seem to have the same level of security
    advtrain(cnn, attack_PGD_k(7), trainX, trainY, valX, valY)
    advtrain(cnn, attack_PGD_k(20), trainX, trainY, valX, valY)
    advtrain(cnn, attack_PGD_k(40), trainX, trainY, valX, valY)

    evaluate_attack(cnn, attack_FGSM, trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD, trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD_k(7), trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD_k(20), trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD_k(40), trainX, trainY, testX, testY)
end
