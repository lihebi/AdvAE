# https://github.com/jaypmorgan/Adversarial.jl.git
#
# I'm specifically not using PGD, but myPGD instead.
using Adversarial: FGSM
using Images
using Plots

include("model.jl")

function train_MNIST_model(model, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = crossentropy(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(0.001);
    @epochs 10 mytrain!(loss, params(model), zip(trainX, trainY), opt, cb=evalcb)
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
    x_adv = clamp.(x + (gpu(randn(Float32, size(x)...))
                        * Float32(step_size)),
                   clamp_range...);
    iter = 1; while iter <= iters
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        eta = x_adv - x
        eta = clamp.(eta, -ϵ, ϵ)
        x_adv = x + Float32.(eta)
        x_adv = clamp.(x_adv, clamp_range...)
        iter += 1
    end
    return x_adv
end

function attack_PGD(model, loss, x, y)
    x_adv = myPGD(model, loss, x, y;
                ϵ = 0.3,
                step_size = 0.01,
                iters = 40);
    x_adv
end

function attack_FGSM(model, loss, x, y)
    x_adv = FGSM(model, loss, x, y; ϵ = 0.3)
    x_adv
end

"""Attack the model, get adversarial inputs, and evaluate accuracy

TODO different attack methods
"""
function evaluate_attack(model, attack_fn, trainX, trainY, testX, testY)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println("Clean accuracy:")
    # print out training details, e.g. accuracy
    @show accuracy(trainX[1], trainY[1])
    # Test set accuracy
    @show accuracy(testX[1], testY[1])

    sample_and_view(testX[1], testY[1])

    println("Doing adversarial attack ..")
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
    adversarial_pred = model(x_adv) |> Flux.onecold
    original_pred = model(x) |> Flux.onecold
    # @show adversarial_pred[1:10]
    # @show original_pred[1:10]
    @show sum(adversarial_pred .== original_pred)
    @show size(adversarial_pred)[1]
    adv_acc = sum(adversarial_pred .== original_pred) / size(adversarial_pred)[1]
    @show adv_acc

    sample_and_view(x_adv, model(x_adv))
end

function test_attack()
    (trainX, trainY), (valX, valY), (testX, testY) = load_MNIST();
    cnn = get_MNIST_CNN_model()
    train_MNIST_model(cnn, trainX, trainY, valX, valY)
    # 0.17
    evaluate_attack(cnn, attack_FGSM, trainX, trainY, testX, testY)
    # 0.03
    evaluate_attack(cnn, attack_PGD, trainX, trainY, testX, testY)
end
