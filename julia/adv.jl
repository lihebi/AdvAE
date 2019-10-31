# https://github.com/jaypmorgan/Adversarial.jl.git
using Adversarial
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

function attack_PGD(model, loss, x, y)
    # FIXME PGD performance problem
    # acc: 0.03
    x_adv = PGD(model, loss, x, y;
                ϵ = 0.3,
                step_size = 0.01,
                iters = 40);
    x_adv
end

function attack_FGSM(model, loss, x, y)
    # FIXME acc: 0.97
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
    x = testX[1][:,:,:,1:10]
    y = testY[1][:,1:10]
    # use entire batch
    # x = testX[1]
    # y = testY[1]

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
    evaluate_attack(cnn, attack_FGSM, trainX, trainY, testX, testY)
    evaluate_attack(cnn, attack_PGD, trainX, trainY, testX, testY)
end
