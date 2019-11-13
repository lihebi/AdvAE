function train_MNIST_model(model, trainX, trainY, valX, valY)
    model(trainX[1]);

    loss(x, y) = my_xent(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

    evalcb = throttle(() -> @show(loss(valX[1], valY[1])) , 5);
    opt = ADAM(1e-3);
    @epochs 10 mytrain!(loss, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end

function advtrain!(model, attack_fn, loss, ps, data, opt; cb = () -> ())
    ps = Flux.Tracker.Params(ps)
    cb = runall(cb)
    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()
    step = 0
    @showprogress 0.1 "Training..." for d in data
        # FIXME can I use x,y in for variable?
        x, y = d
        x_adv = attack_fn(model, d...)

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
            # l = adv_loss + clean_loss
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

    # evalcb = throttle(cb_fn , 5);
    evalcb = () -> ()

    # train
    opt = ADAM(1e-4);
    @epochs 3 advtrain!(model, attack_fn, Flux.params(model), zip(trainX, trainY), opt, cb=evalcb)
end

function evaluate_attack(model, attack_fn, trainX, trainY, testX, testY)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    println("Clean accuracy:")
    # FIXME this may need to be evaluated on CPU
    @show accuracy(trainX[1], trainY[1])
    @show accuracy(testX[1], testY[1])

    sample_and_view(testX[1], testY[1])

    x = testX[1]
    y = testY[1]

    @info "performing attack .."
    x_adv = attack_fn(model, x, y)
    @info "attack done."

    @show accuracy(x_adv, y)

    sample_and_view(x_adv, model(x_adv))

    # all test data
    m_acc = MeanMetric()
    @showprogress 0.1 "testing all data .." for d in zip(testX, testY)
        x, y = d
        x_adv = attack_fn(model, x, y)
        acc = accuracy(x_adv, y)
        add!(m_acc, acc)
    end
    @show get(m_acc)
    nothing
end
