include("data.jl")
include("model.jl")
include("train.jl")
include("exp.jl")

# CAUTION the function names are the same for CIFAR

function exp_itadv(lr, total_steps)
    expID = "f0-$lr"
    MNIST_exp_helper(expID, lr, total_steps, 0)
end

function exp_pretrain(lr, total_steps)
    expID = "pretrain-$lr"
    MNIST_exp_helper(expID, lr, total_steps, 0, pretrain=true)
end

function exp_f1(lr, total_steps)
    expID = "f1-$lr"
    MNIST_exp_helper(expID, lr, total_steps, 1)
end

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

##############################
## AdvAE experiments
##############################

function exp_advae_f0(lr, total_steps)
    expID = "f0-$lr"
    MNIST_advae_exp_helper(expID, lr, total_steps, λ=0)
end

function exp_advae_pretrain(lr, total_steps)
    expID = "pretrain-$lr"
    MNIST_advae_exp_helper(expID, lr, total_steps, λ=0, pretrain=true)
end

function exp_advae_f1(lr, total_steps)
    expID = "f1-$lr"
    MNIST_advae_exp_helper(expID, lr, total_steps, λ=1)
end

function exp_advae_f01(lr, total_steps)
    expID = "f01-$lr"
    MNIST_advae_exp_helper(expID, lr, total_steps, λ=0, γ=1)
end

function exp_advae_test(lr, total_steps; expID="test-$(now())")
    @show expID
    MNIST_advae_exp_helper(expID, lr, total_steps, λ=1, γ=10, β=1)
end

function test()
    exp_advae_test(2e-3, 2000, expID="test-2019-11-17T22:46:27.503")
    exp_advae_test(2e-3, 1000)

    @load "trained/pretrain-MNIST.bson" model
    # gpu(model)(x)
    exp_advae_f0(1e-3, 2000)
    exp_advae_f0(1e-4, 2000)
    exp_advae_pretrain(1e-4, 2000)
    # this works pretty good. Two things:
    #
    # 1. pretraining seems to be important here. I probably want to enable
    # pretrain by default, that does not hurt.
    #
    # 2. TODO IMPORTANT lr decay might help
    # 3. what is the performance gap (both acc and time) with itadv?
    exp_advae_pretrain(1e-3, 3000)
    # surprisingly this does not work smoothly, it struggled until a point clean
    # acc jumps
    exp_advae_f1(1e-3, 2000)
    exp_advae_f1(2e-3, 2000)

    # rec loss
    exp_advae_f01(1e-3, 2000)
end
