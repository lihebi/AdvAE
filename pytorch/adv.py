# Not going to use this. I'm going to implement my own.
#
# from advertorch.attacks import LinfPGDAttack

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from model import get_MNIST_FC_model, get_MNIST_CNN_model, train_MNIST_model, evaluate
from data_utils import imrepl, get_mnist

# epsilons = [0, .05, .1, .15, .2, .25, .3]
# pretrained_model = "data/lenet_mnist_model.pth"
# use_cuda=True

def FGSM(model, loss_fn, x, y, ε = 0.1):
    adv = x.clone()
    adv.requires_grad = True
    # FIXME where to perform softmax
    loss = loss_fn(model(adv), y)
    model.zero_grad()
    loss.backward()
    adv.data += ε * adv.grad.data.sign()
    torch.clamp(adv, 0, 1)
    return adv.detach()

def PGD(model, loss_fn, x, y, ε = 0.3, step=0.01, iters=40):
    x_adv = x.clone()
    # random start
    Δ = torch.zeros_like(xs)
    Δ.data.uniform_(-1, 1)
    x_adv += Δ * step

    for i in range(iters):
        x_adv = FGSM(model, loss_fn, x_adv, y, ε = step)
        eta = x_adv - x
        eta = torch.clamp(eta, -ε, ε)
        x_adv = x + eta
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def test():
    model = get_MNIST_CNN_model()
    
    train_dl, valid_dl, test_dl = get_mnist()

    train_MNIST_model(model, train_dl, valid_dl)
    evaluate(model, test_dl)

    # FIXME nll_loss?
    # loss = F.nll_loss(output, target)
    loss_fn = nn.CrossEntropyLoss()
    xs, ys = next(iter(train_dl))
    x_adv = FGSM(model, loss_fn, xs, ys)
    x_adv = PGD(model, loss_fn, xs, ys)
    imrepl(make_grid(x_adv[:10], nrow=10))

    evaluate(model, [(x_adv, ys)])
    evaluate(model, [(xs, ys)])
