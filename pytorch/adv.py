# Not going to use this. I'm going to implement my own.
#
# from advertorch.attacks import LinfPGDAttack

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from tqdm import tqdm

from model import get_Madry_model, get_LeNet5, evaluate, train_MNIST_model
from data_utils import imrepl, get_mnist
from utils import clear_tqdm

def FGSM(model, loss_fn, x, y, ε = 0.1):
    adv = x.clone()
    adv.requires_grad = True
    # FIXME where to perform softmax
    loss = loss_fn(model(adv), y)
    # DEBUG does not seem to mess up with outer training loop
    model.zero_grad()
    loss.backward()
    adv.data += ε * adv.grad.data.sign()
    torch.clamp(adv, 0, 1)
    # FIXME do I need to zero out adv.grad?
    return adv.detach()

def PGD(model, loss_fn, x, y, ε = 0.3, step=0.01, iters=40):
    """FIXME a clean version so as to not mess up with the model.
    """
    x_adv = x.clone()
    # random start
    Δ = torch.zeros_like(x)
    Δ.data.uniform_(-1, 1)
    x_adv += Δ * step
    x_adv = torch.clamp(x_adv, 0, 1)

    for i in range(iters):
        x_adv = FGSM(model, loss_fn, x_adv, y, ε = step)
        eta = x_adv - x
        eta = torch.clamp(eta, -ε, ε)
        x_adv = x + eta
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def _accuracy(model, x, y):
    pred = torch.argmax(model(x), 1)
    total = y.size(0)
    correct = (pred == y).sum().item()
    return correct / total

def advtrain(model, opt, loss_fn, dl, cb=None, epoch=10):
    """
    cb(iter, loss)
    """
    for _ in range(epoch):
        running_loss = 0.0
        clear_tqdm()
        for i, data in enumerate(tqdm(dl), 0):
            inputs, labels = data
            target = labels

            # FIXME I should not have to do this
            model.train()

            xadv = PGD(model, loss_fn, inputs, labels)
            loss = loss_fn(model(xadv), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()

            running_loss += loss.item()
            if cb:
                # FIXME would this affect correctness?
                with torch.no_grad():
                    cb(i, running_loss)
            # FIXME move this to cb
            if i % 20 == 9:
                with torch.no_grad():
                    nat_acc = _accuracy(model, inputs, labels)
                    adv_acc = _accuracy(model, xadv, labels)
                    # adv_acc = 0
                    print('nat acc: {:.3f}, adv acc: {:.3f}'.format(nat_acc, adv_acc))

def do_advtrain(model, train_dl):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    advtrain(model, opt, loss_fn, train_dl, epoch=1)

def evaluate_attack(model, dl):
    advcorrect = 0
    total = 0
    # first show some examples

    # FIXME nll_loss?
    # loss = F.nll_loss(output, target)
    loss_fn = nn.CrossEntropyLoss()
    xs, ys = next(iter(train_dl))
    print('evaluting clean model ..')
    evaluate(model, [(xs, ys)])
    print('evaluating FGSM ..')
    x_adv = FGSM(model, loss_fn, xs, ys)
    # imrepl(make_grid(x_adv[:10], nrow=10))
    evaluate(model, [(x_adv, ys)])
    print('evaluating PGD ..')
    x_adv = PGD(model, loss_fn, xs, ys)
    evaluate(model, [(x_adv, ys)])

    print('evaluting all test data on PGD ..')
    clear_tqdm()
    for data in tqdm(dl):
        x, y = data
        adv = PGD(model, nn.CrossEntropyLoss(), x, y)
        with torch.no_grad():
            output = model(adv)
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(y.view_as(pred)).sum().item()
        total += len(x)
    advacc = advcorrect / total
    print('advacc: ', advacc)

def test():
    torch.manual_seed(0)

    model = get_Madry_model()
    # model = get_LeNet5()

    train_dl, valid_dl, test_dl = get_mnist(batch_size=50)

    train_MNIST_model(model, train_dl, valid_dl)

    do_advtrain(model, train_dl)

    evaluate(model, test_dl)
    evaluate_attack(model, test_dl)
