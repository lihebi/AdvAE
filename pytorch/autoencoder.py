import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from tqdm import tqdm

from model import get_Madry_model, get_LeNet5, evaluate, Lambda
from model import train_MNIST_model
from data_utils import imrepl, get_mnist
from utils import clear_tqdm
from adv import advtrain, evaluate_attack, PGD, do_evaluate_attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dense_AE():
    encoder = nn.Sequential(
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(28*28, 32),
        nn.ReLU())
    decoder = nn.Sequential(
        nn.Linear(32, 28*28),
        Lambda(lambda x: x.view(x.size(0), 1, 28, 28)),
        nn.Sigmoid())
    return nn.Sequential(encoder, decoder).to(device)

def CNN_AE():
    encoder = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)))
    decoder = nn.Sequential(
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(16, 1, kernel_size=3, padding=1),
        nn.Sigmoid())
    return nn.Sequential(encoder, decoder).to(device)

def _train_AE(model, opt, loss_fn, dl, cb=None, epoch=10):
    for _ in range(epoch):
        running_loss = 0.0
        clear_tqdm()
        for i, data in enumerate(tqdm(dl), 0):
            x, _ = data

            outputs = model(x)
            # not using y
            loss = loss_fn(outputs, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if cb:
                with torch.no_grad():
                    cb(i, running_loss)

def train_AE(model, train_dl, valid_dl):
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # this (and NLLLoss in general) expects label as target, thus no need to
    # one-hot encoding
    loss_fn = nn.MSELoss()

    # TODO throttle
    def cb(i, loss):
        # print statistics
        if i % 100 == 99:
            print('\n[step {:5d}] loss: {:.5f}'.format(i + 1, loss / i))

    _train_AE(model, opt, loss_fn, train_dl, cb=cb, epoch=5)

def evaluate_AE(model, dl):
    xs, _ = next(iter(dl))
    imrepl(make_grid(xs[:10], nrow=10))
    decoded = model(xs).detach()
    imrepl(make_grid(decoded[:10], nrow=10))

def test_AE():
    cnn = get_Madry_model()
    # cnn = get_LeNet5()
    ae = dense_AE()
    ae = CNN_AE()

    train_dl, valid_dl, test_dl = get_mnist(batch_size=50)

    train_MNIST_model(cnn, train_dl, valid_dl)
    evaluate(cnn, test_dl)

    train_AE(ae, train_dl, valid_dl)
    evaluate_AE(ae, test_dl)

    # FIXME not converging, similar as julia code
    do_advae_train(ae, cnn, train_dl)

    model = nn.Sequential(ae, cnn)
    # FIXME direct AE without training is increasing adv accuracy, 26%/22% vs 22%/12%
    do_evaluate_attack(model, test_dl)
    # FIXME PGD and LinfPGD difference: 22% vs 12%
    # FIXME clean CNN should have 0% PGD accuracy. Maybe because of LeNet?
    do_evaluate_attack(cnn, test_dl)


def do_advae_train(ae, cnn, dl):
    # opt = optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model = nn.Sequential(ae, cnn)
    # FIXME should I reset cnn grads inside the training loop?
    advae_train(model, opt, loss_fn, train_dl, epoch=1)

def _accuracy(model, x, y):
    pred = torch.argmax(model(x), 1)
    total = y.size(0)
    correct = (pred == y).sum().item()
    return correct / total
def advae_train(model, opt, loss_fn, dl, cb=None, epoch=10):
    for _ in range(epoch):
        running_loss = 0.0
        clear_tqdm()
        for i, data in enumerate(tqdm(dl), 0):
            inputs, labels = data
            target = labels

            model.train()

            xadv = PGD(model, loss_fn, inputs, labels)
            loss = loss_fn(model(xadv), labels)

            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            model.eval()

            if i % 20 == 9:
                with torch.no_grad():
                    adv_acc = _accuracy(model, xadv, labels)
                    adv_loss = running_loss / i
                    # adv_acc = 0
                    nat_loss = loss_fn(model(inputs), labels)
                    nat_acc = _accuracy(model, inputs, labels)
                    print('nat acc: {:.3f}, adv acc: {:.3f}'.format(nat_acc, adv_acc))
                    print('nat loss: {:.3f}, adv loss: {:.3f}'.format(nat_loss, adv_loss))
