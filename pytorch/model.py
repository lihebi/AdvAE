import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.utils import make_grid

from tqdm import tqdm
import time

from utils import clear_tqdm
from data_utils import imrepl, get_mnist

__all__ = ['get_Madry_model', 'get_LeNet5', 'train_MNIST_model', 'evaluate', 'Lambda']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def get_Madry_model():
    """The MNIST CNN used in my tf code. This is also the Madry model used in MNIST
challenge."""
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(7*7*64, 1024),
        nn.Linear(1024, 10))
    return model.to(device)


def get_LeNet5():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(7*7*64, 200),
        nn.ReLU(),
        nn.Linear(200, 10))
    return model.to(device)


def train(model, opt, loss_fn, dl, cb=None, epoch=10):
    """
    cb(iter, loss)
    """
    for _ in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        clear_tqdm()
        for i, data in enumerate(tqdm(dl), 0):
            inputs, labels = data

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # FIXME should I zero grad before or after? Or both for safety?
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if cb:
                # FIXME would this affect correctness?
                with torch.no_grad():
                    cb(i, running_loss)

def accuracy(model, dl):
    correct = 0
    total = 0
    for data in dl:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

def train_MNIST_model(model, train_dl, valid_dl):
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # this (and NLLLoss in general) expects label as target, thus no need to
    # one-hot encoding
    loss_fn = nn.CrossEntropyLoss()

    # TODO throttle
    def cb(i, loss):
        # print statistics
        if i % 100 == 99:
            acc = accuracy(model, valid_dl)
            print('\n[step {:5d}] loss: {:.3f}, valid_acc: {:.3f}'
                  .format(i + 1, loss / i, acc))

    # FIXME tqdm show bar
    train(model, opt, loss_fn, train_dl,
          cb=cb, epoch=5)
    # train(model, opt, loss_fn, train_dl)
    return None

def evaluate(model, dl):
    """Run model and compute accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        clear_tqdm()
        for data in tqdm(dl):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # TODO visualize
    dataiter = iter(dl)
    images, labels = next(dataiter)
    # print images
    imrepl(torchvision.utils.make_grid(images[:10], nrow=12))
    # TODO labels is tensor object, not one-hot encoded. It can be direclty
    # printed out, or convert to list like this
    print('labels:', list(map(int, labels[:10])))
    preds = torch.argmax(model(images[:10]), 1)
    print('preds: ', list(map(int, preds)))

def test():
    model = get_Madry_model()
    model = get_LeNet5()
    
    train_dl, valid_dl, test_dl = get_mnist()

    train_MNIST_model(model, train_dl, valid_dl)

    # TODO save net.state_dict()
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    evaluate(model, test_dl)
    evaluate(model, train_dl)

if __name__ == '__main__':
    test()
