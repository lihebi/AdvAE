import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# tqdm has too many bugs
# from tqdm import tqdm
# from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm
import time

from data_utils import imrepl, get_mnist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_MNIST_FC_model():
    model = nn.Sequential(
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(28*28, 32),
        nn.ReLU(),
        nn.Linear(32, 10))
    return model.to(device)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def get_MNIST_CNN_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
        # nn.AvgPool2d(4),
        # nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Linear(3*3*32, 10))
    return model.to(device)

def train(model, opt, loss_fn, dl, cb=None, epoch=10):
    """
    cb(iter, loss)
    """
    for _ in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        tqdm._instances.clear()
        for i, data in enumerate(tqdm(dl), 0):
            inputs, labels = data

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            opt.step()
            opt.zero_grad()

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

def evaluate(model, dl):
    """Run model and compute accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        tqdm._instances.clear()
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

def test():
    model = get_MNIST_FC_model()
    model = get_MNIST_CNN_model()
    
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # this (and NLLLoss in general) expects label as target, thus no need to
    # one-hot encoding
    loss_fn = nn.CrossEntropyLoss()

    train_dl, valid_dl, test_dl = get_mnist()

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

    # TODO save net.state_dict()
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    evaluate(model, test_dl)
    evaluate(model, train_dl)

if __name__ == '__main__':
    test()
