import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10

import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

__all__ = ['imrepl', 'get_mnist', 'get_cifar10']

def _imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imrepl(img):
    """Show image in Emacs repl"""
    npimg = img.cpu().numpy()
    # plt.ioff()
    fig = plt.figure(dpi=150)
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(tmp,
                format='png',
                bbox_inches='tight',
                # transparent=True,
                pad_inches=0)

    print('#<Image: ' + tmp.name + '>')
    plt.close()

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def _gpu(dl):
    return WrappedDataLoader(dl, lambda x, y: (x.to(device), y.to(device)))

def valid_split(ds, valid_ratio=0.1):
    valid = int(0.1 * len(ds))
    train = len(ds) - valid
    train_ds, valid_ds = random_split(ds, [train, valid])
    return train_ds, valid_ds

def get_mnist(batch_size=128, cachedir='./cache'):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    # ds stands for dataset
    ds = MNIST(root=cachedir, train=True, download=True, transform=transform)
    train_ds, valid_ds = valid_split(ds, 0.1)
    test_ds = MNIST(root=cachedir, train=False, download=True, transform=transform)

    # dl stands for dataloader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    # TODO batch size to be doubled, because validation has no backward pass
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    # (optional) WrappedDataLoader, preprocess, reshape
    # valid_dl = WrappedDataLoader(train_dl,
    #                              lambda x, y: (x.view(-1, 1, 28, 28), y))
    return _gpu(train_dl), _gpu(valid_dl), _gpu(test_dl)

def get_cifar10(batch_size=128, cachedir='./cache'):
    """
    TODO data augmentation
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = CIFAR10(root=cachedir, train=True, download=True, transform=transform)
    train_ds, valid_ds = valid_split(ds, 0.1)
    test_ds = CIFAR10(root=cachedir, train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # FIXME trainset.classes
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return _gpu(train_dl), _gpu(valid_dl), _gpu(test_dl)

def test_use():

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    _imshow(torchvision.utils.make_grid(images))
    imrepl(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

