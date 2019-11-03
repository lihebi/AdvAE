import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

# __all__ = ['imrepl', 'get_mnist', 'get_cifar10']

def _imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imrepl(img):
    """Show image in Emacs repl"""
    npimg = img.numpy()
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

def get_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./datacache', train=True,
                                          download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='./datacache', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False)
    return trainloader, testloader

def get_cifar10():
    """
    FIXME split train and val
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    # FIXME trainset.classes
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader

def test_use():

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    _imshow(torchvision.utils.make_grid(images))
    imrepl(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

