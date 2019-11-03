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

__all__ = ['Net', 'train']

# import os
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'relative/path/to/file/you/want')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Mnist_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, xb):
        xb = xb.view(-1, 28*28)
        xb = F.relu(self.fc1(xb))
        xb = F.softmax(self.fc2(xb))
        return xb.view(-1, 10)

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def train(model, opt, loss, trainloader):
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            opt.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            opt.step()

            # print statistics
            running_loss += l.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('\n[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def evaluate(model, testloader):
    """Run model and compute accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # TODO visualize
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # print images
    # imrepl(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

def test():
    model = Mnist_CNN()
    model = Mnist_FC()

    model.to(device)
    
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    trainloader, testloader = get_mnist()

    train(model, opt, loss, trainloader)

    # TODO save net.state_dict()
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    evaluate(model, testloader)

if __name__ == '__main__':
    test()
