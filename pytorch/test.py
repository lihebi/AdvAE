import torch
import torchvision
import torch.nn.functional as F

# import os
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'relative/path/to/file/you/want')


def test_manual_data():
    from pathlib import Path
    import requests

    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    import pickle
    import gzip

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")



net = Net()

def test_tqdm():
    for i,_ in enumerate(tqdm(range(10000))):
        time.sleep(0.01)

    # clear tqdm bug
    tqdm._instances.clear()

    # FIXME elpy repl problem
    for i in tqdm(range(100)):
        time.sleep(0.01)
    for i in range(100):
        time.sleep(0.1)
    with tqdm(total=100) as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)

# TODO sequential
def get_CNN_model():
    model = nn.Sequential(
        Lambda(preprocess),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Lambda(lambda x: x.view(x.size(0), -1)))
    return model

# Unknown model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MNIST_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, xb):
        xb = xb.view(-1, 28*28)
        xb = F.relu(self.fc1(xb))
        xb = F.softmax(self.fc2(xb))
        return xb.view(-1, 10)

class MNIST_CNN(nn.Module):
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

def get_MNIST_CNN_model():
    model = MNIST_CNN()
    return model.to(device)
# device = torch.device("cpu")
# def gpu(data):
#     return map(lambda x: x.to(device), data)
