import numpy as np
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from keras import datasets
from keras.utils import to_categorical

__all__ = ['load_mnist', 'sample_and_view']

def validation_split(train_x, train_y, valid_ratio):
    nval = int(train_x.shape[0] * valid_ratio)
    val_x = train_x[-nval:]
    val_y = train_y[-nval:]
    train_x = train_x[:-nval]
    train_y = train_y[:-nval]
    return (train_x, train_y), (val_x, val_y)

def load_mnist(valid_ratio=0.1):
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    # convert data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    # train_x.shape
    # FIXME should I have the channel here?
    train_x = np.reshape(train_x, (train_x.shape[0], 28,28,1))
    test_x = np.reshape(test_x, (test_x.shape[0], 28,28,1))

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y, valid_ratio)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def sample_and_view(x, num=10):
    stacked = np.concatenate(x[:num], 1)
    arr = (np.reshape(stacked, (28,28*num)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')

    # rescale to 2 because it is a bit small
    img = img.resize((img.width*3, img.height*3), Image.ANTIALIAS)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp)
    print('#<Image: ' + tmp.name + '>')


def test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    x, y = train_x, train_y
    sample_and_view(train_x)
