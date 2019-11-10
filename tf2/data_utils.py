import numpy as np
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

__all__ = ['load_mnist', 'sample_and_view']

def load_mnist():
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

    return (train_x, train_y), (test_x, test_y)


def load_cifar10():
    # 32, 32, 3, uint8
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    return (train_x, train_y), (test_x, test_y)



def sample_and_view(x, num=10):
    stacked = np.concatenate(x[:num], 1)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    if train_x.shape[1] == 28:
        arr = (np.reshape(stacked, (28,28*num)) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        # rescale to 2 because it is a bit small
        img = img.resize((img.width*3, img.height*3), Image.ANTIALIAS)
        img.save(tmp)
    elif train_x.shape[1] == 32:
        # FIXME how to scale it?
        plt.imsave(tmp, stacked, dpi=500)
    print('#<Image: ' + tmp.name + '>')


def test():
    (train_x, train_y), (test_x, test_y) = load_mnist()
    (train_x, train_y), (test_x, test_y) = load_cifar10()
    sample_and_view(train_x)
