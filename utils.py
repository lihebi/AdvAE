import keras
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
# train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)
def load_mnist_data():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    # convert data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    train_x.shape
    train_x = np.reshape(train_x, (train_x.shape[0], 28,28,1))
    test_x = np.reshape(test_x, (test_x.shape[0], 28,28,1))
    
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)
    
    # split validation
    nval = train_x.shape[0] // 10
    val_x = train_x[-nval:]
    val_y = train_y[-nval:]
    train_x = train_x[:-nval]
    train_y = train_y[:-nval]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    
def convert_image_255(img):
    return np.round(255 - (img) * 255).reshape((28, 28))
    # return np.round(img).reshape((28, 28))
    # return np.round((img + 0.5) * 255).reshape((32, 32, 3))

def grid_show_image(images, width, height, filename='out.png', titles=None):
    """
    Sample 10 images, and save it.
    """
    assert len(images) == width * height
    assert titles is None or len(images) == len(titles)
    plt.ioff()
    fig, axes = plt.subplots(nrows=height, ncols=width)
    fig.canvas.set_window_title('My Grid Visualization')
    if titles is None:
        titles = [''] * len(images)
    for image, ax, title in zip(images, axes.reshape(-1), titles):
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title)
        ax.imshow(convert_image_255(image), cmap='gray')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def __test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    grid_show_image(train_x[:10], 5, 2, ['hell'] * 10)

def mynorm(a, b, p):
    size = a.shape[0]
    delta = a.reshape((size,-1)) - b.reshape((size,-1))
    return np.linalg.norm(delta, ord=p, axis=1)
