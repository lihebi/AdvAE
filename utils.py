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
    return (train_x, train_y), (test_x, test_y)

def load_fashion_mnist_data():
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
    # convert data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    train_x.shape
    train_x = np.reshape(train_x, (train_x.shape[0], 28,28,1))
    test_x = np.reshape(test_x, (test_x.shape[0], 28,28,1))
    
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)
    return (train_x, train_y), (test_x, test_y)

def validation_split(train_x, train_y):
    nval = train_x.shape[0] // 10
    val_x = train_x[-nval:]
    val_y = train_y[-nval:]
    train_x = train_x[:-nval]
    train_y = train_y[:-nval]
    return (train_x, train_y), (val_x, val_y)
    
def __test():
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()

def load_cifar10_data():
    # 32, 32, 3, uint8
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    # convert data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    # train_x = np.reshape(train_x, (train_x.shape[0], 32, 32,3))
    # test_x = np.reshape(test_x, (test_x.shape[0], 32, 32,3))
    
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)
    
    return (train_x, train_y), (test_x, test_y)
    
def convert_image_255(img):
    return np.round(255 - (img) * 255).reshape((28, 28))
    # return np.round(img).reshape((28, 28))
    # return np.round((img + 0.5) * 255).reshape((32, 32, 3))

def grid_show_image(images,
                    # width, height,
                    filename='out.pdf',
                    titles=None,
                    fringes=None):
    """
    Sample 10 images, and save it.

    images: a 2d matrix of images
    titles: a 2d matrix of strings
    fringes: a 1d strings
    """
    # assert len(images) == width * height
    # assert titles is None or len(images) == len(titles)
    plt.ioff()

    images = np.array(images)
    height = images.shape[0]
    width = images.shape[1]
    
    if fringes is not None:
        fringes = np.array(fringes)
        assert len(fringes) == images.shape[0]
    else:
        fringes = np.array([''] * height)
        
    if titles is not None:
        titles = np.array(titles)
        assert titles.shape[:2] == images.shape[:2]
    else:
        titles = np.array([['']*width]*height)


    # try to be smart on different kinds of images
    # images = np.array(images)
    if images.shape[2] == 28:
        # MNIST
        # images = images.reshape((-1, 28, 28))
        images = images.reshape(images.shape[:-1])
        images = 1 - images
    elif images.shape[2] == 32:
        # CIFAR10, it has three channels, and cmap will be ignored
        pass
    else:
        # TODO IMAGENET
        assert False

    # the size of image should increase as the number of images
    fig_width = 6.4 * (width+1) / 4
    fig_height = 4.8 * height / 3
    
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=300)

    for i in range(height):
        ax = fig.add_subplot(height, width+1, i*(width+1)+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.text(0.5,0.5,fringes[i],
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        
    for i in range(height):
        for j in range(width):
            ax = fig.add_subplot(height, width+1, i*(width+1)+(j+1)+1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(titles[i][j], loc='left')
            ax.imshow(images[i][j], cmap='gray')

    
    # fig, axes = plt.subplots(nrows=height, ncols=width,
    #                          # figsize=(12.8, 9.6),
    #                          figsize=(fig_width, fig_height),
    #                          dpi=300)
    
    fig.canvas.set_window_title('My Grid Visualization')
    
    # if titles is None:
    #     titles = [''] * len(images)

    # for image, ax, title in zip(images, axes.reshape(-1), titles):
    #     # ax.set_axis_off()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     # TITLE has to appear on top. I want it to be on bottom, so using xlabel
    #     ax.set_title(title, loc='left')
    #     # ax.set_xlabel(title)
    #     # ax.imshow(convert_image_255(image), cmap='gray')
    #     # ax.imshow(image)
    #     ax.imshow(image, cmap='gray')
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    train_x[:10].shape
    (5,2) + train_x[:10].shape[2:]
    train_x[:10].reshape((5,2) + train_x.shape).shape
    (5,2) + train_x.shape[1:]
    train_x[:10].reshape((5,2) + train_x.shape[1:]).shape
    
    grid_show_image(train_x[:10].reshape((5,2) + train_x.shape[1:]),
                    titles=([['hell']*2]*5),
                    fringes=['fringe'] * 5)

    (train_x, train_y), (test_x, test_y) = load_cifar10_data()

    plt.imshow(train_x[0].reshape((28,28)), cmap='gray')
    plt.show()
    

def mynorm(a, b, p):
    size = a.shape[0]
    delta = a.reshape((size,-1)) - b.reshape((size,-1))
    return np.linalg.norm(delta, ord=p, axis=1)

