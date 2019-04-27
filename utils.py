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

def grid_show_image(images, width, height, filename='out.png'):
    """
    Sample 10 images, and save it.
    """
    assert len(images) == width * height
    plt.ioff()
    figure = plt.figure()
    # figure = plt.figure(figsize=(6.4, 1.2))
    figure.canvas.set_window_title('My Grid Visualization')
    for x in range(height):
        for y in range(width):
            # print(x,y)
            # figure.add_subplot(height, width, x*width + y + 1)
            ax = plt.subplot(height, width, x*width + y + 1)
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.set_visible(False)
            # plt.axis('off')
            # plt.imshow(images[x*width+y], cmap='gray')
            plt.imshow(convert_image_255(images[x*width+y]), cmap='gray')
            # plt.imshow(convert_image_255(images[x*width+y]))
            # plt.imshow(images[x*width+y])
    # plt.show()
    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.savefig(filename)
    return figure
