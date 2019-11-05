import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


from data_utils import load_mnist, sample_and_view

__all__ = ['get_Madry_model', 'get_LeNet5', 'dense_AE', 'train_CNN', 'train_AE']

def get_Madry_model():
    model = Sequential([Conv2D(32, 5, padding='same'),
                        ReLU(),
                        MaxPool2D((2,2)),
                        Conv2D(64, kernel_size=5, padding='same'),
                        ReLU(),
                        MaxPool2D((2,2)),
                        Flatten(),
                        # NOTE: verified this is 7*7*64
                        Dense(1024),
                        ReLU(),
                        Dense(10),
                        Softmax()])
    return model

def get_LeNet5():
    model = Sequential([Conv2D(filters=32, kernel_size=3, padding='same'),
                        ReLU(),
                        MaxPool2D((2,2)),
                        Conv2D(64, kernel_size=3, padding='same'),
                        ReLU(),
                        MaxPool2D((2,2)),
                        Flatten(),
                        # NOTE: verified this is 7*7*64
                        Dense(200),
                        ReLU(),
                        Dense(10),
                        Softmax()])
    return model

def train_CNN(model, data):
    x, y = data
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # loss='categorical_crossentropy',
        optimizer=Adam(0.01),
        metrics=['accuracy'])
    model.fit(x, y, epochs=5, batch_size=32)

def dense_AE():
    encoder = Sequential([Flatten(),
                          Dense(32),
                          ReLU()])
    decoder = Sequential([Dense(28*28),
                          Reshape((28,28,1)),
                          Activation('sigmoid')])
    return Sequential([encoder, decoder])

def train_AE(model, x):
    ae.compile(loss=tf.keras.losses.MSE,
               optimizer=Adam(0.01))
    ae.fit(train_x, train_x, epochs=3)

def test_AE(model, x, y):
    # imrepl(torchvision.utils.make_grid(images))
    print('original:')
    sample_and_view(x[:10])
    decoded = model.predict(x)
    print('decoded:')
    sample_and_view(decoded[:10])
    print('labels:', np.argmax(y[:10], 1))


def test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    cnn = get_LeNet5()
    # cnn = get_Madry_model()

    train_CNN(cnn, (train_x, train_y))

    # cnn.predict(val_x).shape
    cnn.evaluate(test_x, test_y)

    ae = dense_AE()
    train_AE(ae, train_x)
    test_AE(ae, test_x, test_y)
