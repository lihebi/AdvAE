import tensorflow as tf
import keras
from keras.layers import Input, Flatten, Dense, Activation, UpSampling2D
from keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from keras import Sequential
from keras.optimizers import Adam
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
    # FIXME input layer?
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
        loss=keras.losses.CategoricalCrossentropy(),
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
    return Sequential(encoder.layers + decoder.layers)


def CNN_AE():
    encoder = Sequential([Conv2D(16, 3, padding='same'),
                          ReLU(),
                          MaxPool2D((2,2))])
    decoder = Sequential([Conv2D(16, 3, padding='same'),
                          ReLU(),
                          UpSampling2D(),
                          Conv2D(1, 3, padding='same'),
                          Activation('sigmoid')])
    return Sequential(encoder.layers + decoder.layers)

def train_AE(model, x):
    model.compile(loss=keras.losses.MSE,
                  optimizer=Adam(0.01))
    model.fit(x, x, epochs=3)

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
    ae = CNN_AE()
    train_AE(ae, train_x)
    test_AE(ae, test_x, test_y)
