import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

from data_utils import load_mnist, sample_and_view

from model import get_Madry_model, get_LeNet5, dense_AE, train_AE, train_CNN

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2

def FGSM(model, x, y=None, params=dict()):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    fgsm_params.update(params)
    # FIXME Cleverhans has a bug in Attack.get_or_guess_labels: it
    # only checks y is in the key, but didn't check whether the value
    # is None. fgsm.generate calls that function
    if y is not None:
        fgsm_params.update({'y': y})
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    return tf.stop_gradient(adv_x)

def PGD(model, x, y=None, params=dict()):
    # DEBUG I would always want to supply the y manually
    # assert y is not None

    # FIXME cleverhans requires the model to be instantiated from
    # cleverhans.Model class. This is ugly. I resist to do that. The tf2
    # implementation fixed that, so I'd try tf2 version first. Stopping here
    pgd = ProjectedGradientDescent(model)
    pgd_params = {'eps': 0.3,
                  # CAUTION I need this, otherwise the adv acc data is
                  # not accurate
                  'y': y,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': 0.,
                  'clip_max': 1.}
    pgd_params.update(params)
    adv_x = pgd.generate(x, **pgd_params)
    # DEBUG do I need to stop gradients here?
    return tf.stop_gradient(adv_x)


def _advmodel(model):
    # I'll need to use functional API
    x = tf.keras.Input(shape=(28,28,1))
    adv = PGD(model, x)
    # FIXME maybe no need to stop gradients?
    adv = tf.stop_gradient(adv)
    y = model(adv)
    return tf.keras.Model(inputs, y)

def advtrain(model, data):
    x, y = data
    # need a custom loss function
    # If using tf1, basically two ways:
    # 1. I need to have a very special training loop, that generate adv examples in the loop
    # 2. I need to wrap adv generation as a tensor, stop gradients, and use that in loss function
    #
    # generate adv

    model = _advmodel(model)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # loss='categorical_crossentropy',
        optimizer=Adam(0.01),
        metrics=['accuracy'])
    model.fit(x, y, epochs=5, batch_size=32)

def test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    cnn = get_LeNet5()
    # cnn = get_Madry_model()

    train_CNN(cnn, (train_x, train_y))

    # cnn.predict(val_x).shape
    cnn.evaluate(test_x, test_y)

    advtrain(cnn, (train_x, train_y))
