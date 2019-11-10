import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

from cleverhans.future.tf2.attacks import fast_gradient_method, projected_gradient_descent, spsa

from tqdm import tqdm
import time
import datetime

from data_utils import load_mnist, sample_and_view, load_cifar10, load_cifar10_ds
from model import get_Madry_model, get_LeNet5, dense_AE, CNN_AE, train_AE, train_CNN, test_AE


from train import custom_train, custom_advtrain, evaluate_model, evaluate_attack

def exp_train(config):
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    # a bread new cnn
    cnn = get_Madry_model()
    # dummy call to instantiate weights
    cnn(train_x[:10])
    # do the training
    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y),
                    config)

def test_free():
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    # a bread new cnn
    cnn = get_Madry_model()
    # dummy call to instantiate weights
    cnn(train_x[:10])
    # do the training
    config = get_default_config()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config['logname'] = 'free-{}'.format(current_time)

    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y),
                    config)



def test():
    # FIXME remove validation data
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    # DEBUG try lenet5
    # cnn = get_LeNet5()
    cnn = get_Madry_model()

    custom_train(cnn, (train_x, train_y))
    evaluate_model(cnn, (test_x, test_y))

    cnn(train_x[:10])
    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y))

    do_evaluate_attack(cnn, (test_x, test_y))
    # on training data
    do_evaluate_attack(cnn, (train_x[:1000], train_y[:1000]))


    ## AdvAE

    # ae = dense_AE()
    ae = CNN_AE()

    train_AE(ae, train_x)
    test_AE(ae, test_x, test_y)

    custom_advtrain(tf.keras.Sequential([ae, cnn]),
                    ae.trainable_variables,
                    (train_x, train_y), (test_x, test_y))

    evaluate_model(cnn, (test_x, test_y))

    evaluate_model(tf.keras.Sequential([ae, cnn]), (test_x, test_y))
    do_evaluate_attack(tf.keras.Sequential([ae, cnn]), (test_x, test_y))


if __name__ == '__main__':
    tfinit()
    # test()
    exp_all()
    test_free()
