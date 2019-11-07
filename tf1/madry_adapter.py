import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Activation
from keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from keras import Sequential
from keras.optimizers import Adam
import numpy as np

from tqdm import tqdm
import time

from data_utils import load_mnist, sample_and_view

from model import get_Madry_model, get_LeNet5, dense_AE, train_AE, train_CNN

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2

# This is for testing Madry's train loop and LinfPGDAttack with my keras model

def custom_train(model, attack, data, batch_size=50):
    # Setting up the optimizer
    global_step = tf.contrib.framework.get_or_create_global_step()
    # FIXME model.xent
    train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                       global_step=global_step)
    with tf.Session() as sess:
        # Initialize the summary writer, global variables, and our time counter.
        sess.run(tf.global_variables_initializer())

        # Main training loop
        npx, npy = data
        nbatch = npx.shape[0] // batch_size
        for ii in range(10000):
            start = ii * batch_size
            end = (ii+1) * batch_size
            x_batch = npx[start:end]
            y_batch = npy[start:end]
            # x_batch, y_batch = mnist.train.next_batch(batch_size)

            # Compute Adversarial Perturbations
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            # FIXME model.x_input
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            # Output to stdout
            if ii % 20 == 0:
                # FIXME model.accuracy
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                print('Step {}:'.format(ii))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            # Actual training step
            sess.run(train_step, feed_dict=adv_dict)


from madry import LinfPGDAttack
def test2():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    # cnn = get_LeNet5()
    cnn = get_Madry_model()
    class Adapter(object):
        def __init__(self, model):
            self.model = model
            self.x_input = Input((28,28,1))
            self.y_input = Input((10,))
            self.pre_softmax = keras.Sequential(model.layers[:-1])(self.x_input)
            # DEBUG
            # self.pre_softmax = model(self.x_input)
            self.xent = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.y_input, logits=self.pre_softmax))
            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.pre_softmax, 1), tf.argmax(self.y_input, 1)),
                    tf.float32))
    m = Adapter(cnn)
    attack = LinfPGDAttack(m, 0.3, 40, 0.01, True, 'xent')
    custom_train(m, attack, (train_x, train_y))
