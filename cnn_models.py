import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import cleverhans.model

from utils import *
from tf_utils import *

from attacks import CLIP_MIN, CLIP_MAX
from attacks import *
from train import *
from resnet import resnet_v2, lr_schedule

class CNNModel(cleverhans.model.Model):
    """This model is used inside AdvAEModel."""
    @staticmethod
    def NAME():
        return "mnistcnn"
    def __init__(self):
        with tf.variable_scope('my_CNN'):
            self.setup_CNN()
        with tf.variable_scope('my_FC'):
            self.setup_FC()
        self.CNN_vars = tf.trainable_variables('my_CNN')
        self.FC_vars = tf.trainable_variables('my_FC')
        
        self.x = keras.layers.Input(shape=self.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.yshape(), dtype='float32')
        
        CNN_logits = self.FC(self.CNN(self.x))
        self.model = keras.models.Model(self.x, CNN_logits)

        # no AE, just CNN
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_loss = my_softmax_xent(CNN_logits, self.y)
        self.CNN_accuracy = my_accuracy_wrapper(CNN_logits, self.y)
        
        self.setup_trainstep()

        self.setup_attack_params()
    def setup_trainstep(self):
        self.CNN_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.CNN_loss, var_list=self.CNN_vars+self.FC_vars)
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(x))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def setup_attack_params(self):
        """These parameter should change according to CNN."""
        self.FGSM_params = {'eps': 0.3}
        self.PGD_params = {'eps': 0.3,
                           'nb_iter': 40,
                           'eps_iter': 0.01}
        self.JSMA_params = {}
        self.CW_params = {'max_iterations': 1000,
                          'learning_rate': 0.2}
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        # x = keras.layers.Reshape(self.xshape())(inputs)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        # (None, 5, 5, 64)
        shape = self.CNN.output_shape
        # (4,4,64,)
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)
    def train_CNN(self, sess, train_x, train_y, patience=10, num_epoches=200):
        my_training(sess, self.x, self.y,
                    self.CNN_loss,
                    self.CNN_train_step,
                    [self.CNN_accuracy], ['accuracy'],
                    train_x, train_y,
                    num_epochs=num_epoches,
                    patience=patience)
        
    def save_weights(self, sess, path):
        """Save weights using keras API."""
        with sess.as_default():
            self.model.save_weights(path)
    def load_weights(self, sess, path):
        with sess.as_default():
            self.model.load_weights(path)
    def xshape(self):
        return (28,28,1)
    def yshape(self):
        return (10,)

class FashionCNNModel(CNNModel):
    @staticmethod
    def NAME():
        return "fashion"

class MyResNet(CNNModel):
    @staticmethod
    def NAME():
        return 'resnet'
    # def __init__(self):
    #     super().__init__()
    def setup_resnet(self):
        n = 3 # 29
        self.depth = n * 9 + 2
    def setup_CNN(self):
        self.setup_resnet()
        print('Creating resnet graph ..')
        model = resnet_v2(input_shape=self.xshape(), depth=self.depth)
        print('Resnet graph created.')
        self.CNN = model
    def setup_FC(self):
        # (None, 5, 5, 64)
        shape = self.CNN.output_shape
        # (4,4,64,)
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)

        # Method 1: Resnet convention: 1 FC layer, he_normal
        # initializer
        #
        # logits = keras.layers.Dense(
        #     # hard coding 10 for cifar10
        #     10,
        #     # removing softmax because I need the presoftmax tensor
        #     # activation='softmax',
        #     kernel_initializer='he_normal')(x)
        #
        # Method 2: traditional CNN, two FC layers
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)
    def setup_attack_params(self):
        self.FGSM_params = {'eps': 8./255}
        self.JSMA_params = {}
        self.PGD_params = {'eps': 8.0/255,
                           'nb_iter': 10,
                           'eps_iter': 2./255}
        self.CW_params = {'max_iterations': 1000,
                          'learning_rate': 0.2}
    def train_CNN(self, sess, train_x, train_y):
        print('Using data augmentation for training Resnet ..')
        augment = Cifar10Augment()
        # augment = KerasAugment(train_x)
        my_training(sess, self.x, self.y,
                    self.CNN_loss,
                    self.CNN_train_step,
                    [self.CNN_accuracy, self.global_step], ['accuracy', 'global_step'],
                    train_x, train_y,
                    num_epochs=200,
                    data_augment=augment,
                    # keras_augment=augment,
                    patience=10)
    def setup_trainstep(self):
        """Resnet needs scheduled training rate decay.
        
        values: 1e-3, 1e-1, 1e-2, 1e-3, 0.5e-3
        boundaries: 0, 80, 120, 160, 180
        
        "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
        
        # schedules = [[0, 0.1], [400, 0.01], [600, 0.001]]
        # schedules = [[0, 1e-3], [400, 1e-4], [800, 1e-5], [1200, 1e-6]]
        """
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        # These numbers are batches
        schedules = [[0, 1e-3], [400*30, 1e-4], [400*60, 1e-5]]
        boundaries = [s[0] for s in schedules][1:]
        values = [s[1] for s in schedules]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values)
        self.CNN_train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(
                self.CNN_loss,
                global_step=global_step,
                var_list=self.CNN_vars+self.FC_vars)
        # self.CNN_train_step = tf.train.AdamOptimizer(0.001).minimize(
        #     self.CNN_loss, var_list=self.CNN_vars+self.FC_vars)
    def xshape(self):
        return (32,32,3,)
    def yshape(self):
        return (10,)
class MyResNet56(MyResNet):
    def NAME():
        return 'resnet56'
    # def __init__(self):
    #     super()._init__()
    def setup_resnet(self):
        n = 6 # 56
        self.depth = n * 9 + 2
class MyResNet110(MyResNet):
    def NAME():
        return 'resnet110'
    # def __init__(self):
    #     super()._init__()
    def setup_resnet(self):
        n = 12 # 110
        self.depth = n * 9 + 2

