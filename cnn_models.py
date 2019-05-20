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
        raise NotImplementedError()
    def __init__(self):
        with tf.variable_scope('my_CNN'):
            self.setup_CNN()
        with tf.variable_scope('my_FC'):
            self.setup_FC()
        self.CNN_vars = tf.trainable_variables('my_CNN')
        self.FC_vars = tf.trainable_variables('my_FC')
        
        self.x = keras.layers.Input(shape=self.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.yshape(), dtype='float32')

        self.logits = self.FC(self.CNN(self.x))
        self.model = keras.models.Model(self.x, self.logits)

        self.setup_attack_params()
    def xshape(self):
        raise NotImplementedError()
    def yshape(self):
        raise NotImplementedError()
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(x))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def setup_attack_params(self):
        raise NotImplementedError()
    def setup_CNN(self):
        raise NotImplementedError()
    def setup_FC(self):
        raise NotImplementedError()
    
    def train_CNN(self, sess, train_x, train_y, patience=10, num_epoches=200):
        # (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)
        
        # outputs = keras.layers.Activation('softmax')(self.FC(self.CNN(self.x)))
        # model = keras.models.Model(self.x, outputs)
        model = self.model
        
        # no AE, just CNN
        logits = self.FC(self.CNN(self.x))
        loss = my_softmax_xent(logits, self.y)
        accuracy = my_accuracy_wrapper(logits, self.y)
        
        def myloss(ytrue, ypred):
            return loss
        def acc(ytrue, ypred): return accuracy
            
        with sess.as_default():
            callbacks = [get_lr_scheduler(),
                         get_lr_reducer(patience=5),
                         get_es(patience=10),
                         get_mc('best_model.hdf5')]
            model.compile(loss=myloss,
                          metrics=[acc],
                          optimizer=keras.optimizers.Adam(lr=1e-3),
                          target_tensors=self.y)
            model.fit(train_x, train_y,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      validation_split=0.1,
                      epochs=100,
                      callbacks=callbacks)
            model.load_weights('best_model.hdf5')
        
    def save_weights(self, sess, path):
        """Save weights using keras API."""
        with sess.as_default():
            self.model.save_weights(path)
    def load_weights(self, sess, path):
        with sess.as_default():
            self.model.load_weights(path)


class MNISTModel(CNNModel):
    @staticmethod
    def NAME():
        return "mnistcnn"
    def xshape(self):
        return (28,28,1)
    def yshape(self):
        return (10,)
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
 
# class FashionModel(MNISTModel):
#     @staticmethod
#     def NAME():
#         return "fashion"

class CifarModel(CNNModel):
    def NAME():
        raise NotImplementedError()

    def setup_attack_params(self):
        self.FGSM_params = {'eps': 8./255}
        self.JSMA_params = {}
        self.PGD_params = {'eps': 8.0/255,
                           'nb_iter': 10,
                           'eps_iter': 2./255}
        self.CW_params = {'max_iterations': 1000,
                          'learning_rate': 0.2}
    def xshape(self):
        return (32,32,3,)
    def yshape(self):
        return (10,)
    def train_CNN(self, sess, train_x, train_y, patience=10, num_epoches=200):
        (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)
        
        model = self.model
        
        # no AE, just CNN
        logits = self.FC(self.CNN(self.x))
        loss = my_softmax_xent(logits, self.y)
        accuracy = my_accuracy_wrapper(logits, self.y)
        
        def myloss(ytrue, ypred):
            return loss
        def acc(ytrue, ypred): return accuracy

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,)
        datagen.fit(train_x)
            
        with sess.as_default():
            callbacks = [get_lr_scheduler(),
                         get_lr_reducer(patience=5),
                         get_es(patience=10),
                         get_mc('best_model.hdf5')]
            model.compile(loss=myloss,
                          metrics=[acc],
                          optimizer=keras.optimizers.Adam(lr=1e-3),
                          target_tensors=self.y)
            model.fit_generator(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),
                                validation_data=(val_x, val_y),
                                verbose=1,
                                workers=4,
                                steps_per_epoch=math.ceil(train_x.shape[0] / BATCH_SIZE),
                                # batch_size=BATCH_SIZE,
                                # validation_split=0.1,
                                epochs=100,
                                callbacks=callbacks)
            model.load_weights('best_model.hdf5')

class MyResNet(CifarModel):
    @staticmethod
    def NAME():
        raise NotImplementedError()
    def setup_resnet(self):
        raise NotImplementedError()
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
        logits = keras.layers.Dense(10, kernel_initializer='he_normal')(x)
        self.FC = keras.models.Model(inputs, logits)
class MyResNet29(MyResNet):
    def NAME():
        return 'resnet29'
    def setup_resnet(self):
        n = 3 # 29
        self.depth = n * 9 + 2
class MyResNet56(MyResNet):
    def NAME():
        return 'resnet56'
    def setup_resnet(self):
        n = 6 # 56
        self.depth = n * 9 + 2
class MyResNet110(MyResNet):
    def NAME():
        return 'resnet110'
    def setup_resnet(self):
        n = 12 # 110
        self.depth = n * 9 + 2

class VGGModel():
    # TODO
    pass


from wrn import *
def __test():
    m = MyWideResNet()

class MyWideResNet(CifarModel):
    def __init__(self):
        super().__init__()
    @staticmethod
    def NAME():
        return "WRN"
    def setup_CNN(self):
        # WRN-16-8
        # input_dim = (32, 32, 3)
        input_dim = self.xshape()
        # param N: Depth of the network. Compute N = (n - 4) / 6.
        #           Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
        N = 2
        # param k: Width of the network.
        k = 8
        nb_classes=10
        # param dropout: Adds dropout if value is greater than 0.0
        dropout=0.00

        channel_axis = -1

        ip = Input(shape=input_dim)

        x = initial_conv(ip)
        nb_conv = 4

        x = expand_conv(x, 16, k)
        nb_conv += 2

        for i in range(N - 1):
            x = conv1_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = expand_conv(x, 32, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv2_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = expand_conv(x, 64, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv3_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = keras.layers.AveragePooling2D((8, 8))(x)
        
        # x = Flatten()(x)

        model = Model(ip, x)
        self.CNN = model
    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = Dense(10, kernel_regularizer=l2(weight_decay))(x)
        self.FC = keras.models.Model(inputs, x)
