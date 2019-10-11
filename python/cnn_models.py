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
from densenet import DenseNet

# sys.path.append('/home/XXX/github/reading/')
# pip3 install --user git+https://www.github.com/keras-team/keras-contrib.git
# from keras_contrib.keras_contrib.applications.wide_resnet import WideResidualNetwork
from keras_contrib.applications.wide_resnet import WideResidualNetwork
from keras_contrib.applications.densenet import DenseNetFCN

class CNNModel(cleverhans.model.Model):
    """This model is used inside AdvAEModel."""
    @staticmethod
    def NAME():
        raise NotImplementedError()
    def __init__(self, training):
        # CAUTION This mode indicates test mode or train mode. CIFAR
        # resnet models must inspect this value and set the
        # batch_normalization layer trainable to False when training
        # for the auto encoder
        self.training = training
        
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
            callbacks = [get_lr_reducer(), get_es()]
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
        x = keras.layers.Conv2D(32, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)
        x = keras.layers.Conv2D(64, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        # (None, 5, 5, 64)
        shape = self.CNN.output_shape
        # (4,4,64,)
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(1024)(x)
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
                           'nb_iter': 7,
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
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        # datagen = keras.preprocessing.image.ImageDataGenerator(
        #     rotation_range=10,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     horizontal_flip=True,)
        datagen.fit(train_x)
            
        with sess.as_default():
            # DEBUG resnet training broken again
            callbacks = [get_es(),
                         get_lr_reducer(),
                         # get_lr_scheduler()
            ]
            model.compile(loss=myloss,
                          metrics=[acc],
                          optimizer=keras.optimizers.Adam(lr=1e-3),
                          target_tensors=self.y)
            model.fit_generator(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),
                                validation_data=(val_x, val_y),
                                verbose=1,
                                workers=4,
                                steps_per_epoch=math.ceil(train_x.shape[0] / BATCH_SIZE),
                                epochs=100,
                                callbacks=callbacks)

class MyResNet(CifarModel):
    @staticmethod
    def NAME():
        raise NotImplementedError()
    def setup_resnet(self):
        raise NotImplementedError()
    def setup_CNN(self):
        self.setup_resnet()
        print('Creating resnet graph ..')
        model = resnet_v2(input_shape=self.xshape(), depth=self.depth, training=self.training)
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
    @staticmethod
    def NAME():
        return 'resnet29'
    def setup_resnet(self):
        n = 3 # 29
        self.depth = n * 9 + 2
class MyResNet56(MyResNet):
    @staticmethod
    def NAME():
        return 'resnet56'
    def setup_resnet(self):
        n = 6 # 56
        self.depth = n * 9 + 2
class MyResNet110(MyResNet):
    @staticmethod
    def NAME():
        return 'resnet110'
    def setup_resnet(self):
        n = 12 # 110
        self.depth = n * 9 + 2


class MyWideResNet2(CifarModel):
    @staticmethod
    def NAME():
        return "WRN2"
    def setup_CNN(self):
        model = WideResidualNetwork(include_top=False, weights=None)
        self.CNN = model
    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, x)

# class MyDenseNet(CifarModel):
#     def __init__(self):
#         super().__init__()
#     @staticmethod
#     def NAME():
#         return "densenet"
#     def setup_CNN(self):
#         depth = 7
#         nb_dense_block = 1
#         growth_rate = 12
#         nb_filter = 16
#         dropout_rate = 0.2
#         self.weight_decay = 1e-4
#         learning_rate = 1e-3
#         model = DenseNet(10,
#                          self.xshape(),
#                          depth,
#                          nb_dense_block,
#                          growth_rate,
#                          nb_filter,
#                          dropout_rate=dropout_rate,
#                          weight_decay=self.weight_decay)
#         self.CNN = model

#     def setup_FC(self):
#         shape = self.CNN.output_shape
#         inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
#         x = keras.layers.GlobalAveragePooling2D(data_format=keras.backend.image_data_format())(inputs)
#         x = keras.layers.Dense(10,
#                                # activation='softmax',
#                                # kernel_regularizer=keras.regularizers.l2(self.weight_decay),
#                                # bias_regularizer=keras.regularizers.l2(self.weight_decay)
#         )(x)
#         self.FC = keras.models.Model(inputs, x)

class MyDenseNetFCN(CifarModel):
    @staticmethod
    def NAME():
        return "densenetFCN"
    def setup_CNN(self):
        model = DenseNetFCN(self.xshape(), include_top=False)
        self.CNN = model

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, x)


def __test():
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    depth = 7
    nb_dense_block = 1
    growth_rate = 12
    nb_filter = 16
    dropout_rate = 0.2
    weight_decay = 1e-4
    learning_rate = 1e-3
    
    model = DenseNet(10,
                     train_x.shape[1:],
                     depth,
                     nb_dense_block,
                     growth_rate,
                     nb_filter,
                     dropout_rate=dropout_rate,
                     weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])
    callbacks = [get_es(), get_lr_reducer()]
    model.fit(train_x, train_y,
              batch_size=64, validation_split=0.1,
              epochs=100,
              callbacks=callbacks)
