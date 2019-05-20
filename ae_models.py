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

def my_ae_conv_layer(inputs, f, k, p, batch_norm=True, up=False):
    """
    f: num of filters
    k: kernel size
    p: pooling size

    example config:
    32, (3,3), (2,2)
    """
    x = keras.layers.Conv2D(f, k, padding='same')(inputs)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    if up:
        x = keras.layers.UpSampling2D(p)(x)
    else:
        x = keras.layers.MaxPooling2D(p, padding='same')(x)
    return x


class AEModel():
    """This is the default model for mnist."""
    @staticmethod
    def NAME():
        return "ae"
    def __init__(self, cnn_model):
        """I have cnn model here because I might use CNN model to train this AE."""
        # FIXME I need to make sure these setups are only called
        # once. Otherwise I need to set the AE_vars variables accordingly

        self.cnn_model = cnn_model
        self.shape = self.cnn_model.xshape()
        
        with tf.variable_scope('my_AE'):
            self.setup_AE()
            
        self.AE_vars = tf.trainable_variables('my_AE')
        self.additional_train_steps = [self.AE.updates]

    def setup_AE(self):
        """From noise_x to x.

        Different numbers for AE:

        My Default:
        32, (3,3), (2,2)
        32, (3,3), (2,2)
        ---
        32, (3,3), (2,2)
        32, (3,3), (2,2)
        ---
        3, (3,3)

        """
        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        # inputs = keras.layers.Input(shape=(28,28,1), dtype='float32')
        # denoising
        x = my_ae_conv_layer(inputs, 32, (3,3), (2,2))
        encoded = my_ae_conv_layer(x, 32, (3,3), (2,2))
        # at this point the representation is (7, 7, 32)
        x = my_ae_conv_layer(encoded, 32, (3,3), (2,2), batch_norm=False, up=True)
        x = my_ae_conv_layer(x, 32, (3,3), (2,2), batch_norm=False, up=True)

        # channel = 1 or 3 depending on dataset
        channel = self.shape[2]
        decoded = keras.layers.Conv2D(channel, (3, 3), padding='same')(x)
        
        self.AE1 = keras.models.Model(inputs, decoded)
        
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))
        
        
    def save_weights(self, sess, path):
        """Save weights using keras API."""
        with sess.as_default():
            self.AE.save_weights(path)
    def load_weights(self, sess, path):
        with sess.as_default():
            self.AE.load_weights(path)
class CifarAEModel(AEModel):
    """This autoencoder has more capacity"""
    @staticmethod
    def NAME():
        return "cifarae"
    # def __init__(self, cnn_model):
    #     super().__init__(cnn_model)
    def setup_AE(self):
        """
        64, (3,3), (2,2)
        32, (3,3), (2,2)
        16, (3,3), (2,2)
        ---
        16, (3,3), (2,2)
        32, (3,3), (2,2)
        64, (3,3), (2,2)
        ---
        3, (3,3)
        
        """
        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        # inputs = keras.layers.Input(shape=(28,28,1), dtype='float32')
        # denoising
        x = my_ae_conv_layer(inputs, 64, (3,3), (2,2))
        x = my_ae_conv_layer(x, 32, (3,3), (2,2))
        encoded = my_ae_conv_layer(x, 16, (3,3), (2,2))
        
        x = my_ae_conv_layer(encoded, 16, (3,3), (2,2), batch_norm=False, up=True)
        x = my_ae_conv_layer(x, 32, (3,3), (2,2), batch_norm=False, up=True)
        x = my_ae_conv_layer(x, 64, (3,3), (2,2), batch_norm=False, up=True)

        # channel = 1 or 3 depending on dataset
        channel = self.shape[2]
        decoded = keras.layers.Conv2D(channel, (3, 3), padding='same')(x)
        
        self.AE1 = keras.models.Model(inputs, decoded)
        
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))

class TestCifarAEModel(AEModel):
    """This autoencoder has more capacity"""
    @staticmethod
    def NAME():
        return "testcifarae"
    # def __init__(self, cnn_model):
    #     super().__init__(cnn_model)
    def setup_AE(self):
        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        # inputs = keras.layers.Input(shape=(32,32,3), dtype='float32')
        # denoising
        x = my_ae_conv_layer(inputs, 32, (3,3), (2,2))
        x = my_ae_conv_layer(x, 64, (3,3), (2,2))
        # x = my_ae_conv_layer(x, 128, (3,3), (2,2))
        encoded = my_ae_conv_layer(x, 128, (3,3), (2,2))
        
        x = my_ae_conv_layer(encoded, 128, (3,3), (2,2), batch_norm=False, up=True)
        # x = my_ae_conv_layer(x, 128, (3,3), (2,2), batch_norm=False, up=True)
        x = my_ae_conv_layer(x, 64, (3,3), (2,2), batch_norm=False, up=True)
        x = my_ae_conv_layer(x, 32, (3,3), (2,2), batch_norm=False, up=True)

        # channel = 1 or 3 depending on dataset
        channel = self.shape[2]
        decoded = keras.layers.Conv2D(channel, (3, 3), padding='same')(x)
        
        self.AE1 = keras.models.Model(inputs, decoded)
        
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))
class DunetModel(AEModel):
    """Unet TODO"""
    @staticmethod
    def NAME():
        # FIXME change NAME to CLSNAME to avoid bugs
        return "dunet"
    # def __init__(self, cnn_model):
    #     super().__init__(cnn_model)
    def setup_AE(self):
        """
        inputs = Input((32, 32, 3))
        """
        # inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        inputs = keras.layers.Input(shape=(32,32,3), dtype='float32')

        conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = keras.layers.concatenate(
            [keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        up7 = keras.layers.concatenate(
            [keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        up8 = keras.layers.concatenate(
            [keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        up9 = keras.layers.concatenate(
            [keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        # conv10 = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)        
        conv10 = keras.layers.Conv2D(3, (3, 3), padding='same')(conv9)        
        # denoising
        # x = my_ae_conv_layer(inputs, 32, (3,3), (2,2))
        # encoded = my_ae_conv_layer(x, 32, (3,3), (2,2))
        # # at this point the representation is (7, 7, 32)
        # x = my_ae_conv_layer(encoded, 32, (3,3), (2,2), batch_norm=False, up=True)
        # x = my_ae_conv_layer(x, 32, (3,3), (2,2), batch_norm=False, up=True)


        # channel = 1 or 3 depending on dataset
        # channel = self.shape[2]
        # decoded = keras.layers.Conv2D(channel, (3, 3), padding='same')(x)

        # DEBUG XXX TODO HEBI NOW use add?
        decoded = keras.layers.Add()([inputs, conv10])
        # decoded = keras.layers.Subtract()([inputs, conv10])
        # DEBUG what if I just use this as output?
        # decoded = conv10
        self.AE1 = keras.models.Model(inputs, decoded)
        
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))

class IdentityAEModel(AEModel):
    """This model implementation is a HACK. The AE model does not have any
parameters. It is merely an identity map. However, it still supports
the load and save function, in which the underline CNN model
parameters are saved and loaded. The AE_vars are set to the CNN/FC
vars, so that adv training would change those instead.

    """
    def setup_AE(self):
        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        outputs = keras.layers.Lambda(lambda x: x, output_shape=self.shape)(inputs)

        # FIMXE this is dummy
        self.AE1 = keras.models.Model(inputs, outputs)
        self.AE = keras.models.Model(inputs, outputs)
    @staticmethod
    def NAME():
        return "identityAE"
    def __init__(self, cnn_model):
        """I have cnn model here because I might use CNN model to train this AE."""
        # FIXME I need to make sure these setups are only called
        # once. Otherwise I need to set the AE_vars variables accordingly

        self.cnn_model = cnn_model
        self.shape = self.cnn_model.xshape()
        
        with tf.variable_scope('my_AE'):
            self.setup_AE()
            
        # self.AE_vars = tf.trainable_variables('my_AE')
        # self.AE_vars = self.cnn_model.CNN_vars + self.cnn_model.FC_vars
    def save_weights(self, sess, path):
        self.cnn_model.save_weights(sess, path)
    def load_weights(self, sess, path):
        self.cnn_model.load_weights(sess, path)
