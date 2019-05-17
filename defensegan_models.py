from cnn_models import CNNModel
import keras
from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Dropout
import tensorflow as tf
import numpy as np
import sys
import math
            
class DefenseGAN_a(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_a"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(64, 5, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 5, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dropout(rate=0.25)(x)
        x = keras.layers.Dense(128)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)

class DefenseGAN_b(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_b"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = keras.layers.Dropout(rate=0.2)(x)
        x = Conv2D(64, 8, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 6, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 5, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)

class DefenseGAN_c(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_c"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(128, 3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dropout(rate=0.25)(x)
        x = keras.layers.Dense(128)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)

class DefenseGAN_d(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_d"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        outputs = keras.layers.Lambda(lambda x: x, output_shape=self.xshape())(inputs)
        self.CNN = keras.models.Model(inputs, outputs)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        # FIXME the paper has this dropout while code doesn't
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)
class DefenseGAN_e(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_e"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        outputs = keras.layers.Lambda(lambda x: x, output_shape=self.xshape())(inputs)
        self.CNN = keras.models.Model(inputs, outputs)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)

class DefenseGAN_f(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_f"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(64, 8, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 6, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 5, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)


class DefenseGAN_y(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_y"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(64, 3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(256)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)
    
class DefenseGAN_q(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_q"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(32, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(256)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)

class DefenseGAN_z(CNNModel):
    @staticmethod
    def NAME():
        return "DefenseGAN_z"
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        x = inputs
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(32, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides=2, padding='valid')(x)
        x = Activation('relu')(x)
        self.CNN = keras.models.Model(inputs, x)

    def setup_FC(self):
        shape = self.CNN.output_shape
        inputs = keras.layers.Input(batch_shape=shape, dtype='float32')
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(600)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(600)(x)
        x = Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        logits = keras.layers.Dense(10)(x)
        self.FC = keras.models.Model(inputs, logits)
