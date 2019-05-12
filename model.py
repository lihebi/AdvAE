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
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2


from utils import *

from attacks import CLIP_MIN, CLIP_MAX
from attacks import *
from train import *
from resnet import resnet_v2, lr_schedule

def my_add_noise(x, noise_factor=0.5):
    # noise_factor = 0.5
    noisy_x = x + noise_factor * tf.random.normal(shape=tf.shape(x), mean=0., stddev=1.)
    noisy_x = tf.clip_by_value(noisy_x, CLIP_MIN, CLIP_MAX)
    return noisy_x

def my_sigmoid_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))

def my_softmax_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))

def my_accuracy_wrapper(logits=None, labels=None):
    return tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1),
        tf.argmax(labels, 1)), dtype=tf.float32))

def my_compute_accuracy(sess, x, y, preds, x_test, y_test):
    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
        dtype=tf.float32))
    return sess.run(acc, feed_dict={x: x_test, y: y_test})

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
        # FIXME this also assigns self.conv_layers
        x = keras.layers.Reshape(self.xshape())(inputs)
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
    def train_CNN(self, sess, train_x, train_y):
        my_training(sess, self.x, self.y,
                    self.CNN_loss,
                    self.CNN_train_step,
                    [self.CNN_accuracy], ['accuracy'],
                    train_x, train_y,
                    num_epochs=200,
                    patience=10)
        
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


def __test():
    sys.path.append('/home/hebi/github/reading/')
    from cifar10_challenge import model as madry_resnet
    madrynet = madry_resnet.Model(mode='train')
    
    madrynet.x_input
    keras.layers.Lambda()

    sys.path.append('/home/hebi/github/reading/tensorflow-models')
    from official.resnet.keras.resnet_cifar_model import resnet56
    # the problem of using this model is that, the softmax is
    # applied. I have to modify the function to get a pre-softmax
    # tensor.

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
        schedules = [[0, 1e-3], [400*30, 1e-4], [400*40, 1e-5], [400*50, 1e-6]]
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


    
def __test():
    sess = tf.Session()
    model = MNIST_CNN()
    loss = model.cross_entropy()
    my_adv_training(sess, model, loss)

def restore_model(path):
    tf.reset_default_graph()
    sess = tf.Session()
    model = MNIST_CNN()
    saver = tf.train.Saver()
    saver.restore(sess, path)
    print("Model restored.")
    return sess, model

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

        self.x = keras.layers.Input(shape=self.cnn_model.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.cnn_model.yshape(), dtype='float32')

        noisy_x = my_add_noise(self.x, noise_factor=0.5)

        # print(self.AE1(noisy_x).shape)
        # print(self.x.shape)

        ##############################
        ## noisy
        
        # noisy, xent, pixel
        # loss = my_sigmoid_xent(self.AE1(noisy_x), self.x)
        
        # noisy, l2, pixel, 0.3
        # loss = tf.reduce_mean(tf.square(self.AE(noisy_x) - self.x))
        
        # noisy, l1, pixel
        # loss = tf.reduce_mean(tf.abs(self.AE(noisy_x) - self.x))

        # noisy+clean, l1, pixel
        # testing whether noisy loss can be used as regularizer
        # 0.42
        # mu = 0.5
        # 0.48
        # mu = 0.2
        # loss = (mu * tf.reduce_mean(tf.abs(self.AE(noisy_x) - self.x))
        #         + (1 - mu) * tf.reduce_mean(tf.abs(self.AE(self.x) - self.x)))
        ###############################
        ## Clean

        # xent, pixel, 0.515
        # loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        # l2, pixel, 0.48
        # loss = tf.reduce_mean(tf.square(self.AE(self.x) - self.x))
        # l1, pixel, 0.52
        # loss = tf.reduce_mean(tf.abs(self.AE(self.x) - self.x))


        # L2, logits, .62
        loss = tf.reduce_mean(tf.square(self.cnn_model.predict(self.AE(self.x)) -
                                        self.cnn_model.predict(self.x)))
        # L1, logits, .30
        # loss = tf.reduce_mean(tf.abs(self.cnn_model.predict(self.AE(self.x)) -
        #                              self.cnn_model.predict(self.x)))
        # L2, feature, 0.39
        # loss = tf.reduce_mean(tf.square(self.cnn_model.CNN(self.AE(self.x)) -
        #                                 self.cnn_model.CNN(self.x)))
        # L1, feature, .445
        # loss = tf.reduce_mean(tf.abs(self.cnn_model.CNN(self.AE(self.x)) -
        #                              self.cnn_model.CNN(self.x)))

        # xent, logits, .30
        # loss = my_sigmoid_xent(self.cnn_model.predict(self.AE(self.x)),
        #                        tf.nn.sigmoid(self.cnn_model.predict(self.x)))
        # xent, label, .38
        # loss = my_softmax_xent(self.cnn_model.predict(self.AE(self.x)),
        #                        self.y)
        
        self.AE_loss = loss
        
        self.noisy_accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.AE(noisy_x)),
                                                  self.y)

        self.accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.AE(self.x)),
                                            self.y)

        self.clean_accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.x),
                                                  self.y)

        self.setup_trainstep()
    def setup_trainstep(self):
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        # These numbers are batches
        schedules = [[0, 1e-3], [400*20, 1e-4], [400*40, 1e-5], [400*60, 1e-6]]
        boundaries = [s[0] for s in schedules][1:]
        values = [s[1] for s in schedules]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values)
        self.AE_train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(
                self.AE_loss,
                global_step=global_step,
                var_list=self.AE_vars)
        # self.AE_train_step = tf.train.AdamOptimizer(0.001).minimize(
        #     self.AE_loss, var_list=self.AE_vars)
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
        
    def train_AE(self, sess, train_x, train_y):
        """Technically I don't need clean y, but I'm going to monitor the noisy accuracy."""
        my_training(sess, self.x, self.y,
                    self.AE_loss, self.AE_train_step,
                    [self.noisy_accuracy, self.accuracy, self.clean_accuracy, self.global_step],
                    ['noisy_acc', 'accuracy', 'clean_accuracy', 'global step'],
                    train_x, train_y,
                    # Seems that the denoiser converge fast
                    patience=5,
                    print_interval=50,
                    do_plot=False)

    def train_AE_keras(self, sess, train_x, train_y):
        noise_factor = 0.1
        noisy_x = train_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)

        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)
        
        optimizer = keras.optimizers.RMSprop(0.001)
        # 'adadelta'
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      # loss=keras.losses.mean_squared_error
                      # metrics=[self.noisy_accuracy]
        )
        with sess.as_default():
            model.fit(noisy_x, train_x, epochs=100, validation_split=0.1, verbose=2)
    def test_AE(self, sess, test_x, test_y):
        """Test AE models."""
        # select 10
        images = []
        titles = []
        fringes = []

        test_x = test_x[:100]
        test_y = test_y[:100]

        to_run = {}
            
        # Test AE rec output before and after
        rec = self.AE(self.x)
        noisy_x = my_add_noise(self.x)
        noisy_rec = self.AE(noisy_x)

        to_run['x'] = self.x
        to_run['y'] = tf.argmax(self.y, 1)
        to_run['rec'] = rec
        to_run['noisy_x'] = noisy_x
        to_run['noisy_rec'] = noisy_rec
        print('Testing CNN and AE ..')
        res = sess.run(to_run, feed_dict={self.x: test_x, self.y: test_y})

        
        images.append(res['x'][:10])
        titles.append(res['y'][:10])
        fringes.append('x')
        
        for name in ['rec', 'noisy_x', 'noisy_rec']:
            images.append(res[name][:10])
            titles.append(['']*10)
            fringes.append('{}'.format(name))

        print('Plotting result ..')
        grid_show_image(images, filename='out.pdf', titles=titles, fringes=fringes)
        print('Done. Saved to {}'.format('out.pdf'))
        
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
    def name():
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
        
        decoded = keras.layers.Add()([inputs, conv10])
        # decoded = keras.layers.Subtract()([inputs, conv10])
        # DEBUG what if I just use this as output?
        # decoded = conv10
        self.AE1 = keras.models.Model(inputs, decoded)
        
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))


class AdvAEModel(cleverhans.model.Model):
    """Implement a denoising auto encoder.
    """
    def __init__(self, cnn_model, ae_model):
        # this CNN model is used as victim for attacks
        self.cnn_model = cnn_model
        self.CNN = cnn_model.CNN
        self.FC = cnn_model.FC

        self.ae_model = ae_model
        self.AE = ae_model.AE
        self.AE1 = ae_model.AE1
        
        self.x = keras.layers.Input(shape=self.cnn_model.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.cnn_model.yshape(), dtype='float32')

        # self.setup_attack_params()
        self.setup_loss()
        # self.setup_loss_simple()
        
        self.setup_trainloss()
        self.adv_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.adv_loss, var_list=self.ae_model.AE_vars)

    @staticmethod
    def NAME():
        "Override by children models."
        return "AdvAE"
    def name(self):
        return '{}-{}-{}'.format(self.cnn_model.NAME(),
                                 self.ae_model.NAME(),
                                 # FIXME ugly and why??
                                 type(self).NAME())
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(self.AE(x)))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def setup_loss_simple(self):
        print('Seting up loss ..')
        high = self.CNN(self.x)
        # logits = self.FC(high)
        
        rec = self.AE(self.x)
        rec_high = self.CNN(rec)
        rec_logits = self.FC(rec_high)

        self.C0_loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        self.C1_loss = my_sigmoid_xent(rec_high, tf.nn.sigmoid(high))
        self.C2_loss = my_softmax_xent(rec_logits, self.y)

        noisy_x = my_add_noise(self.x)
        noisy_rec = self.AE(noisy_x)
        noisy_rec_high = self.CNN(noisy_rec)
        noisy_rec_logits = self.FC(noisy_rec_high)

        self.N0_loss = my_sigmoid_xent(self.AE1(noisy_x), self.x)
        self.N0_l2_loss = tf.reduce_mean(tf.pow(self.AE(noisy_x) - self.x, 2))
        self.N1_loss = my_sigmoid_xent(noisy_rec_high, tf.nn.sigmoid(high))
        self.N2_loss = my_softmax_xent(noisy_rec_logits, self.y)

        adv_x = my_PGD(self, self.x, params=self.cnn_model.PGD_params)
        adv_high = self.CNN(adv_x)
        adv_logits = self.FC(adv_high)

        adv_rec = self.AE(adv_x)
        adv_rec_high = self.CNN(adv_rec)
        adv_rec_logits = self.FC(adv_rec_high)
        
        self.A0_loss = my_sigmoid_xent(self.AE1(adv_x), self.x)
        self.A1_loss = my_sigmoid_xent(adv_rec_high, tf.nn.sigmoid(high))
        self.A2_loss = my_softmax_xent(adv_rec_logits, self.y)

        print('Setting up prediction ..')
        # no AE, just CNN
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_accuracy = my_accuracy_wrapper(CNN_logits, self.y)
        # Should work on rec_logits
        logits = self.FC(self.CNN(self.AE(self.x)))
        self.accuracy = my_accuracy_wrapper(logits, self.y)
        
        noisy_x = my_add_noise(self.x)
        noisy_logits = self.FC(self.CNN(self.AE(noisy_x)))
        self.noisy_accuracy = my_accuracy_wrapper(noisy_logits, self.y)

        print('Setting up metrics ..')
        # TODO monitoring each of these losses and adjust weights
        self.metrics = [
            # clean data
            self.C0_loss, self.C1_loss, self.C2_loss, self.accuracy,
            # noisy data
            self.N0_loss, self.N1_loss, self.N2_loss, self.noisy_accuracy,
            # adv data
            # self.A0_loss, self.A1_loss, self.A2_loss, self.adv_accuracy,
            # postadv
            # self.P0_loss, self.P1_loss, self.P2_loss, self.postadv_accuracy,
            # I also need a baseline for the CNN performance probably?
            # self.CNN_accuracy, self.B1_loss, self.B2_loss, self.obli_accuracy
        ]
        
        self.metric_names = [
            'C0_loss', 'C1_loss', 'C2_loss', 'accuracy',
            'N0_loss', 'N1_loss', 'N2_loss', 'noisy_accuracy',
            # 'A0_loss', 'A1_loss', 'A2_loss', 'adv_accuracy',
            # 'P0_loss', 'P1_loss', 'P2_loss', 'postadv_accuracy',
            # 'cnn_accuracy', 'B1_loss', 'B2_loss', 'obli_accuracy'
        ]
        
    def setup_loss(self):
        print('Seting up loss ..')
        high = self.CNN(self.x)
        # logits = self.FC(high)
        
        rec = self.AE(self.x)
        rec_high = self.CNN(rec)
        rec_logits = self.FC(rec_high)

        self.C0_loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        self.C1_loss = my_sigmoid_xent(rec_high, tf.nn.sigmoid(high))
        self.C2_loss = my_softmax_xent(rec_logits, self.y)

        noisy_x = my_add_noise(self.x)
        noisy_rec = self.AE(noisy_x)
        noisy_rec_high = self.CNN(noisy_rec)
        noisy_rec_logits = self.FC(noisy_rec_high)

        self.N0_loss = my_sigmoid_xent(self.AE1(noisy_x), self.x)
        self.N0_l2_loss = tf.reduce_mean(tf.pow(self.AE(noisy_x) - self.x, 2))
        self.N1_loss = my_sigmoid_xent(noisy_rec_high, tf.nn.sigmoid(high))
        self.N2_loss = my_softmax_xent(noisy_rec_logits, self.y)

        adv_x = my_PGD(self, self.x, params=self.cnn_model.PGD_params)
        adv_high = self.CNN(adv_x)
        adv_logits = self.FC(adv_high)

        adv_rec = self.AE(adv_x)
        adv_rec_high = self.CNN(adv_rec)
        adv_rec_logits = self.FC(adv_rec_high)
        
        self.A0_loss = my_sigmoid_xent(self.AE1(adv_x), self.x)
        self.A1_loss = my_sigmoid_xent(adv_rec_high, tf.nn.sigmoid(high))
        self.A2_loss = my_softmax_xent(adv_rec_logits, self.y)

        postadv = my_PGD(self.cnn_model, rec, params=self.cnn_model.PGD_params)
        postadv_high = self.CNN(postadv)
        postadv_logits = self.FC(postadv_high)

        self.P0_loss = my_sigmoid_xent(postadv, tf.nn.sigmoid(self.x))
        self.P1_loss = my_sigmoid_xent(postadv_high, tf.nn.sigmoid(high))
        self.P2_loss = my_softmax_xent(postadv_logits, self.y)

        obliadv = my_PGD(self.cnn_model, self.x, params=self.cnn_model.PGD_params)
        obliadv_high = self.CNN(self.AE(obliadv))
        obliadv_logits = self.FC(obliadv_high)

        self.B1_loss = my_sigmoid_xent(obliadv_high, tf.nn.sigmoid(high))
        self.B2_loss = my_softmax_xent(obliadv_logits, self.y)
        
        print('Setting up prediction ..')
        # no AE, just CNN
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_accuracy = my_accuracy_wrapper(CNN_logits, self.y)
        # Should work on rec_logits
        logits = self.FC(self.CNN(self.AE(self.x)))
        self.accuracy = my_accuracy_wrapper(logits, self.y)
        
        adv_logits = self.FC(self.CNN(self.AE(my_PGD(self, self.x, params=self.cnn_model.PGD_params))))
        self.adv_accuracy = my_accuracy_wrapper(adv_logits, self.y)

        postadv_logits = self.FC(self.CNN(my_PGD(self.cnn_model, self.AE(self.x), params=self.cnn_model.PGD_params)))
        self.postadv_accuracy = my_accuracy_wrapper(postadv_logits, self.y)
        
        obli_logits = self.FC(self.CNN(self.AE(my_PGD(self.cnn_model, self.x, params=self.cnn_model.PGD_params))))
        self.obli_accuracy = my_accuracy_wrapper(obli_logits, self.y)

        noisy_x = my_add_noise(self.x)
        noisy_logits = self.FC(self.CNN(self.AE(noisy_x)))
        self.noisy_accuracy = my_accuracy_wrapper(noisy_logits, self.y)
        
        print('Setting up metrics ..')
        # TODO monitoring each of these losses and adjust weights
        self.metrics = [
            # clean data
            self.C0_loss, self.C1_loss, self.C2_loss, self.accuracy,
            # noisy data
            self.N0_loss, self.N1_loss, self.N2_loss, self.noisy_accuracy,
            # adv data
            self.A0_loss, self.A1_loss, self.A2_loss, self.adv_accuracy,
            # postadv
            self.P0_loss, self.P1_loss, self.P2_loss, self.postadv_accuracy,
            # I also need a baseline for the CNN performance probably?
            self.CNN_accuracy, self.B1_loss, self.B2_loss, self.obli_accuracy
        ]
        
        self.metric_names = [
            'C0_loss', 'C1_loss', 'C2_loss', 'accuracy',
            'N0_loss', 'N1_loss', 'N2_loss', 'noisy_accuracy',
            'A0_loss', 'A1_loss', 'A2_loss', 'adv_accuracy',
            'P0_loss', 'P1_loss', 'P2_loss', 'postadv_accuracy',
            'cnn_accuracy', 'B1_loss', 'B2_loss', 'obli_accuracy']


    def train_Adv(self, sess, train_x, train_y):
        """Adv training."""
        my_training(sess, self.x, self.y,
                    self.adv_loss,
                    self.adv_train_step,
                    self.metrics, self.metric_names,
                    train_x, train_y,
                    plot_prefix=self.name(),
                    num_epochs=10,
                    patience=3)



    def test_attack(self, sess, test_x, test_y, name):
        """Reurn images and titles."""
        # TODO mark the correct and incorrect predictions
        # to_run += [adv_x, adv_rec, postadv]
        # TODO add prediction result
        # titles += ['{} adv_x'.format(name), 'adv_rec', 'postadv']
        
            
        # TODO select 5 correct and incorrect examples
        # indices = random.sample(range(test_x.shape[0]), 10)
        # test_x = test_x[indices]
        # test_y = test_y[indices]
        # feed_dict = {self.x: test_x, self.y: test_y}
        images = []
        titles = []
        fringes = []
        # JSMA is pretty slow ..
        # attack_funcs = [my_FGSM, my_PGD, my_JSMA]
        # attack_names = ['FGSM', 'PGD', 'JSMA']
        if name is 'FGSM':
            attack = lambda m, x: my_FGSM(m, x, params=self.cnn_model.FGSM_params)
        elif name is 'PGD':
            attack = lambda m, x: my_PGD(m, x, params=self.cnn_model.PGD_params)
        elif name is 'JSMA':
            attack = lambda m, x: my_JSMA(m, x, params=self.cnn_model.JSMA_params)
        elif name is 'CW':
            attack = lambda m, x: my_CW(m, sess, x, self.y, params=self.cnn_model.CW_params)
        else:
            assert False


        # This is tricky. I want to get the baseline of attacking
        # clean image AND VANILLA CNN. Thus, I'm using the proxy
        # cnn model.
        obliadv = attack(self.cnn_model, self.x)
        obliadv_rec = self.AE(obliadv)
        
        baseline_logits = self.FC(self.CNN(obliadv))
        baseline_acc = my_accuracy_wrapper(baseline_logits, self.y)
        obliadv_logits = self.FC(self.CNN(self.AE(obliadv)))
        obliadv_acc = my_accuracy_wrapper(obliadv_logits, self.y)
        
        whiteadv = attack(self, self.x)
        whiteadv_rec = self.AE(whiteadv)
        whiteadv_logits = self.FC(self.CNN(self.AE(whiteadv)))
        whiteadv_acc = my_accuracy_wrapper(whiteadv_logits, self.y)

        # remember to compare visually postadv with rec (i.e. AE(x))
        postadv = attack(self.cnn_model, self.AE(self.x))
        postadv_logits = self.FC(self.CNN(postadv))
        postadv_acc = my_accuracy_wrapper(postadv_logits, self.y)

        
        to_run = {}

        to_run['obliadv'] = obliadv, tf.argmax(baseline_logits, axis=1), baseline_acc
        to_run['obliadv_rec'] = obliadv_rec, tf.argmax(obliadv_logits, axis=1), obliadv_acc
        to_run['whiteadv'] = whiteadv, tf.constant(0, shape=(10,)), tf.constant(-1)
        to_run['whiteadv_rec'] = whiteadv_rec, tf.argmax(whiteadv_logits, axis=1), whiteadv_acc
        to_run['postadv'] = postadv, tf.argmax(postadv_logits, axis=1), postadv_acc
        
        print('Testing attack {} ..'.format(name))
        t = time.time()
        res = sess.run(to_run, feed_dict={self.x: test_x, self.y: test_y})
        print('Attack done. Time: {:.3f}'.format(time.time()-t))

        for key in res:
            tmp_images, tmp_titles, acc = res[key]
            images.append(tmp_images[:10])
            if acc == -1:
                titles.append(['']*10)
                fringe = '{}\n{}'.format(name, key)
            else:
                titles.append(tmp_titles[:10])
                fringe = '{}\n{}\n{:.3f}'.format(name, key, acc)
            fringes.append(fringe)
            print(fringe.replace('\n', ' '))
        return images, titles, fringes

    def test_Model(self, sess, test_x, test_y):
        """Test CNN and AE models."""
        # select 10
        images = []
        titles = []
        fringes = []

        to_run = {}
            
        # Test AE rec output before and after
        rec = self.AE(self.x)
        noisy_x = my_add_noise(self.x)
        noisy_rec = self.AE(noisy_x)

        to_run['x'] = self.x
        to_run['y'] = tf.argmax(self.y, 1)
        to_run['rec'] = rec
        to_run['noisy_x'] = noisy_x
        to_run['noisy_rec'] = noisy_rec
        print('Testing CNN and AE ..')
        res = sess.run(to_run, feed_dict={self.x: test_x, self.y: test_y})

        accs = sess.run([self.CNN_accuracy, self.accuracy, self.noisy_accuracy],
                        feed_dict={self.x: test_x, self.y: test_y})
        print('raw accuracies (raw CNN, AE clean, AE noisy): {}'.format(accs))
        
        images.append(res['x'][:10])
        titles.append(res['y'][:10])
        fringes.append('x\n{:.3f}'.format(accs[0]))
        
        for name, acc in zip(['rec', 'noisy_x', 'noisy_rec'], [accs[1], accs[2], accs[2]]):
            images.append(res[name][:10])
            titles.append(['']*10)
            fringes.append('{}\n{:.3f}'.format(name, acc))

        return images, titles, fringes
        
        

    def test_all(self, sess, test_x, test_y, attacks=[], filename='out.pdf', num_sample=100, batch_size=100):
        """Test clean data x and y.

        - Use only CNN, to test whether CNN is functioning
        - Test AE, output image for before and after
        - Test AE + CNN, to test whether the entire system works

        CW is costly, so by default no running for it. If you want it,
        set it to true.

        """
        # FIXME when the num_sample is large, I need to batch them
        # random 100 images for testing
        indices = random.sample(range(test_x.shape[0]), num_sample)
        test_x = test_x[indices]
        test_y = test_y[indices]

        all_images = []
        all_titles = []
        all_fringes = []
        
        # nbatch = num_sample // batch_size
        # for i in range(nbatch):
        #     start = i * batch_size
        #     end = (i+1) * batch_size
        #     batch_x = test_x[indices[start:end]]
        #     batch_y = test_y[indices[start:end]]

        images, titles, fringes = self.test_Model(sess, test_x, test_y)
        all_images.extend(images)
        all_titles.extend(titles)
        all_fringes.extend(fringes)
        for name in attacks:
            images, titles, fringes = self.test_attack(sess, test_x, test_y, name)
            all_images.extend(images)
            all_titles.extend(titles)
            all_fringes.extend(fringes)
        
        print('Plotting result ..')
        grid_show_image(all_images, filename=filename, titles=all_titles, fringes=all_fringes)
        print('Done. Saved to {}'.format(filename))

    def setup_trainloss(self):
        """DEPRECATED Overwrite by subclasses to customize the loss.

        - 0: pixel
        - 1: high level conv
        - 2: logits

        The data:
        - C: Clean
        - N: Noisy
        - A: Adv
        - P: Post
        
        Available loss terms:

        # clean data
        self.C0_loss, self.C1_loss, self.C2_loss,
        
        # noisy data
        self.N0_loss, self.N1_loss, self.N2_loss,
        
        # adv data
        self.A0_loss, self.A1_loss, self.A2_loss,
        
        # postadv
        self.P0_loss, self.P1_loss, self.P2_loss,
        
        """
        self.adv_loss = (self.A2_loss)
    
# one
class Test_Model(AdvAEModel):
    """This is a dummy class, exactly same as AdvAEModel except name()."""
    def NAME():
        return 'Test'
    def setup_trainloss(self):
        # self.adv_loss = self.C0_loss
        self.adv_loss = self.A2_loss
        
class A2_Model(AdvAEModel):
    """This is a dummy class, exactly same as AdvAEModel except name()."""
    def NAME():
        return 'A2'
    def setup_trainloss(self):
        self.adv_loss = self.A2_loss
class B2_Model(AdvAEModel):
    """B2/B1 loss is exactly the oblivious loss, corresponding to HGD."""
    def setup_trainloss(self):
        self.adv_loss = self.B2_loss
    def NAME():
        return 'B2'
class A1_Model(AdvAEModel):
    def NAME():
        return 'A1'
    def setup_trainloss(self):
        self.adv_loss = (self.A1_loss)
class A12_Model(AdvAEModel):
    def NAME():
        return 'A12'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.A1_loss)
# two
class N0_A2_Model(AdvAEModel):
    def NAME():
        return 'N0_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.N0_loss)
class C0_A2_Model(AdvAEModel):
    def NAME():
        return 'C0_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.C0_loss)
class C2_A2_Model(AdvAEModel):
    def NAME():
        return 'C2_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.C2_loss)

# three
class C0_N0_A2_Model(AdvAEModel):
    def NAME():
        return 'C0_N0_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.N0_loss + self.C0_loss)
class C2_N0_A2_Model(AdvAEModel):
    def NAME():
        return 'C2_N0_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.N0_loss + self.C2_loss)

# P models
class P2_Model(AdvAEModel):
    def NAME():
        return 'P2'
    def setup_trainloss(self):
        self.adv_loss = (self.P2_loss)
class A2_P2_Model(AdvAEModel):
    def NAME():
        return 'A2_P2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P2_loss)
class N0_A2_P2_Model(AdvAEModel):
    def NAME():
        return 'N0_A2_P2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P2_loss + self.N0_loss)
        
class C0_N0_A2_P2_Model(AdvAEModel):
    def NAME():
        return 'C0_N0_A2_P2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.N0_loss + self.C0_loss)
class C2_A2_P2_Model(AdvAEModel):
    def NAME():
        return 'C2_A2_P2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.C2_loss)
class C2_N0_A2_P2_Model(AdvAEModel):
    def NAME():
        return 'C2_N0_A2_P2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.N0_loss + self.C2_loss)
class A2_P1_Model(AdvAEModel):
    def NAME():
        return 'A2_P1'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.P1_loss)
        
class AdvAEModel_Var0(AdvAEModel):
    "This is the same as AdvAEModel. I'm not using it, just use as a reference."
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # FIXME this also assigns self.conv_layers
        x = keras.layers.Reshape([28, 28, 1])(inputs)
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
class AdvAEModel_Var1(AdvAEModel):
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        x = keras.layers.Reshape([28, 28, 1])(inputs)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        # x = keras.layers.Conv2D(32, 3)(x)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        # x = keras.layers.Conv2D(64, 3)(x)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)
        self.CNN = keras.models.Model(inputs, x)
class AdvAEModel_Var2(AdvAEModel):
    def setup_CNN(self):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        x = keras.layers.Reshape([28, 28, 1])(inputs)
        x = keras.layers.Conv2D(32, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Conv2D(64, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, 5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)
        self.CNN = keras.models.Model(inputs, x)

class FCAdvAEModel(AdvAEModel):
    """Instead of an auto encoder, uses just an FC.

    FIXME This does not work.
    """
    def setup_AE(self):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        """From noise_x to x."""
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dense(28*28)(x)
        decoded = keras.layers.Reshape([28,28,1])(x)
        self.AE = keras.models.Model(inputs, decoded)

    
def my_distillation():
    """1. train a normal model, save the weights
    
    2. teacher model: load the weights, train the model with ce loss
    function but replace: logits=predicted/train_temp(10)
    
    3. predicted = teacher.predict(data.train_data)
       y = sess.run(tf.nn.softmax(predicted/train_temp))
       data.train_labels = y
    
    4. Use the updated data to train the student. The student should
    load the original weights. Still use the loss function with temp.

    https://github.com/carlini/nn_robust_attacks

    """
    pass

   

def __test():
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file('saved_model/AE.ckpt', tensor_name='', all_tensors=True)

def __test():
    train_double_backprop("saved_model/double-backprop.ckpt")
    train_group_lasso("saved_model/group-lasso.ckpt")
    
    sess, model = restore_model("saved_model/double-backprop.ckpt")
    sess, model = restore_model("saved_model/group-lasso.ckpt")
    sess, model = restore_model("saved_model/ce.ckpt")
    sess, model = restore_model("saved_model/adv.ckpt")
    # sess, model = restore_model("saved_model/adv2.ckpt")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    # testing model accuracy on clean data
    model.compute_accuracy(sess, model.x, model.y, model.logits, test_x, test_y)
    
    # 10 targeted inputs for demonstration
    inputs, labels, targets = generate_victim(test_x, test_y)
    # all 10000 inputs
    inputs, labels = test_x, test_y
    # random 100 inputs
    indices = random.sample(range(test_x.shape[0]), 100)
    inputs = test_x[indices]
    labels = test_y[indices]
    
    test_model_against_attack(sess, model, my_FGSM, inputs, labels, None, prefix='FGSM')
    test_model_against_attack(sess, model, my_PGD, inputs, labels, None, prefix='PGD')
    test_model_against_attack(sess, model, my_JSMA, inputs, labels, None, prefix='JSMA')
    # test_model_against_attack(sess, model, my_CW_targeted, inputs, labels, targets, prefix='CW_targeted')
    test_model_against_attack(sess, model, my_CW, inputs, labels, None, prefix='CW')

class TestA():
    def __init__(self, x):
        self.a = 1
        self.x = x
        self.foo()
        self.bar(x)
    def foo(self):
        self.b = 2
    def bar(self, xx):
        self.xx = xx
class TestB(TestA):
    def foo(self):
        self.b = 3
class TestC(TestA):
    def __init__(self, x):
        super().__init__(x)
    def foo(self):
        self.b = 4

def __test():
    TestA(1).b
    TestB(2).a
    TestB(2).b
    TestB(8).x
    TestB(8).xx
    TestC(4).a
    TestC(5).b
    TestC(3).x
