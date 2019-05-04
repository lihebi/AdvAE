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

class CNNModel(cleverhans.model.Model):
    """This model is used inside AdvAEModel."""
    def __init__(self, CNN, FC):
        self.CNN = CNN
        self.FC = FC
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(x))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}


class AdvAEModel(cleverhans.model.Model):
    """Implement a denoising auto encoder.
    """
    def __init__(self):
        # FIXME I need to make sure these setups are only called
        # once. Otherwise I need to set the AE_vars variables accordingly
        with tf.variable_scope('my_AE'):
            self.setup_AE()
        with tf.variable_scope('my_CNN'):
            self.setup_CNN()
        with tf.variable_scope('my_FC'):
            self.setup_FC()

        self.AE_vars = tf.trainable_variables('my_AE')
        self.CNN_vars = tf.trainable_variables('my_CNN')
        self.FC_vars = tf.trainable_variables('my_FC')

        # this CNN model is used as victim for attacks
        self.cnn_model = CNNModel(self.CNN, self.FC)
        
        self.x = keras.layers.Input(shape=self.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.yshape(), dtype='float32')

        self.setup_attack_params()
        self.setup_loss()
        self.setup_prediction()
        self.setup_metrics()
        self.setup_trainloss()
        self.setup_trainstep()
        

    @staticmethod
    def name():
        "Override by children models."
        return "AdvAE"
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(self.AE(x)))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def xshape(self):
        return (28,28,1)
    def yshape(self):
        return (10,)

    @staticmethod
    def add_noise(x):
        noise_factor = 0.5
        noisy_x = x + noise_factor * tf.random.normal(shape=tf.shape(x), mean=0., stddev=1.)
        noisy_x = tf.clip_by_value(noisy_x, CLIP_MIN, CLIP_MAX)
        return noisy_x
    @staticmethod
    def sigmoid_xent(logits=None, labels=None):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels))
    @staticmethod
    def softmax_xent(logits=None, labels=None):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels))
    @staticmethod
    def accuracy_wrapper(logits=None, labels=None):
        return tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits, axis=1),
            tf.argmax(labels, 1)), dtype=tf.float32))
    @staticmethod
    def compute_accuracy(sess, x, y, preds, x_test, y_test):
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
            dtype=tf.float32))
        return sess.run(acc, feed_dict={x: x_test, y: y_test})

    def setup_loss(self):
        print('Seting up loss ..')
        high = self.CNN(self.x)
        # logits = self.FC(high)
        
        rec = self.AE(self.x)
        rec_high = self.CNN(rec)
        rec_logits = self.FC(rec_high)

        self.C0_loss = self.sigmoid_xent(self.AE1(self.x), self.x)
        self.C1_loss = self.sigmoid_xent(rec_high, tf.nn.sigmoid(high))
        self.C2_loss = self.softmax_xent(rec_logits, self.y)

        noisy_x = self.add_noise(self.x)
        noisy_rec = self.AE(noisy_x)
        noisy_rec_high = self.CNN(noisy_rec)
        noisy_rec_logits = self.FC(noisy_rec_high)

        self.N0_loss = self.sigmoid_xent(self.AE1(noisy_x), self.x)
        self.N0_l2_loss = tf.reduce_mean(tf.pow(self.AE(noisy_x) - self.x, 2))
        self.N1_loss = self.sigmoid_xent(noisy_rec_high, tf.nn.sigmoid(high))
        self.N2_loss = self.softmax_xent(noisy_rec_logits, self.y)

        adv_x = my_PGD(self, self.x)
        adv_high = self.CNN(adv_x)
        adv_logits = self.FC(adv_high)

        adv_rec = self.AE(adv_x)
        adv_rec_high = self.CNN(adv_rec)
        adv_rec_logits = self.FC(adv_rec_high)
        
        self.A0_loss = self.sigmoid_xent(self.AE1(adv_x), self.x)
        self.A1_loss = self.sigmoid_xent(adv_rec_high, tf.nn.sigmoid(high))
        self.A2_loss = self.softmax_xent(adv_rec_logits, self.y)

        # pre-denoiser: train denoiser such that the denoised x is
        # itself robust
        
        postadv = my_PGD(self.cnn_model, rec)
        postadv_high = self.CNN(postadv)
        postadv_logits = self.FC(postadv_high)

        self.P0_loss = self.sigmoid_xent(postadv, tf.nn.sigmoid(self.x))
        self.P1_loss = self.sigmoid_xent(postadv_high, tf.nn.sigmoid(high))
        self.P2_loss = self.softmax_xent(postadv_logits, self.y)
        
    def setup_prediction(self):
        print('Setting up prediction ..')
        # no AE, just CNN
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_accuracy = self.accuracy_wrapper(CNN_logits, self.y)
        # Should work on rec_logits
        logits = self.FC(self.CNN(self.AE(self.x)))
        self.accuracy = self.accuracy_wrapper(logits, self.y)
        
        adv_logits = self.FC(self.CNN(self.AE(my_PGD(self, self.x))))
        self.adv_accuracy = self.accuracy_wrapper(adv_logits, self.y)

        postadv_logits = self.FC(self.CNN(my_PGD(self.cnn_model, self.AE(self.x))))
        self.postadv_accuracy = self.accuracy_wrapper(postadv_logits, self.y)

        noisy_x = self.add_noise(self.x)
        noisy_logits = self.FC(self.CNN(self.AE(noisy_x)))
        self.noisy_accuracy = self.accuracy_wrapper(noisy_logits, self.y)
    def setup_metrics(self):
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
            self.P0_loss, self.P1_loss, self.P2_loss, self.postadv_accuracy
            # I also need a baseline for the CNN performance probably?
        ]
        
        self.metric_names = [
            'C0_loss', 'C1_loss', 'C2_loss', 'accuracy',
            'N0_loss', 'N1_loss', 'N2_loss', 'noisy_accuracy',
            'A0_loss', 'A1_loss', 'A2_loss', 'adv_accuracy',
            'P0_loss', 'P1_loss', 'P2_loss', 'postadv_accuracy']

    # def setup_AE(self):
    #     """From noise_x to x."""
    #     inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
    #     x = keras.layers.Flatten()(inputs)
    #     x = keras.layers.Dense(100, activation='relu')(x)
    #     x = keras.layers.Dense(20, activation='relu')(x)
    #     x = keras.layers.Dense(100, activation='relu')(x)
    #     x = keras.layers.Dense(28*28, activation='sigmoid')(x)
    #     decoded = keras.layers.Reshape([28,28,1])(x)
    #     self.AE = keras.models.Model(inputs, decoded)
        
    def setup_AE(self):
        """From noise_x to x."""
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        # inputs = keras.layers.Input(shape=(28,28,1), dtype='float32')
        # denoising
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        # at this point the representation is (7, 7, 32)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        self.AE1 = keras.models.Model(inputs, decoded)
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))

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
        
    def save_CNN(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.CNN_vars + self.FC_vars)
        saver.save(sess, path)

    def save_AE(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.save(sess, path)

    def load_CNN(self, sess, path):
        saver = tf.train.Saver(var_list=self.CNN_vars + self.FC_vars)
        saver.restore(sess, path)

    def load_AE(self, sess, path):
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.restore(sess, path)

    def train_CNN(self, sess, train_x, train_y):
        my_training(sess, self.x, self.y,
                    self.CNN_loss,
                    self.CNN_train_step,
                    [self.CNN_accuracy], ['accuracy'],
                    train_x, train_y,
                    num_epochs=200,
                    patience=10)
    def train_Adv(self, sess, train_x, train_y):
        """Adv training."""
        my_training(sess, self.x, self.y,
                    self.adv_loss,
                    self.adv_train_step,
                    self.metrics, model.metric_names,
                    train_x, train_y,
                    plot_prefix=advprefix,
                    num_epochs=10,
                    patience=2)

    def train_AE(self, sess, train_x, train_y):
        """Technically I don't need clean y, but I'm going to monitor the noisy accuracy."""
        my_training(sess, self.x, self.y,
                    self.N0_loss, self.AE_train_step,
                    [self.noisy_accuracy], ['noisy_acc'],
                    train_x, train_y,
                    patience=5)

    def setup_attack_params(self):
        self.FGSM_params = {'eps': 0.3}
        self.PGD_params = {'eps': 0.3,
                           'nb_iter': 40,
                           'eps_iter': 0.01}
        self.JSMA_params = {}
        self.CW_params = {'max_iterations': 1000,
                          'learning_rate': 0.2}

    def test_attack(self, sess, test_x, test_y, name):
        """Reurn images and titles."""
        images = []
        titles = []
        # JSMA is pretty slow ..
        # attack_funcs = [my_FGSM, my_PGD, my_JSMA]
        # attack_names = ['FGSM', 'PGD', 'JSMA']
        if name is 'FGSM':
            attack = lambda m, x: my_FGSM(m, x, self.FGSM_params)
        elif name is 'PGD':
            attack = lambda m, x: my_PGD(m, x, self.PGD_params)
        elif name is 'JSMA':
            attack = lambda m, x: my_JSMA(m, x, self.JSMA_params)
        elif name is 'CW':
            attack = lambda m, x: my_CW(m, sess, x, self.y, self.CW_params)
        else:
            assert False

        adv_x = attack(self, self.x)
        adv_rec = self.AE(adv_x)

        # This is tricky. I want to get the baseline of attacking
        # clean image AND VANILLA CNN. Thus, I'm using the proxy
        # cnn model.
        adv_x_CNN = attack(self.cnn_model, self.x)
        logits = self.FC(self.CNN(adv_x_CNN))
        accuracy = self.accuracy_wrapper(logits, self.y)

        adv_logits = self.FC(self.CNN(self.AE(adv_x)))
        adv_accuracy = self.accuracy_wrapper(adv_logits, self.y)

        # remember to compare visually postadv with rec (i.e. AE(x))
        postadv = attack(self.cnn_model, self.AE(self.x))
        postadv_logits = self.FC(self.CNN(postadv))
        postadv_accuracy = self.accuracy_wrapper(postadv_logits, self.y)

        to_run = {}
        to_run['adv_x'] = adv_x

        to_run['adv_x_CNN'] = adv_x_CNN
        to_run['pred'] = tf.argmax(logits, axis=1)
        to_run['accuracy'] = accuracy

        to_run['adv_rec'] = adv_rec
        to_run['adv_pred'] = tf.argmax(adv_logits, axis=1)
        to_run['adv_acc'] = adv_accuracy

        to_run['postadv'] = postadv
        to_run['postadv_pred'] = tf.argmax(postadv_logits, axis=1)
        to_run['postadv_acc'] = postadv_accuracy

        # TODO mark the correct and incorrect predictions
        # to_run += [adv_x, adv_rec, postadv]
        # TODO add prediction result
        # titles += ['{} adv_x'.format(name), 'adv_rec', 'postadv']
        
            
        # TODO select 5 correct and incorrect examples
        # indices = random.sample(range(test_x.shape[0]), 10)
        # test_x = test_x[indices]
        # test_y = test_y[indices]
        # feed_dict = {self.x: test_x, self.y: test_y}
        
        print('Testing attack {} ..'.format(name))
        t = time.time()
        res = sess.run(to_run, feed_dict={self.x: test_x, self.y: test_y})
        print('Attack done. Time: {:.3f}'.format(time.time()-t))

        print('acc: {}, adv_acc: {}, postadv_acc: {}'
              .format(res['accuracy'], res['adv_acc'], res['postadv_acc']))
        
        images += list(res['adv_x_CNN'][:10])
        tmp = list(res['pred'][:10])
        tmp[0] = '{} adv_x_CNN\n({:.3f}): {}'.format(name, res['accuracy'], tmp[0])
        titles += tmp

        images += list(res['adv_x'][:10])
        titles += ['adv_x'] + ['']*9

        images += list(res['adv_rec'][:10])
        tmp = list(res['adv_pred'][:10])
        tmp[0] = '{} adv_rec\n({:.3f}): {}'.format(name, res['adv_acc'], tmp[0])
        titles += tmp

        images += list(res['postadv'][:10])
        tmp = list(res['postadv_pred'][:10])
        tmp[0] = 'postadv\n({:.3f}): {}'.format(res['postadv_acc'], tmp[0])
        titles += tmp
        return images, titles

    def test_Model(self, sess, test_x, test_y):
        """Test CNN and AE models."""
        # select 10
        images = []
        titles = []

        to_run = {}
            
        # Test AE rec output before and after
        rec = self.AE(self.x)
        noisy_x = self.add_noise(self.x)
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
        
        images += list(res['x'][:10])
        tmp = list(res['y'][:10])
        tmp[0] = 'x ({:.3f}): {}'.format(accs[0], tmp[0])
        titles += tmp
        
        for name, acc in zip(['rec', 'noisy_x', 'noisy_rec'], [accs[1], accs[2], accs[2]]):
            images += list(res[name][:10])
            titles += ['{} ({:.3f})'.format(name, acc)] + ['']*9

        return images, titles
        
        

    def test_all(self, sess, test_x, test_y, attacks=[], filename='out.pdf'):
        """Test clean data x and y.

        - Use only CNN, to test whether CNN is functioning
        - Test AE, output image for before and after
        - Test AE + CNN, to test whether the entire system works

        CW is costly, so by default no running for it. If you want it,
        set it to true.

        """
        # random 100 images for testing
        indices = random.sample(range(test_x.shape[0]), 100)
        test_x = test_x[indices]
        test_y = test_y[indices]

        all_images = []
        all_titles = []

        images, titles = self.test_Model(sess, test_x, test_y)
        all_images.extend(images)
        all_titles.extend(titles)
        for name in attacks:
            images, titles = self.test_attack(sess, test_x, test_y, name)
            all_images.extend(images)
            all_titles.extend(titles)
        
        print('Plotting result ..')
        grid_show_image(all_images, 10, int(len(all_images)/10), filename=filename, titles=all_titles)
        print('Done. Saved to {}'.format(filename))

    def setup_trainstep(self):
        print('Setting up train step ..')
        self.AE_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.AE_loss, var_list=self.AE_vars)
        self.adv_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.adv_loss, var_list=self.AE_vars)
        self.CNN_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.CNN_loss, var_list=self.CNN_vars+self.FC_vars)
        
    def setup_trainloss(self):
        """Overwrite by subclasses to customize the loss.

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
        print('Setting up train loss ..')
        self.adv_loss = (self.A2_loss)
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_loss = self.softmax_xent(CNN_logits, self.y)
        self.AE_loss = self.N0_loss + self.N0_l2_loss
# one
class A2_Model(AdvAEModel):
    """This is a dummy class, exactly same as AdvAEModel except name()."""
    def name():
        return 'A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = self.A2_loss
class A1_Model(AdvAEModel):
    def name():
        return 'A1'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A1_loss)
class A12_Model(AdvAEModel):
    def name():
        return 'A12'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.A1_loss)
# two
class N0_A2_Model(AdvAEModel):
    def name():
        return 'N0_A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.N0_loss)
class C0_A2_Model(AdvAEModel):
    def name():
        return 'C0_A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.C0_loss)
class C2_A2_Model(AdvAEModel):
    def name():
        return 'C2_A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.C2_loss)

# three
class C0_N0_A2_Model(AdvAEModel):
    def name():
        return 'C0_N0_A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.N0_loss + self.C0_loss)
class C2_N0_A2_Model(AdvAEModel):
    def name():
        return 'C2_N0_A2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.N0_loss + self.C2_loss)

# P models
class P2_Model(AdvAEModel):
    def name():
        return 'P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.P2_loss)
class A2_P2_Model(AdvAEModel):
    def name():
        return 'A2_P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.P2_loss)
class N0_A2_P2_Model(AdvAEModel):
    def name():
        return 'N0_A2_P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.P2_loss + self.N0_loss)
        
class C0_N0_A2_P2_Model(AdvAEModel):
    def name():
        return 'C0_N0_A2_P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.N0_loss + self.C0_loss)
class C2_A2_P2_Model(AdvAEModel):
    def name():
        return 'C2_A2_P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.C2_loss)
class C2_N0_A2_P2_Model(AdvAEModel):
    def name():
        return 'C2_N0_A2_P2'
    def setup_trainloss(self):
        super().setup_trainloss()
        self.adv_loss = (self.A2_loss + self.P2_loss +
                                 self.N0_loss + self.C2_loss)
class A2_P1_Model(AdvAEModel):
    def name():
        return 'A2_P1'
    def setup_trainloss(self):
        super().setup_trainloss()
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

# TODO use data augmentation during training
# TODO add lr schedule
class MyResNet(AdvAEModel):
    def setup_CNN(self):
        n = 3
        depth = n * 9 + 2
        print('Creating resnet graph ..')
        model = resnet_v2(input_shape=self.xshape(), depth=depth)
        print('Resnet graph created.')
        self.CNN = model
    def train_CNN(self, sess, train_x, train_y):
        print('Using data augmentation for training Resnet ..')
        augment = Cifar10Augment()
        my_training(sess, self.x, self.y,
                    self.CNN_loss,
                    self.CNN_train_step,
                    [self.CNN_accuracy], ['accuracy'],
                    train_x, train_y,
                    num_epochs=200,
                    data_augment=augment,
                    patience=10)
    def setup_trainstep(self):
        """Resnet needs scheduled training rate decay.

        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        values: 1e-3, 1e-1, 1e-2, 1e-3, 0.5e-3
        boundaries: 0, 80, 120, 160, 180
        
        "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
        
        """
        # FIXME what optimizer to use?
        self.AE_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.AE_loss, var_list=self.AE_vars)
        self.adv_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.adv_loss, var_list=self.AE_vars)

        global_step = tf.train.get_or_create_global_step()
        # schedules = [[0, 0.1], [400, 0.01], [600, 0.001]]
        schedules = [[0, 1e-3], [400, 1e-3], [800, 0.5e-3]]
        boundaries = [s[0] for s in schedules][1:]
        values = [s[1] for s in schedules]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values)
        momentum = 0.9
        # FIXME verify the learning rate schedule is functioning
        # self.CNN_train_step = tf.train.MomentumOptimizer(
        self.CNN_train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(
            # learning_rate, momentum).minimize(
                self.CNN_loss,
                global_step=global_step,
                var_list=self.CNN_vars+self.FC_vars)
        # self.CNN_train_step = tf.train.AdamOptimizer(0.001).minimize(
        #     self.CNN_loss, var_list=self.CNN_vars+self.FC_vars)
    def xshape(self):
        return (32,32,3,)
    def yshape(self):
        return (10,)
    def setup_AE(self):
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        # denoising
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        # at this point the representation is (7, 7, 32)
        # (?, 8, 8, 32) for cifar10

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        # here the fisrt channel is 3 for CIFAR model
        decoded = keras.layers.Conv2D(3, (3, 3), padding='same')(x)
        
        self.AE1 = keras.models.Model(inputs, decoded)
        self.AE = keras.models.Model(inputs, keras.layers.Activation('sigmoid')(decoded))
        
    def setup_attack_params(self):
        self.FGSM_params = {'eps': 4}
        self.PGD_params = {'eps': 8.,
                           'nb_iter': 10,
                           'eps_iter': 2.}
        self.JSMA_params = {}
        self.CW_params = {'max_iterations': 1000,
                          'learning_rate': 0.2}
    
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


def __test():

    grid_show_image(inputs, 5, 2, 'orig.pdf')
    np.argmax(labels, 1)
    np.argmax(targets, 1)
    

    # accuracy
    grid_show_image(inputs, 5, 2)
    
    sess.run([model.accuracy],
             feed_dict={model.x: inputs, model.y: targets})

    with tf.Session() as sess:
        train(sess, model, loss)


