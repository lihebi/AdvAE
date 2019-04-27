import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cleverhans.model
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2


from utils import *

BATCH_SIZE = 128
NUM_EPOCHS = 100

CLIP_MIN = 0.
CLIP_MAX = 1.


class CNNModel(cleverhans.model.Model):
    def __init__(self):
        self.setup_model()
    # abstract interfaces
    def x_shape(self):
        assert(False)
        return None
    def y_shape(self):
        assert(False)
        return None
    def rebuild_model(self):
        assert(False)
        return None
    # general methods
    def setup_model(self):
        shape = self.x_shape()
        # DEBUG I'm trying the input layer vs. placeholders
        self.x = keras.layers.Input(shape=self.x_shape(), dtype='float32')
        self.y = tf.placeholder(shape=self.y_shape(), dtype='float32')
        self.logits = self.rebuild_model(self.x)
        self.preds = tf.argmax(self.logits, axis=1)
        self.probs = tf.nn.softmax(self.logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))
        # FIXME num_classes
        # self.num_classes = np.product(self.y_shape[1:])
        self.ce_grad = tf.gradients(self.cross_entropy(), self.x)[0]
    def cross_entropy_with(self, y):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y))
    def logps(self):
        """Symbolic TF variable returning an Nx1 vector of log-probabilities"""
        return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)
    def predict(self, x):
        return keras.models.Model(self.x, self.logits)(x)
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}


    @staticmethod
    def compute_accuracy(sess, x, y, preds, x_test, y_test):
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
            dtype=tf.float32))
        return sess.run(acc, feed_dict={x: x_test, y: y_test})

    # loss functions. Using different loss function would result in
    # different model to compare
    def certainty_sensitivity(self):
        """Symbolic TF variable returning the gradients (wrt each input
        component) of the sum of log probabilities, also interpretable
        as the sensitivity of the model's certainty to changes in `X`
        or the model's score function

        """
        crossent_w_1 = self.cross_entropy_with(tf.ones_like(self.y) / self.num_classes)
        return tf.gradients(crossent_w_1, self.x)[0]
    def l1_certainty_sensitivity(self):
        """Cache the L1 loss of that product"""
        return tf.reduce_mean(tf.abs(self.certainty_sensitivity()))
    def l2_certainty_sensitivity(self):
        """Cache the L2 loss of that product"""
        return tf.nn.l2_loss(self.certainty_sensitivity())
    def cross_entropy(self):
        """Symbolic TF variable returning information distance between the
        model's predictions and the true labels y

        """
        return self.cross_entropy_with(self.y)
    def cross_entropy_grads(self):
        """Symbolic TF variable returning the input gradients of the cross
        entropy.  Note that if you pass in y=(1^{NxK})/K, this returns
        the same value as certainty_sensitivity.

        """
        # return tf.gradients(self.cross_entropy(), self.x)[0]
        # FIXME why [0]
        return tf.gradients(self.cross_entropy(), self.x)

    def l1_double_backprop(self):
        """L1 loss of the sensitivity of the cross entropy"""
        return tf.reduce_sum(tf.abs(self.cross_entropy_grads()))

    def l2_double_backprop(self):
        """L2 loss of the sensitivity of the cross entropy"""
        # tf.sqrt(tf.reduce_mean(tf.square(grads)))
        return tf.nn.l2_loss(self.cross_entropy_grads())
    def gradients_group_lasso(self):
        grads = tf.abs(self.cross_entropy_grads())
        return tf.reduce_mean(tf.sqrt(
            keras.layers.AvgPool3D(4, 4, 'valid')(tf.square(grads))
            + 1e-10))
    
class MNIST_CNN(CNNModel):
    def x_shape(self):
        # return [None, 28*28]
        return (28,28,1,)
    def y_shape(self):
        return [None, 10]
        # return (10,)
    def rebuild_model(self, xin):
        # FIXME may not need reshape layer
        x = keras.layers.Reshape([28, 28, 1])(xin)
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

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        return logits
    
# model = DenoisedCNN()

class DenoisedCNN(cleverhans.model.Model):
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
        
        self.x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        self.y = keras.layers.Input(shape=(10,), dtype='float32')
        # when self.x is feeding adv input, this acts as the clean reference
        self.x_ref = keras.layers.Input(shape=(28,28,1,), dtype='float32')

        rec = self.AE(self.x)
        self.rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=rec, labels=self.x_ref))

        clean_rec = self.AE(self.x_ref)
        self.clean_rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=clean_rec, labels=self.x_ref))
        
        self.logits = self.FC(self.CNN(self.AE(self.x)))
        self.ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))
        self.ce_grad = tf.gradients(self.ce_loss, self.x)[0]

        clean_logits = self.FC(self.CNN(self.AE(self.x_ref)))
        self.clean_ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=clean_logits, labels=self.y))
        
        self.preds = tf.argmax(self.logits, axis=1)
        # self.probs = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))
        self.adv_train_step = tf.train.AdamOptimizer(0.001).minimize(self.ce_loss, var_list=self.AE_vars)
        self.adv_rec_train_step = tf.train.AdamOptimizer(0.001).minimize(self.rec_loss, var_list=self.AE_vars)

        # TODO monitoring each of these losses and adjust weights
        self.unified_adv_loss = self.rec_loss + self.clean_rec_loss + self.ce_loss + self.clean_ce_loss
        self.unified_adv_train_step = tf.train.AdamOptimizer(0.001).minimize(self.unified_adv_loss, var_list=self.AE_vars)

        # self.clean_x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # self.noise_x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # # self.x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # self.y = tf.placeholder(shape=[None, 10], dtype='float32')

        # loss function
        # ce_grad, used for my_fast_PGD
        # self.ce_grad = tf.gradients(ce_loss, self.noise_x)[0]
        # denoising objective
    @staticmethod
    def compute_accuracy(sess, x, y, preds, x_test, y_test):
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
            dtype=tf.float32))
        return sess.run(acc, feed_dict={x: x_test, y: y_test})
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(self.AE(x)))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}
    def setup_AE(self):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        """From noise_x to x."""
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
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        self.AE = keras.models.Model(inputs, decoded)
        
    def save_CNN(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.CNN_vars)
        saver.save(sess, path)

    def save_AE(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.save(sess, path)

    def load_CNN(self, sess, path):
        saver = tf.train.Saver(var_list=self.CNN_vars)
        saver.restore(sess, path)

    def load_AE(self, sess, path):
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.restore(sess, path)

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
        

    def train_CNN(self, sess, clean_x, clean_y):
        """Train CNN part of the graph."""
        # TODO I'm going to compute a hash of current CNN
        # architecture. If the file already there, I'm just going to
        # load it.
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        logits = self.FC(self.CNN(inputs))
        model = keras.models.Model(inputs, logits)
        def loss(y_true, y_pred):
            return tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=y_pred, labels=y_true))
        optimizer = keras.optimizers.Adam(0.001)
        # 'rmsprop'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        with sess.as_default():
            model.fit(clean_x, clean_y, verbose=2)

    def test_CNN(self, sess, clean_x, clean_y):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')
        
        logits = self.FC(self.CNN(inputs))
        preds = tf.argmax(logits, axis=1)
        probs = tf.nn.softmax(logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={inputs: clean_x, labels: clean_y})
        print('accuracy: {}'.format(acc))
        
    def train_AE(self, sess, clean_x):
        noise_factor = 0.5
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)
        
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)
        # def loss(y_true, y_pred):
        #     return tf.reduce_mean(
        #         tf.nn.softmax_cross_entropy_with_logits(
        #             logits=y_pred, labels=y_true))
        # model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])
        optimizer = keras.optimizers.Adam(0.001)
        # 'adadelta'
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy')
        with sess.as_default():
            model.fit(noisy_x, clean_x, epochs=5, batch_size=128, verbose=2)

    def test_AE(self, sess, clean_x):
        # Testing denoiser
        noise_factor = 0.5
        # clean_x = test_x
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)

        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)

        rec = sess.run(rec, feed_dict={inputs: noisy_x[:5]})
        print('generating png ..')
        to_view = np.concatenate((noisy_x[:5], rec), 0)
        grid_show_image(to_view, 5, 2, 'AE_out.png')
        print('done')

    def test_Entire(self, sess, clean_x, clean_y):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')
        
        logits = self.FC(self.CNN(self.AE(inputs)))
        preds = tf.argmax(logits, axis=1)
        probs = tf.nn.softmax(logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={inputs: clean_x, labels: clean_y})
        print('accuracy: {}'.format(acc))

    def test_Adv_Denoiser(self, sess, clean_x, clean_y):
        """Visualize the denoised image against adversarial examples."""
        # TODO
        # FIXME can I pass this self?
        adv_x_concrete = my_fast_PGD(sess, self, clean_x, clean_y)
        
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')

        rec = self.AE(inputs)
        
        logits = self.FC(self.CNN(self.AE(inputs)))
        preds = tf.argmax(logits, axis=1)
        probs = tf.nn.softmax(logits)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={inputs: adv_x_concrete, labels: clean_y})

        print('accuracy: {}'.format(acc))
        denoised_x = sess.run(rec, feed_dict={inputs: adv_x_concrete, labels: clean_y})
        print('generating png ..')
        to_view = np.concatenate((noisy_x[:5], rec), 0)
        grid_show_image(to_view, 5, 2, 'AE_out.png')
        print('done')


    def train_AE_high(self, sess, clean_x):
        """Train AE using high level feature guidance."""
        # high-level guided loss function
        high_rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.CNN(self.clean_x),
                labels=self.CNN(self.decoded)))
        pass
        
    def train_AE_adv(self, sess):
        # FIXME can I pass myself as a class instance like this?
        my_adv_training(sess, self, self.ce_loss,
                        var_list=self.AE_vars, do_init=False)

class DenoisedCNN_Var0(DenoisedCNN):
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
class DenoisedCNN_Var1(DenoisedCNN):
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
class DenoisedCNN_Var2(DenoisedCNN):
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


def my_adv_training(sess, model, loss, metrics, train_step, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    """Adversarially training the model. Each batch, use PGD to generate
    adv examples, and use that to train the model.

    Print clean and adv rate each time.

    Early stopping when the clean rate is good and adv rate does not
    improve.

    """
    # FIXME refactor this with train.
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()

    best_loss = math.inf
    best_epoch = 0
    # DEBUG
    PATIENCE = 2
    patience = PATIENCE
    print_interval = 20

    saver = tf.train.Saver(max_to_keep=10)
    for i in range(num_epochs):
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // batch_size
        ct = 0
        for j in range(nbatch):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            clean_dict = {model.x: batch_x, model.y: batch_y, model.x_ref: batch_x}

            # generate adv examples

            # Using cleverhans library to generate PGD intances. This
            # approach is constructing a lot of graph strucutres, and
            # it is not clear what it does. The runnning is quite slow.
            
            # adv_x = my_PGD(sess, model)
            # adv_x_concrete = sess.run(adv_x, feed_dict=clean_dict)

            # Using a homemade PGD adapted from mnist_challenge is
            # much faster. But there is a caveat. See below.
            
            adv_x_concrete = my_fast_PGD(sess, model, batch_x, batch_y)
            
            adv_dict = {model.x: adv_x_concrete, model.y: batch_y, model.x_ref: batch_x}
            # evaluate current accuracy
            clean_acc = sess.run(model.accuracy, feed_dict=clean_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            # actual training
            #
            # clean training turns out to be important when using
            # homemade PGD. Otherwise the training does not
            # progress. Using cleverhans PGD won't need a clean data,
            # and still approaches the clean accuracy fast. However
            # since the running time of cleverhans PGD is so slow, the
            # overall running time is still far worse.
            #
            # _, l, a, m = sess.run([train_step, loss, model.accuracy, metrics],
            #                    feed_dict=clean_dict)
            _, l, a, m = sess.run([train_step, loss, model.accuracy, metrics],
                               feed_dict=adv_dict)
            ct += 1
            if ct % print_interval == 0:
                print('{} / {}: clean acc: {:.5f}, \tadv acc: {:.5f}, \tloss: {:.5f}, \tmetrics: {}'
                      .format(ct, nbatch, clean_acc, adv_acc, l, m))
        print('EPOCH ends, calculating total loss')
        # evaluate the loss for whole epoch
        adv_x_concrete = my_fast_PGD(sess, model, val_x, val_y)
        adv_dict = {model.x: adv_x_concrete, model.y: val_y, model.x_ref: val_x}
        adv_acc, l, m = sess.run([model.accuracy, loss, metrics], feed_dict=adv_dict)

        # save the weights
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        
        if best_loss < l:
            patience -= 1
        else:
            best_loss = l
            best_epoch = i
            patience = PATIENCE
        print('EPOCH {}: loss: {:.5f}, adv acc: {:.5f}, patience: {}, metrics: {}'
              .format(i, l, adv_acc, patience, m))
        if patience <= 0:
            # restore the best model
            saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
            # verify if the best model is restored
            adv_x_concrete = my_fast_PGD(sess, model, val_x, val_y)
            adv_dict = {model.x: adv_x_concrete, model.y: val_y, model.x_ref: val_x}
            adv_acc, l = sess.run([model.accuracy, loss], feed_dict=adv_dict)
            print('Best loss: {}, restored loss: {}'.format(l, best_loss))
            # FIXME this is not equal, as there's randomness in PGD?
            # assert abs(l - best_loss) < 0.001
            print('Early stopping .., best loss: {}'.format(best_loss))
            break
    
def train(sess, model, loss):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    best_loss = math.inf
    best_epoch = 0
    patience = 5
    print_interval = 500

    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(max_to_keep=10)
    for i in range(NUM_EPOCHS):
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // BATCH_SIZE
        
        for j in range(nbatch):
            start = j * BATCH_SIZE
            end = (j+1) * BATCH_SIZE
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            _, l, a = sess.run([train_step, loss, model.accuracy],
                               feed_dict={model.x: batch_x,
                                          model.y: batch_y})
        print('EPOCH ends, calculating total loss')
        l, a = sess.run([loss, model.accuracy],
                        feed_dict={model.x: val_x,
                                   model.y: val_y})
        print('EPOCH {}: loss: {:.5f}, acc: {:.5f}' .format(i, l, a,))
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        if best_loss < l:
            patience -= 1
            if patience <= 0:
                # restore the best model
                saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
                print('Early stopping .., best loss: {}'.format(best_loss))
                # verify if the best model is restored
                l, a = sess.run([loss, model.accuracy],
                                feed_dict={model.x: val_x,
                                           model.y: val_y})
                assert l == best_loss
                break
        else:
            best_loss = l
            best_epoch = i
            patience = 5


def generate_victim(test_x, test_y):
    inputs = []
    labels = []
    targets = []
    nlabels = 10
    for i in range(nlabels):
        for j in range(1000):
            x = test_x[j]
            y = test_y[j]
            if i == np.argmax(y):
                inputs.append(x)
                labels.append(y)
                onehot = np.zeros(nlabels)
                onehot[(i+1) % 10] = 1
                targets.append(onehot)
                break
    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    return inputs, labels, targets

def my_FGSM(sess, model):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX
    }
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(model.x, **fgsm_params)
    return adv_x

def my_PGD(sess, model):
    pgd = ProjectedGradientDescent(model, sess=sess)
    pgd_params = {'eps': 0.3,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': CLIP_MIN,
                  'clip_max': CLIP_MAX}
    adv_x = pgd.generate(model.x, **pgd_params)
    return adv_x

def my_fast_PGD(sess, model, x, y):
    epsilon = 0.3
    k = 40
    a = 0.01
    
    # loss = model.cross_entropy()
    # grad = tf.gradients(loss, model.x)[0]
    
    # FGSM, PGD, JSMA, CW
    # without random
    # 0.97, 0.97, 0.82, 0.87
    # using random, the performance improved a bit:
    # 0.99, 0.98, 0.87, 0.93
    #
    # adv_x = np.copy(x)
    adv_x = x + np.random.uniform(-epsilon, epsilon, x.shape)
    
    for i in range(k):
        g = sess.run(model.ce_grad, feed_dict={model.x: adv_x, model.y: y})
        adv_x += a * np.sign(g)
        adv_x = np.clip(adv_x, x - epsilon, x + epsilon)
        # adv_x = np.clip(adv_x, 0., 1.)
        adv_x = np.clip(adv_x, CLIP_MIN, CLIP_MAX)
    return adv_x

def my_JSMA(sess, model):
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': CLIP_MIN, 'clip_max': CLIP_MAX,
                   'y_target': None}
    return jsma.generate(model.x, **jsma_params)
    

def my_CW(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 # 'y_target': model.y,
                 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 100,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

def my_CW_targeted(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 'y_target': model.y,
                 # 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 10,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

        
def mynorm(a, b, p):
    size = a.shape[0]
    delta = a.reshape((size,-1)) - b.reshape((size,-1))
    return np.linalg.norm(delta, ord=p, axis=1)

def test_model_against_attack(sess, model, attack_func, inputs, labels, targets, prefix=''):
    """
    testing against several attacks
    """
    # generate victims
    adv_x = attack_func(sess, model)
    # adv_x = my_CW(sess, model)
    adv_preds = model.predict(adv_x)
    if targets is None:
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: labels})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: labels})
        adv_acc = model.compute_accuracy(sess, model.x, model.y, adv_preds, inputs, labels)
    else:
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: targets})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: targets})
        adv_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, labels)

    l2 = np.mean(mynorm(inputs, adv_x_concrete, 2))

    pngfile = prefix + '-out.png'
    print('Writing adv examples to {} ..'.format(pngfile))
    grid_show_image(adv_x_concrete[:10], 5, 2, pngfile)
    
    print('True label: {}'.format(np.argmax(labels, 1)))
    print('Predicted label: {}'.format(np.argmax(adv_preds_concrete, 1)))
    print('Adv input accuracy: {}'.format(adv_acc))
    print("L2: {}".format(l2))
    if targets is not None:
        target_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, targets)
        print('Targeted label: {}'.format(np.argmax(targets, 1)))
        print('Targeted accuracy: {}'.format(target_acc))

def train_double_backprop(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        c=100
        # TODO verify the results in their paper
        loss = model.cross_entropy() + c * model.l2_double_backprop()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)


def train_group_lasso(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        # group lasso
        c=100
        loss = model.cross_entropy() + c * model.gradients_group_lasso()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_ce(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_adv(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        train_step = tf.train.AdamOptimizer(0.001).minimize(model.ce_loss)

        # my_adv_training does not contain initialization of variable
        init = tf.global_variables_initializer()
        sess.run(init)

        my_adv_training(sess, model, loss, train_step=train_step, batch_size=128, num_epochs=50)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

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
    

def train_models_main():
    # training of CNN models
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "saved_model/double-backprop.ckpt")
        
    
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

def train_denoising(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = DenoisedCNN()
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
        # 1. train CNN
        # TODO train more epochs
        model.train_CNN(sess, train_x, train_y)
        model.test_CNN(sess, test_x, test_y)
        # 2. train denoiser
        model.train_AE(sess, train_x)
        model.test_AE(sess, test_x)
        model.test_Entire(sess, test_x, test_y)
        # 3. train denoiser using adv training, with high level feature guidance
        acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
        print('Model accuracy on clean data: {}'.format(acc))
        
        # my_adv_training(sess, model, model.ce_loss, [],
        #                 model.adv_train_step)
        # my_adv_training(sess, model, model.rec_loss, [],
        #                 model.adv_rec_train_step)

        print('Adv training ..')
        my_adv_training(sess, model, model.unified_adv_loss,
                        [model.rec_loss, model.clean_rec_loss, model.ce_loss, model.clean_ce_loss],
                        model.unified_adv_train_step,
                        num_epochs=2)
        print('Done. Saving model ..')
        pCNN = os.path.join(path, 'CNN.ckpt')
        pAE = os.path.join(path, 'AE.ckpt')
        model.save_CNN(sess, pCNN)
        model.save_AE(sess, pAE)
    
def __test_denoising(path):
    train_denoising('saved_model/AdvDenoiser')
    
    tf.reset_default_graph()
    
    sess = tf.Session()
    
    model = DenoisedCNN()
    model = DenoisedCNN_Var1()
    model = DenoisedCNN_Var2()
    
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    init = tf.global_variables_initializer()
    sess.run(init)

    model.train_CNN(sess, train_x, train_y)
    model.test_CNN(sess, test_x, test_y)
    
    model.load_CNN(sess, 'saved_model/AdvDenoiser/CNN.ckpt')
    
    model.load_AE(sess, 'saved_model/AdvDenoiser/AE.ckpt')

    
    model.test_CNN(sess, test_x, test_y)
    model.test_AE(sess, test_x)
    model.test_Entire(sess, test_x, test_y)
    acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
    print('Model accuracy on clean data: {}'.format(acc))

def __test():
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file('saved_model/AE.ckpt', tensor_name='', all_tensors=True)

def __test():
    train_double_backprop("saved_model/double-backprop.ckpt")
    train_group_lasso("saved_model/group-lasso.ckpt")
    train_ce("saved_model/ce.ckpt")
    train_adv("saved_model/adv.ckpt")
    # train_adv("saved_model/adv2.ckpt")
    
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

    # adv_x_concrete = my_fast_PGD(sess, model, inputs, labels)
    # my_PGD(sess, model)
    # adv_x = my_CW(sess, model)
    # adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: labels})
    # adv_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, labels)
    

def __test():

    grid_show_image(inputs, 5, 2, 'orig.png')
    np.argmax(labels, 1)
    np.argmax(targets, 1)
    

    # accuracy
    grid_show_image(inputs, 5, 2)
    
    sess.run([model.accuracy],
             feed_dict={model.x: inputs, model.y: targets})

    with tf.Session() as sess:
        train(sess, model, loss)


