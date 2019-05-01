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

from attacks import CLIP_MIN, CLIP_MAX

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
    
# model = AdvAEModel()

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
        
        self.x = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        self.y = keras.layers.Input(shape=(10,), dtype='float32')
        # FIXME get rid of x_ref, integrate attack into model
        # when self.x is feeding adv input, this acts as the clean reference
        self.x_ref = keras.layers.Input(shape=(28,28,1,), dtype='float32')

        self.setup_loss()
        self.setup_prediction()
        self.setup_metrics()
        self.setup_trainstep()
        self.unified_adv_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.unified_adv_loss, var_list=self.AE_vars)
    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(self.AE(x)))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def myFGSM(self, x):
        # FGSM attack
        fgsm_params = {
            'eps': 0.3,
            'clip_min': CLIP_MIN,
            'clip_max': CLIP_MAX
        }
        fgsm = FastGradientMethod(self)
        adv_x = fgsm.generate(x, **fgsm_params)
        return adv_x
    def myPGD(self, x):
        pgd = ProjectedGradientDescent(self)
        pgd_params = {'eps': 0.3,
                      'nb_iter': 40,
                      'eps_iter': 0.01,
                      'clip_min': CLIP_MIN,
                      'clip_max': CLIP_MAX}
        adv_x = pgd.generate(x, **pgd_params)
        return adv_x
    def myJSMA(self, x):
        jsma = SaliencyMapMethod(self)
        jsma_params = {'theta': 1., 'gamma': 0.1,
                       'clip_min': CLIP_MIN, 'clip_max': CLIP_MAX,
                       'y_target': None}
        return jsma.generate(x, **jsma_params)
    def myCW(self, sess, x, y, targeted=False):
        """When targeted=True, remember to put target as y."""
        # CW attack
        cw = CarliniWagnerL2(self, sess=sess)
        yname = 'y_target' if targeted else 'y'
        cw_params = {'binary_search_steps': 1,
                     yname: y,
                     'max_iterations': 10000,
                     'learning_rate': 0.2,
                     # setting to 100 instead of 128, because the test_x
                     # has 10000, and the last 16 left over will cause
                     # exception
                     'batch_size': 100,
                     'initial_const': 10,
                     'clip_min': CLIP_MIN,
                     'clip_max': CLIP_MAX}
        adv_x = cw.generate(x, **cw_params)
        return adv_x

    @staticmethod
    def add_noise(x):
        noise_factor = 0.5
        noisy_x = x + noise_factor * tf.random.normal(shape=tf.shape(x), mean=0., stddev=1.)
        noisy_x = tf.clip_by_value(noisy_x, CLIP_MIN, CLIP_MAX)
        return noisy_x
    @staticmethod
    def ce_wrapper(logits, labels):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels))
    @staticmethod
    def accuracy_wrapper(logits, labels):
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
        high = self.CNN(self.x)
        # logits = self.FC(high)
        
        rec = self.AE(self.x)
        rec_high = self.CNN(rec)
        rec_logits = self.FC(rec_high)

        self.rec_loss = self.ce_wrapper(rec, self.x)
        self.rec_high_loss = self.ce_wrapper(rec_high, high)
        self.rec_ce_loss = self.ce_wrapper(rec_logits, self.y)

        noisy_x = self.add_noise(self.x)
        noisy_rec = self.AE(noisy_x)
        noisy_rec_high = self.CNN(noisy_rec)
        noisy_rec_logits = self.FC(noisy_rec_high)

        self.noisy_rec_loss = self.ce_wrapper(noisy_rec, self.x)
        self.noisy_rec_high_loss = self.ce_wrapper(noisy_rec_high, high)
        self.noisy_rec_ce_loss = self.ce_wrapper(noisy_rec_logits, self.y)

        adv_x = self.myPGD(self.x)
        adv_high = self.CNN(adv_x)
        adv_logits = self.FC(adv_high)

        adv_rec = self.AE(adv_x)
        adv_rec_high = self.CNN(adv_rec)
        adv_rec_logits = self.FC(adv_rec_high)
        
        self.adv_rec_loss = self.ce_wrapper(adv_rec, self.x)
        self.adv_rec_high_loss = self.ce_wrapper(adv_rec_high, high)
        self.adv_rec_ce_loss = self.ce_wrapper(adv_rec_logits, self.y)

        # pre-denoiser: Trian denoiser such that the denoised x is
        # itself robust
        postadv = self.myPGD(rec)
        postadv_high = self.CNN(postadv)
        postadv_logits = self.FC(postadv_high)

        self.postadv_rec_loss = self.ce_wrapper(postadv, self.x)
        self.postadv_rec_high_loss = self.ce_wrapper(postadv_high, high)
        self.postadv_rec_ce_loss = self.ce_wrapper(postadv_logits, self.y)
        
    def setup_prediction(self):
        # no AE, just CNN
        CNN_logits = self.FC(self.CNN(self.x))
        self.CNN_accuracy = self.accuracy_wrapper(CNN_logits, self.y)
        # Should work on rec_logits
        logits = self.FC(self.CNN(self.AE(self.x)))
        self.accuracy = self.accuracy_wrapper(logits, self.y)
        
        adv_logits = self.FC(self.CNN(self.AE(self.myPGD(self.x))))
        self.adv_accuracy = self.accuracy_wrapper(adv_logits, self.y)

        postadv_logits = self.FC(self.CNN(self.myPGD(self.AE(self.x))))
        self.postadv_accuracy = self.accuracy_wrapper(postadv_logits, self.y)

        noisy_x = self.add_noise(self.x)
        noisy_logits = self.FC(self.CNN(self.AE(noisy_x)))
        self.noisy_accuracy = self.accuracy_wrapper(noisy_logits, self.y)
    def setup_metrics(self):
        # TODO monitoring each of these losses and adjust weights
        self.metrics = [
            # clean data
            self.rec_loss, self.rec_high_loss, self.rec_ce_loss, self.accuracy,
            # noisy data
            self.noisy_rec_loss, self.noisy_rec_high_loss, self.noisy_rec_ce_loss, self.noisy_accuracy,
            # adv data
            self.adv_rec_loss, self.adv_rec_high_loss, self.adv_rec_ce_loss, self.adv_accuracy,
            # postadv
            self.postadv_rec_loss, self.postadv_rec_high_loss, self.postadv_rec_ce_loss, self.postadv_accuracy
        ]
        
        self.metric_names = [
            'rec_loss', 'rec_high_loss', 'rec_ce_loss', 'accuracy',
            'noisy_rec_loss', 'noisy_rec_high_loss', 'noisy_rec_ce_loss', 'noisy_accuracy',
            'adv_rec_loss', 'adv_rec_high_loss', 'adv_rec_ce_loss', 'adv_accuracy',
            'postadv_rec_loss', 'postadv_rec_high_loss', 'postadv_rec_ce_loss', 'postadv_accuracy']

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
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=5, verbose=0,
                                               mode='auto')
            mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                                 monitor='val_loss', mode='auto', verbose=0,
                                                 save_weights_only=True,
                                                 save_best_only=True)
            model.fit(clean_x, clean_y, epochs=100, validation_split=0.1, verbose=2, callbacks=[es, mc])
            model.load_weights('best_model.ckpt')

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
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=5, verbose=0,
                                               mode='auto')
            mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                                 monitor='val_loss', mode='auto', verbose=0,
                                                 save_weights_only=True,
                                                 save_best_only=True)
            model.fit(noisy_x, clean_x, epochs=100, validation_split=0.1, verbose=2, callbacks=[es, mc])
            model.load_weights('best_model.ckpt')


    def test_all(self, sess, test_x, test_y, run_CW=False):
        """Test clean data x and y.

        - Use only CNN, to test whether CNN is functioning
        - Test AE, output png for before and after
        - Test AE + CNN, to test whether the entire system works

        CW is costly, so by default no running for it. If you want it,
        set it to true.

        """
        # inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        # labels = keras.layers.Input(shape=(10,), dtype='float32')

        indices = random.sample(range(test_x.shape[0]), 100)
        test_x = test_x[indices]
        test_y = test_y[indices]
        feed_dict = {self.x: test_x, self.y: test_y}
        
        accs = sess.run([self.CNN_accuracy, self.accuracy, self.noisy_accuracy], feed_dict=feed_dict)
        print('raw accuracies (raw CNN, AE clean, AE noisy): {}'.format(accs))

        to_run = {}
        titles = []
        # Test AE rec output before and after
        rec = self.AE(self.x)
        noisy_x = self.add_noise(self.x)
        noisy_rec = self.AE(noisy_x)

        to_run['x'] = self.x
        to_run['y'] = tf.argmax(self.y, 1)
        to_run['rec'] = rec
        to_run['noisy_x'] = noisy_x
        to_run['noisy_rec'] = noisy_rec

        # to_run += [self.x, rec, noisy_x, noisy_rec]
        # titles += ['x', 'rec', 'noisy_x', 'noisy_rec']
        # titles += [tf.constant('x', dtype=tf.string, shape=self.x.shape[0])]

        # JSMA is very slow ..
        attack_funcs = [self.myFGSM, self.myPGD, self.myJSMA]
        attack_names = ['FGSM', 'PGD', 'JSMA']
        # attack_funcs = [self.myFGSM, self.myPGD]
        # attack_names = ['FGSM', 'PGD']
        if run_CW:
            attack_funcs += [lambda x: self.myCW(sess, x, self.y)]
            attack_names += ['CW']

        # first class methods!
        for attack, name in zip(attack_funcs, attack_names):
            adv_x = attack(self.x)
            adv_rec = self.AE(adv_x)
            # remember to compare visually postadv with rec (i.e. AE(x))
            postadv = attack(self.AE(self.x))
            
            adv_logits = self.FC(self.CNN(self.AE(attack(self.x))))
            adv_accuracy = self.accuracy_wrapper(adv_logits, self.y)

            postadv_logits = self.FC(self.CNN(attack(self.AE(self.x))))
            postadv_accuracy = self.accuracy_wrapper(postadv_logits, self.y)

            to_run[name] = {}
            to_run[name]['adv_x'] = adv_x
            to_run[name]['adv_rec'] = adv_rec
            to_run[name]['postadv'] = postadv
            
            to_run[name]['adv_pred'] = tf.argmax(adv_logits, axis=1)
            to_run[name]['postadv_pred'] = tf.argmax(postadv_logits, axis=1)
            
            to_run[name]['adv_acc'] = adv_accuracy
            to_run[name]['postadv_acc'] = postadv_accuracy
            
            
            # TODO mark the correct and incorrect predictions
            # to_run += [adv_x, adv_rec, postadv]
            # TODO add prediction result
            # titles += ['{} adv_x'.format(name), 'adv_rec', 'postadv']
            
        # TODO select 5 correct and incorrect examples
        # indices = random.sample(range(test_x.shape[0]), 10)
        # test_x = test_x[indices]
        # test_y = test_y[indices]
        # feed_dict = {self.x: test_x, self.y: test_y}
        
        print('Testing AE and Adv attacks ..')
        res = sess.run(to_run, feed_dict=feed_dict)

        # select 10
        images = []
        titles = []
        
        images += list(res['x'][:10])
        tmp = list(res['y'][:10])
        tmp[0] = 'x: {}'.format(tmp[0])
        titles += tmp
        
        for name in ['rec', 'noisy_x', 'noisy_rec']:
            images += list(res[name][:10])
            titles += [name] + ['']*9
        for name in attack_names:
            images += list(res[name]['adv_x'][:10])
            tmp = list(res[name]['adv_pred'][:10])
            tmp[0] = '{} adv_x: {}'.format(name, tmp[0])
            titles += tmp
            
            images += list(res[name]['adv_rec'][:10])
            titles += ['adv_rec'] + ['']*9
            
            images += list(res[name]['postadv'][:10])
            tmp = list(res[name]['postadv_pred'][:10])
            tmp[0] = 'postadv: {}'.format(tmp[0])
            titles += tmp
            # titles += list(res[name]['postadv_pred'][:10])

            print('{} attack accuracy: adv: {}, postadv: {}'
                  .format(name, res[name]['adv_acc'], res[name]['postadv_acc']))
        
        # res = np.array(res)[:,:10]
        # images = np.concatenate(res)
        # titles = np.transpose([titles] * 10).reshape(-1)
        print('Plotting result ..')
        grid_show_image(images, 10, int(len(images)/10), filename='test-all-result.png', titles=titles)
        print('Done. Saved to {}'.format('test-all-result.png'))
    def setup_trainstep(self):
        """Overwrite by subclasses to customize the loss.

        Available loss terms:

        # clean data
        self.rec_loss, self.rec_high_loss, self.rec_ce_loss,
        
        # noisy data
        self.noisy_rec_loss, self.noisy_rec_high_loss, self.noisy_rec_ce_loss,
        
        # adv data
        self.adv_rec_loss, self.adv_rec_high_loss, self.adv_rec_ce_loss,
        
        # postadv
        self.postadv_rec_loss, self.postadv_rec_high_loss, self.postadv_rec_ce_loss,
        
        """
        self.unified_adv_loss = (self.adv_rec_ce_loss)

# stand alones (not likely to work)
class Post_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.postadv_rec_ce_loss)
# combine with adv loss
class Post_Adv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss
                                 + self.postadv_rec_ce_loss)
class Noisy_Adv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss
                                 + self.noisy_rec_loss)
class PostNoisy_Adv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss
                                 + self.postadv_rec_ce_loss + self.noisy_rec_loss)
# add clean data loss (probably add this to all)
class CleanAdv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss
                                 + self.rec_ce_loss)
class Post_CleanAdv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss + self.postadv_rec_ce_loss
                                 + self.rec_ce_loss)
class Noisy_CleanAdv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss + self.noisy_rec_loss
                                 + self.rec_ce_loss)
class PostNoisy_CleanAdv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss + self.postadv_rec_ce_loss + self.noisy_rec_loss
                                 + self.rec_ce_loss)
# high-level guided models
class High_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_high_loss)
class High_Adv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss + self.adv_rec_high_loss)
class PostHigh_Adv_Model(AdvAEModel):
    def setup_trainstep(self):
        self.unified_adv_loss = (self.adv_rec_ce_loss + self.postadv_rec_high_loss)
        
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

    grid_show_image(inputs, 5, 2, 'orig.png')
    np.argmax(labels, 1)
    np.argmax(targets, 1)
    

    # accuracy
    grid_show_image(inputs, 5, 2)
    
    sess.run([model.accuracy],
             feed_dict={model.x: inputs, model.y: targets})

    with tf.Session() as sess:
        train(sess, model, loss)


