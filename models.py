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
from tf_utils import *

from ae_models import *
from cnn_models import *

from attacks import CLIP_MIN, CLIP_MAX
from attacks import *
from train import *
from resnet import resnet_v2, lr_schedule

class AdvAEModel(cleverhans.model.Model):
    """Implement a denoising auto encoder.
    """
    def __init__(self, cnn_model, ae_model, inputs=None, targets=None):
        # this CNN model is used as victim for attacks
        self.cnn_model = cnn_model
        self.CNN = cnn_model.CNN
        self.FC = cnn_model.FC

        self.ae_model = ae_model
        self.AE = ae_model.AE
        self.AE1 = ae_model.AE1

        if inputs is not None:
            self.x = inputs
        else:
            self.x = keras.layers.Input(shape=self.cnn_model.xshape(), dtype='float32')
        if targets is not None:
            self.y = targets
        else:
            self.y = keras.layers.Input(shape=self.cnn_model.yshape(), dtype='float32')

        # adv_x = my_PGD(self, self.x, params=self.cnn_model.PGD_params)
        self.logits = self.FC(self.CNN(self.AE(self.x)))

        self.setup_loss()
        self.setup_trainloss()

    @staticmethod
    def NAME():
        "Override by children models."
        return "AdvAE"
    def metric_funcs(self):
        def advloss(ytrue, ypred): return self.adv_loss
        def advacc(ytrue, ypred): return self.adv_accuracy
        def acc(ytrue, ypred): return self.accuracy
        def cnnacc(ytrue, ypred): return self.CNN_accuracy
        def obliacc(ytrue, ypred): return self.obli_accuracy
        return [advloss, advacc, acc, cnnacc, obliacc]

    # cleverhans
    def predict(self, x):
        return self.FC(self.CNN(self.AE(x)))
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}

    def setup_trainloss(self):
        """DEPRECATED Overwrite by subclasses to customize the loss.
        """
        raise NotImplementedError()
        self.adv_loss = self.A2_loss
        
    def setup_loss(self):
        """
        - 0: pixel
        - 1: high level conv
        - 2: logits

        The data:
        - C: Clean
        - N: Noisy
        - A: Adv
        - P: Post
        """
        print('Seting up loss ..')
        high = self.CNN(self.x)
        # logits = self.FC(high)
        
        rec = self.AE(self.x)
        rec_high = self.CNN(rec)
        rec_logits = self.FC(rec_high)

        self.C0_loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        # I'm replacing all "1" losses from xent to l1
        # FIXME maybe replace "0" losses as well
        # self.C1_loss = my_sigmoid_xent(rec_high, tf.nn.sigmoid(high))
        self.C1_loss = tf.reduce_mean(tf.abs(rec_high - high))
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
        # self.A1_loss = my_sigmoid_xent(adv_rec_high, tf.nn.sigmoid(high))
        self.A1_loss = tf.reduce_mean(tf.abs(adv_rec_high - high))
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

        # self.B1_loss = my_sigmoid_xent(obliadv_high, tf.nn.sigmoid(high))
        self.B1_loss = tf.reduce_mean(tf.abs(obliadv_high - high))
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

    def train_Adv(self, sess, train_x, train_y, plot_prefix=''):
        """Adv training."""
        # (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)
        outputs = keras.layers.Activation('softmax')(self.FC(self.CNN(self.AE(self.x))))
        model = keras.models.Model(self.x, outputs)
        if self.ae_model.NAME() != 'identityAE':
            self.CNN.trainable = False
            self.FC.trainable = False
        # self.AE.trainable = False
        def myloss(ytrue, ypred):
            # return my_softmax_xent(logits=outputs, labels=ytrue)
            return self.adv_loss
        def advacc(ytrue, ypred): return self.adv_accuracy
        def acc(ytrue, ypred): return self.accuracy
        def cnnacc(ytrue, ypred): return self.CNN_accuracy
        def obliacc(ytrue, ypred): return self.obli_accuracy
            
        with sess.as_default():
            callbacks = [get_lr_scheduler(),
                         # get_lr_reducer(patience=5),
                         # DEBUG
                         get_es(patience=5),
                         get_mc('best_model.hdf5')]
            model.compile(loss=myloss,
                          metrics=[cnnacc, acc, obliacc, advacc],
                          optimizer=keras.optimizers.Adam(lr=1e-3),
                          target_tensors=self.y)
            model.fit(train_x, train_y,
                      validation_split=0.1,
                      # validation_data=(val_x, val_y),
                      epochs=100,
                      callbacks=callbacks)
            model.load_weights('best_model.hdf5')

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


        def myl2dist(x1, x2, logits, labels):
            # calculate the correct indices
            indices = tf.not_equal(tf.argmax(logits, axis=1),
                                   tf.argmax(labels, 1))
            diff = tf.boolean_mask(x1, indices) - tf.boolean_mask(x2, indices)
            diff_norm = tf.norm(diff, ord=2, axis=(1,2))
            # print(diff.shape[0])
            # print(diff_norm.shape[0])
            # ? != ?
            # assert diff.shape[0] == diff_norm.shape[0]
            return tf.reduce_mean(diff_norm)

        # This is tricky. I want to get the baseline of attacking
        # clean image AND VANILLA CNN. Thus, I'm using the proxy
        # cnn model.
        obliadv = attack(self.cnn_model, self.x)
        obliadv_rec = self.AE(obliadv)
        
        baseline_logits = self.FC(self.CNN(obliadv))
        baseline_acc = my_accuracy_wrapper(baseline_logits, self.y)
        obliadv_logits = self.FC(self.CNN(self.AE(obliadv)))
        obliadv_acc = my_accuracy_wrapper(obliadv_logits, self.y)
        # obliadv_l2 = myl2dist(obliadv, self.x, obliadv_logits, self.y)
        obliadv_l2 = tf.constant(1, tf.float32)
        
        whiteadv = attack(self, self.x)
        whiteadv_rec = self.AE(whiteadv)
        whiteadv_logits = self.FC(self.CNN(self.AE(whiteadv)))
        whiteadv_acc = my_accuracy_wrapper(whiteadv_logits, self.y)
        # whiteadv_l2 = myl2dist(whiteadv, self.x, whiteadv_logits, self.y)
        whiteadv_l2 = tf.constant(1, tf.float32)

        # remember to compare visually postadv with rec (i.e. AE(x))
        postadv = attack(self.cnn_model, self.AE(self.x))
        postadv_logits = self.FC(self.CNN(postadv))
        postadv_acc = my_accuracy_wrapper(postadv_logits, self.y)
        # postadv_l2 = myl2dist(postadv, self.x, postadv_logits, self.y)
        postadv_l2 = tf.constant(1, tf.float32)
        
        to_run = {}

        to_run['obliadv'] = obliadv, tf.argmax(baseline_logits, axis=1), baseline_acc, tf.constant(-1)
        to_run['obliadv_rec'] = obliadv_rec, tf.argmax(obliadv_logits, axis=1), obliadv_acc, obliadv_l2
        to_run['whiteadv'] = whiteadv, tf.constant(0, shape=(10,)), tf.constant(-1), tf.constant(-1)
        to_run['whiteadv_rec'] = whiteadv_rec, tf.argmax(whiteadv_logits, axis=1), whiteadv_acc, whiteadv_l2
        to_run['postadv'] = postadv, tf.argmax(postadv_logits, axis=1), postadv_acc, postadv_l2

        data_torun = {
            'obliadv': baseline_acc,
            'obliadv_rec': obliadv_acc,
            'whiteadv_rec': whiteadv_acc,
            'whiteadv_l2': whiteadv_l2,
        }
        
        print('Testing attack {} ..'.format(name))
        t = time.time()
        res, data = sess.run([to_run, data_torun], feed_dict={self.x: test_x, self.y: test_y})
        print('Attack done. Time: {:.3f}'.format(time.time()-t))

        tmp = {}
        for key in data:
            tmp[key] = float(data[key])
        data = tmp

        for key in res:
            tmp_images, tmp_titles, acc, l2 = res[key]
            images.append(tmp_images[:10])
            if acc == -1:
                titles.append(['']*10)
                fringe = '{}\n{}'.format(name, key)
            else:
                titles.append(tmp_titles[:10])
                fringe = '{}\n{}\n{:.3f}\nL2: {:.3f}'.format(name, key, acc, l2)
            fringes.append(fringe)
            print(fringe.replace('\n', ' '))
        return images, titles, fringes, data

    def __test():
        c = tf.constant([1,2,3,4])
        i = tf.constant([1,2])
        ii = tf.constant([1,0,0,1])

        v = tf.constant([[[1,2,3],
                          [4,5,200]],
                         [[4,5,6],
                          [1,2,3]]], dtype=tf.float32)
        vv = tf.constant([[1,2,3], [4,5,6]], dtype=tf.float32)
        keras.backend.eval(tf.norm(v, ord=2, axis=(1,2)))
        keras.backend.eval(tf.norm(vv))

        x = keras.backend.eval(tf.constant(1, tf.float32))
        type(x)
        type(float(x))

        
        iii = tf.cast(ii, tf.bool)
        x = tf.gather(c, i)
        xx = tf.gather(c, tf.where(ii))
        xxx = tf.where(iii, c, tf.zeros_like(c))
        xxxx = tf.boolean_mask(c, ii)
        # x = tf.gather_nd(c, i)
        keras.backend.eval(xxxx)
        keras.backend.eval(tf.zeros_like(c))
        keras.backend.eval(tf.where(ii))
        c[iii]
        tf.slice(c, iii)
        

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
        data = {
            'CNN clean': float(accs[0]),
            'AE clean': float(accs[1]),
            'AE noisy': float(accs[2])
        }
        images.append(res['x'][:10])
        titles.append(res['y'][:10])
        fringes.append('x\n{:.3f}'.format(accs[0]))
        
        for name, acc in zip(['rec', 'noisy_x', 'noisy_rec'], [accs[1], accs[2], accs[2]]):
            images.append(res[name][:10])
            titles.append(['']*10)
            fringes.append('{}\n{:.3f}'.format(name, acc))

        return images, titles, fringes, data
        
        

    def test_all(self, sess, test_x, test_y, attacks=[],
                 # filename='out.pdf',
                 save_prefix='test',
                 num_sample=100, batch_size=100):
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
        all_data = []
        
        # nbatch = num_sample // batch_size
        # for i in range(nbatch):
        #     start = i * batch_size
        #     end = (i+1) * batch_size
        #     batch_x = test_x[indices[start:end]]
        #     batch_y = test_y[indices[start:end]]

        images, titles, fringes, data = self.test_Model(sess, test_x, test_y)
        all_images.extend(images)
        all_titles.extend(titles)
        all_fringes.extend(fringes)
        all_data.append(data)
        for name in attacks:
            images, titles, fringes, data = self.test_attack(sess, test_x, test_y, name)
            all_images.extend(images)
            all_titles.extend(titles)
            all_fringes.extend(fringes)
            all_data.append({name: data})
        
        print('Plotting result ..')
        plot_filename = 'images/{}.pdf'.format(save_prefix)
        data_filename = 'images/{}.json'.format(save_prefix)
        grid_show_image(all_images, filename=plot_filename, titles=all_titles, fringes=all_fringes)
        # saving data
        with open(data_filename, 'w') as fp:
            json.dump(all_data, fp, indent=4)
        print('Done. Saved to {}'.format(plot_filename))
    
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
class C0_B2_Model(AdvAEModel):
    def setup_trainloss(self):
        self.adv_loss = self.B2_loss
    def NAME():
        return 'C0_B2'
class A2_B2_C2_Model(AdvAEModel):
    def NAME():
        return 'A2_B2_C2'
    def setup_trainloss(self):
        self.adv_loss = self.A2_loss + self.B2_loss + self.C2_loss
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
        # FIXME DEBUG add mu for all models
        # mu = 0.5
        self.adv_loss = self.A2_loss + self.C0_loss
        
class C2_A2_Model(AdvAEModel):
    def NAME():
        return 'C2_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.C2_loss)

class C1_A1_Model(AdvAEModel):
    def NAME():
        return 'C1_A1'
    def setup_trainloss(self):
        self.adv_loss = self.A1_loss + self.C1_loss
class C1_A2_Model(AdvAEModel):
    def NAME():
        return 'C1_A2'
    def setup_trainloss(self):
        self.adv_loss = (self.A2_loss + self.C1_loss)

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

