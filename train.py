import keras
import tensorflow as tf
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import pickle
import os

import cleverhans.model
from attacks import *
from utils import *


BATCH_SIZE = 128
NUM_EPOCHS = 100

def my_lr_schedule(epoch):
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 70:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    return lr

def get_lr_scheduler():
    # DEPRECATED
    # TODO add parameter to adjust the schedule
    # FIXME this conflicts with lr reducer
    return keras.callbacks.LearningRateScheduler(my_lr_schedule, verbose=1)

def get_lr_reducer(patience=4):
    return keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.5,
                                             cooldown=0,
                                             patience=patience,
                                             # DEBUG
                                             verbose=1,
                                             min_lr=0.5e-4)
def get_es(patience=10, min_delta=0):
    return keras.callbacks.EarlyStopping(monitor='val_loss',
                                         # TODO adjust this to make ensemble training faster
                                         min_delta=min_delta,
                                         patience=patience,
                                         # DEBUG
                                         verbose=1,
                                         # NEW I'm resoringg the weights from keras
                                         restore_best_weights=True,
                                         mode='auto')
def get_mc(filename='best_model.hdf5'):
    # DEPRECATED
    # TODO I'm adding filename patterns, and will enable training from
    # where we left last time
    # FIXME this will conflict with early stopping
    #
    # filename = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    return keras.callbacks.ModelCheckpoint(filename,
                                           monitor='val_loss', mode='auto',
                                           verbose=1,
                                           save_weights_only=True,
                                           save_best_only=True)

def adv_training_plot(data, names, filename, split):
    # print('Plotting internal training process ..')
    if split:
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12.8, 9.6), dpi=300)
        for ax, metric, name in zip(axes.reshape(-1), np.transpose(data), names):
            # print(type(ax))
            ax.plot(metric, label=name)
            ax.set_title(name)
            legend = ax.legend()
        plt.subplots_adjust(hspace=0.5)
    else:
        fig, ax = plt.subplots(figsize=(12.8, 9.6), dpi=300)
        for metric, name in zip(np.transpose(data), names):
            ax.plot(metric, label=name)
        legend = ax.legend()
    # DEBUG output to images folder
    plt.savefig(os.path.join('images', filename))
    plt.close(fig)
    # print('saved to {}'.format(filename))

def __test():
    fig, axes = plt.subplots(nrows=16, ncols=3, figsize=(6.4, 10.8), dpi=500)
    for ax in axes.reshape(-1):
        ax.plot([1,2,3,4])
    plt.subplots_adjust(hspace=0.5, bottom = 0, top=1)
    plt.savefig('out.pdf')
    plt.close(fig)

class Cifar10Augment():
    """Augment CIFAR10 data.

    I'm going to build a TF graph for augmenting it. Then, I have two
    ways to use it:

    1. each time I generate a batch, I'm going to feed the batch data
    into the graph and obtain the concrete augmented data.

    2. build the augmentation graph into the model.

    I'm going to try the first method here, as the testing anti adv
    examples should be generated against only original data.

    But during adv training, the adv examples can (and should) be
    generated from augmented data.

    FIXME seems that original image (without these augmentation) are
    not used for training at all. Why is that? Is it because clean
    ones are not necessary anymore after using these random augmentations?

    """
    def __init__(self):
        self.x = keras.layers.Input(shape=(32,32,3), dtype='float32')
        model = self.__augment_model()
        self.y = model(self.x)
        
    def augment_batch(self, sess, batch_x):
        return sess.run(self.y, feed_dict={self.x: batch_x})
        
    @staticmethod
    def __augment_model():
        IMAGE_SIZE = 32

        # create augmentation computational graph
        # x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        inputs = keras.layers.Input(shape=(32,32,3), dtype='float32')
        def pad_func(x):
            return tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, IMAGE_SIZE + 4, IMAGE_SIZE + 4),
                      x)
        padded = keras.layers.Lambda(pad_func)(inputs)
        def crop_func(x):
            return tf.map_fn(lambda img: tf.random_crop(
                img, [IMAGE_SIZE, IMAGE_SIZE, 3]),
                             x)
        cropped = keras.layers.Lambda(crop_func)(padded)

        def flip_func(x):
            return tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
        flipped = keras.layers.Lambda(flip_func)(cropped)
        return keras.models.Model(inputs, flipped)

class MyPlot(keras.callbacks.Callback):
    def __init__(self, plot_prefix):
        super().__init__()
        self.plot_prefix = plot_prefix
        self.batch_plot_data = {'metrics': [],
                                'loss': []}
        self.epoch_plot_data = {'metrics': [],
                                'loss': []}
    def on_epoch_begin(self, epoch, logs=None):
        print('MyPlot epoch begin')

    def on_epoch_end(self, epoch, logs=None):
        print('MyPlot epoch begin')
        # DEBUG
        print(logs.keys())
        # evaluate the loss on validation data, also batched
        nbatch = val_x.shape[0] // batch_size
        shuffle_idx = np.arange(val_x.shape[0])
        np.random.shuffle(shuffle_idx)
        all_l = []
        all_m = []
        print('validating on {} batches: '.format(nbatch), end='', flush=True)
        for j in range(nbatch):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_x = val_x[shuffle_idx[start:end]]
            batch_y = val_y[shuffle_idx[start:end]]
            feed_dict = {model_x: batch_x, model_y: batch_y}
            # actual training
            l, m = sess.run([loss, metrics], feed_dict=feed_dict)
            all_l.append(l)
            all_m.append(m)
            print('.', end='', flush=True)
        print('')
        l = np.mean(all_l)
        m = np.mean(all_m, 0)
        plot_data['simple'].append(m)

        if do_plot:
            print('plotting ..')
            adv_training_plot(plot_data['simple'], metric_names, 'training-process-simple-{}.pdf'.format(self.plot_prefix), True)
            
        # save plot data
        print('saving plot data ..')
        with open('images/train-data-{}.pkl'.format(self.plot_prefix), 'wb') as fp:
            pickle.dump(plot_data, fp)

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        """Save and plot batch training results."""
        print_interval = 20
        # metrics = self.model.metrics
        # metric_names = self.model.metric_names
        # DEBUG
        print(logs.keys())
        return
        
        logs = logs or {}
        
        if batch % print_interval == 0:
            l = logs.get('loss')
            m = {}
            # FIXME performance issue of running metrics every
            # time. So maybe I should evalute the metrics here. But
            # how can I access the training data here?
            for k in self.params['metrics']:
                if k in logs:
                    m[k] = logs[k]
            self.batch_plot_data['metrics'].append(m)
            self.batch_plot_data['loss'].append(l)

            # FIXME I should more focus on the validation
            # metrics instead of these training ones
            adv_training_plot(self.batch_plot_data['metrics'], metric_names,
                              'training-process-{}.pdf'.format(self.plot_prefix), False)
            adv_training_plot(self.batch_plot_data['metrics'], metric_names,
                              'training-process-split-{}.pdf'.format(self.plot_prefix), True)
            # plot loss
            fig, ax = plt.subplots(dpi=300)
            ax.plot(plot_data['loss'])
            plt.savefig('images/training-process-loss-{}.pdf'.format(self.plot_prefix))
            plt.close(fig)

    def on_train_begin(self, logs=None):
        print('MyPlot train begin')

    def on_train_end(self, logs=None):
        print('MyPlot train end')

def run_on_batch(sess, res, x, y, npx, npy, batch_size=BATCH_SIZE):
    nbatch = npx.shape[0] // batch_size
    shuffle_idx = np.arange(npx.shape[0])
    np.random.shuffle(shuffle_idx)
    print('running on {} batches: '.format(nbatch), end='', flush=True)
    allres = []
    for i in range(nbatch):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_x = npx[shuffle_idx[start:end]]
        batch_y = npy[shuffle_idx[start:end]]
        feed_dict = {x: batch_x, y: batch_y,
                     keras.backend.learning_phase(): 0
        }
        # actual training
        npres = sess.run(res, feed_dict=feed_dict)
        allres.append(npres)
        print('.', end='', flush=True)
    print('')
    return allres
