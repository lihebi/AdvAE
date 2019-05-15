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
    def __init__(self):
        super().__init__()
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
            adv_training_plot(plot_data['simple'], metric_names, 'training-process-simple-{}.pdf'.format(plot_prefix), True)
            
        # save plot data
        print('saving plot data ..')
        with open('images/train-data-{}.pkl'.format(plot_prefix), 'wb') as fp:
            pickle.dump(plot_data, fp)

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        """Save and plot batch training results."""
        print_interval = 20
        metrics = self.model.metrics
        metric_names = self.model.metric_names
        plot_prefix = 'keras_fit_testing'
        # DEBUG
        print(logs.keys())
        
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
                              'training-process-{}.pdf'.format(plot_prefix), False)
            adv_training_plot(self.batch_plot_data['metrics'], metric_names,
                              'training-process-split-{}.pdf'.format(plot_prefix), True)
            # plot loss
            fig, ax = plt.subplots(dpi=300)
            ax.plot(plot_data['loss'])
            plt.savefig('images/training-process-loss-{}.pdf'.format(plot_prefix))
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
        feed_dict = {x: batch_x, y: batch_y}
        # actual training
        npres = sess.run(res, feed_dict=feed_dict)
        allres.append(npres)
        print('.', end='', flush=True)
    print('')
    return allres

def my_training(sess, model_x, model_y,
                loss, train_step, metrics, metric_names,
                train_x, train_y,
                data_augment=None,
                keras_augment=None,
                do_plot=True,
                plot_prefix='', patience=2,
                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                print_interval=20):
    """Adversarially training the model. Each batch, use PGD to generate
    adv examples, and use that to train the model.

    Print clean and adv rate each time.

    Early stopping when the clean rate is good and adv rate does not
    improve.

    FIXME metrics and metric_names are not coming from the same source.

    """
    (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)

    best_loss = math.inf
    best_epoch = 0
    pat = patience

    plot_data = {'metrics': [],
                 'simple': [],
                 'loss': []}

    # FIXME seems that the saver is causing tf memory issues when
    # restoring. So trying keras.
    saver = tf.train.Saver(max_to_keep=100)
    
    for i in range(num_epochs):
        epoch_t = time.time()
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // batch_size
        t = time.time()
        # DEBUG trying keras preprocessor
        for j in range(nbatch):
        # j = 0
        # for batch_x, batch_y in keras_augment.datagen.flow(train_x, train_y, batch_size=batch_size):
        #     j += 1
        #     if j > nbatch: break
            start = j * batch_size
            end = (j+1) * batch_size
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]

            if data_augment is not None:
                batch_x = data_augment.augment_batch(sess, batch_x)
            
            feed_dict = {model_x: batch_x, model_y: batch_y}
            # actual training
            # 0.12
            sess.run([train_step], feed_dict=feed_dict)
            # print('training time: {}'.format(time.time() - t))

            if j % print_interval == 0:
                # t = time.time()
                # 0.46
                l, m = sess.run([loss, metrics], feed_dict=feed_dict)
                # print('metrics time: {}'.format(time.time() - t))

                # the time for running train step
                t1 = time.time() - t
                
                plot_data['metrics'].append(m)
                plot_data['loss'].append(l)

                if do_plot:
                    # this introduces only small overhead
                    #
                    # FIXME I should more focus on the validation
                    # metrics instead of these training ones
                    adv_training_plot(plot_data['metrics'], metric_names,
                                      'training-process-{}.pdf'.format(plot_prefix), False)
                    adv_training_plot(plot_data['metrics'], metric_names,
                                      'training-process-split-{}.pdf'.format(plot_prefix), True)
                    # plot loss
                    fig, ax = plt.subplots(dpi=300)
                    ax.plot(plot_data['loss'])
                    plt.savefig('images/training-process-loss-{}.pdf'.format(plot_prefix))
                    plt.close(fig)
                
                print('Batch {} / {}, \t time {:.2f} ({:.2f}), loss: {:.5f},\t metrics: {}'
                      .format(j, nbatch, time.time()-t, t1, l, ', '.join(['{:.5f}'.format(mi) for mi in m])))
                t = time.time()
            # print('plot time: {}'.format(time.time() - t))

        # evaluate the loss on validation data, also batched
        print('EPOCH ends, calculating total validation loss ..')
        allres = run_on_batch(sess, [loss, metrics], model_x, model_y, val_x, val_y)
        all_l = []
        all_m = []
        for l,m in allres:
            all_l.append(l)
            all_m.append(m)
        # FIXME these are list of list. np.mean should have returned a
        # single number, as I wanted.
        l = np.mean(all_l)
        m = np.mean(all_m, 0)
        plot_data['simple'].append(m)

        if do_plot:
            print('plotting ..')
            adv_training_plot(plot_data['simple'], metric_names, 'training-process-simple-{}.pdf'.format(plot_prefix), True)
            
        # save plot data
        print('saving plot data ..')
        with open('images/train-data-{}.pkl'.format(plot_prefix), 'wb') as fp:
            pickle.dump(plot_data, fp)

        # save the weights
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        
        if best_loss < l:
            pat -= 1
        else:
            best_loss = l
            best_epoch = i
            pat = patience
        print('EPOCH {} / {}: loss: {:.5f}, time: {:.2f}, patience: {},\t metrics: {}'
              .format(i, num_epochs, l, time.time()-epoch_t, pat, ', '.join(['{:.5f}'.format(mi) for mi in m])))
        if pat <= 0:
            # restore the best model
            filename = 'tmp/epoch-{}'.format(best_epoch)
            print('restoring from {} ..'.format(filename))
            saver.restore(sess, filename)

            # verify if the best model is restored
            # FIXME I need batched version. This will run out of GPU memory.
            # clean_dict = {model_x: val_x, model_y: val_y}
            # l,m = sess.run([loss, metrics], feed_dict=clean_dict)
            # print('Best loss: {}, restored loss: {}'.format(best_loss, l))
            # print('Corresponding metris: {}'.format(m))
            
            # FIXME this is not equal, as there's randomness in PGD?
            # assert abs(l - best_loss) < 0.001
            print('Early stopping .., best loss: {}'.format(best_loss))
            # TODO I actually want to implement lr reducer here. When
            # the given patience is reached, I'll reduce the learning
            # rate by half if it is larger than a minimum (say 1e-6),
            # and restore the patience counting. This should be done
            # after restoring the previously best model.
            #
            # FIXME however, seems that I need to create a new
            # optimizer, and thus a new train step. But creating this
            # would create new variables. I need to initialize those
            # variables alone. This should not be very difficult.
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

