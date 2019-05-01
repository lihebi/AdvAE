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

def my_adv_training(sess, model, loss, metrics, train_step, plot_prefix='', batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    """Adversarially training the model. Each batch, use PGD to generate
    adv examples, and use that to train the model.

    Print clean and adv rate each time.

    Early stopping when the clean rate is good and adv rate does not
    improve.

    FIXME metrics and metric_names are not coming from the same source.

    """
    # FIXME refactor this with train.
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()

    best_loss = math.inf
    best_epoch = 0
    # DEBUG
    PATIENCE = 2
    patience = PATIENCE
    print_interval = 20

    plot_data = {'metrics': [],
                 'simple': [],
                 'loss': []}

    saver = tf.train.Saver(max_to_keep=10)
    
    for i in range(num_epochs):
        epoch_t = time.time()
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // batch_size
        t = time.time()
        for j in range(nbatch):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            feed_dict = {model.x: batch_x, model.y: batch_y}
            # actual training
            # 0.12
            sess.run([train_step], feed_dict=feed_dict)
            # print('training time: {}'.format(time.time() - t))

            if j % print_interval == 0:
                # t = time.time()
                # 0.46
                l, m = sess.run([loss, metrics], feed_dict=feed_dict)
                # print('metrics time: {}'.format(time.time() - t))
                
                plot_data['metrics'].append(m)
                plot_data['loss'].append(l)
                # this introduces only small overhead
                adv_training_plot(plot_data['metrics'], model.metric_names, '{}-training-process.pdf'.format(plot_prefix), False)
                adv_training_plot(plot_data['metrics'], model.metric_names, '{}-training-process-split.pdf'.format(plot_prefix), True)
                # plot loss
                fig, ax = plt.subplots(dpi=300)
                ax.plot(plot_data['loss'])
                plt.savefig('images/{}-training-process-loss.pdf'.format(plot_prefix))
                plt.close(fig)
                
                print('Batch {} / {},   \t time {:.2f}, loss: {:.5f},\t metrics: {}'
                      .format(j, nbatch, time.time()-t, l, ', '.join(['{:.5f}'.format(mi) for mi in m])))
                t = time.time()
            # print('plot time: {}'.format(time.time() - t))
        print('EPOCH ends, calculating total loss')

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
            feed_dict = {model.x: batch_x, model.y: batch_y}
            # actual training
            l, m = sess.run([loss, metrics], feed_dict=feed_dict)
            all_l.append(l)
            all_m.append(m)
            print('.', end='', flush=True)
        print('')
        l = np.mean(all_l)
        m = np.mean(all_m, 0)
        plot_data['simple'].append(m)

        adv_training_plot(plot_data['simple'], model.metric_names, '{}-training-process-simple.pdf'.format(plot_prefix), True)
        # save plot data
        with open('images/{}-data.pkl'.format(plot_prefix), 'wb') as fp:
            pickle.dump(plot_data, fp)

        # save the weights
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        
        if best_loss < l:
            patience -= 1
        else:
            best_loss = l
            best_epoch = i
            patience = PATIENCE
        print('EPOCH {}: loss: {:.5f}, time: {:.2f}, patience: {},\t metrics: {}'
              .format(i, l, time.time()-epoch_t, patience, ', '.join(['{:.5f}'.format(mi) for mi in m])))
        if patience <= 0:
            # restore the best model
            saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
            # verify if the best model is restored
            clean_dict = {model.x: val_x, model.y: val_y}
            l = sess.run(loss, feed_dict=clean_dict)
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

