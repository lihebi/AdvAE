import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cleverhans.model

from attacks import *
from utils import *


BATCH_SIZE = 128
NUM_EPOCHS = 100


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
