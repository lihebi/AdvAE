import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import cleverhans.model
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation


from utils import *
from tf_utils import *
from models import *
from defensegan_models import *
from attacks import *

DATA_AUG = 6
LMBDA = .1
AUG_BATCH_SIZE = 512
LEARNING_RATE = .001
HOLDOUT = 150

BATCH_SIZE = 128
NB_EPOCHS = 10
NB_EPOCHS_S = 10

def train_sub(sess, bbox, sub, holdout_x, holdout_y):
    """This function will train sub in place.

    Adapted from cleverhans_tutorial/mnist_blackbox.

    bbox: black box model as an oracle
    sub: substitute model with random initialization
    holdout_x: initial data
    holdout_y: initial data, [int], not categorical
    """
    # Define the Jacobian symbolically using TensorFlow.
    grads = jacobian_graph(sub.predict(sub.x), sub.x, 10)
    
    # Train the substitute and augment dataset alternatively.
    for rho in range(DATA_AUG):
        print("Substitute training epoch #" + str(rho))

        # train the sub model
        # FIXME if I just pass holdout_y, it still works! Why????
        sub.train_CNN(sess, holdout_x, keras.utils.to_categorical(holdout_y))
        # augment the data by Jacobian
        if rho < DATA_AUG - 1:
            print("Augmenting substitute training data.")

            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            # Perform the Jacobian augmentation to generate new synthetic inputs.
            holdout_x = jacobian_augmentation(sess, sub.x, holdout_x, holdout_y, grads,
                                              lmbda_coef * LMBDA, AUG_BATCH_SIZE)
            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box.
            holdout_y = np.hstack([holdout_y, holdout_y])
            holdout_x_prev = holdout_x[int(len(holdout_x) / 2):]
            eval_params = {'batch_size': BATCH_SIZE}

            # To initialize the local variables of Defense-GAN.
            # tf_init_uninitialized(sess)

            bbox_val = cleverhans.utils_tf.batch_eval(sess, [bbox.x],
                                                      [bbox.predict(bbox.x)],
                                                      [holdout_x_prev],
                                                      args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model.
            holdout_y[int(len(holdout_x) / 2):] = np.argmax(bbox_val, axis=1)

def test_sub(sess, bbox, sub, test_x, test_y):
    """Perform black box attack and get result accuracy.

    bbox: trained model
    sub: untrained substitue model

    """
    # FIXME this seems to cause troubles
    indices = random.sample(range(test_x.shape[0]), test_x.shape[0])
    test_x = test_x[indices]
    test_y = test_y[indices]
    
    # test_x = np.copy(test_x)
    # test_y = np.copy(test_y)
    # np.random.shuffle(test_x)
    # np.random.shuffle(test_y)
    
    holdout_x = test_x[:150]
    holdout_y = test_y[:150]
    test_x = test_x[150:]
    test_y = test_y[150:]

    # FIXME this is no longer necessary as I shuffled the array
    num_sample = 1000
    batch_size = 100
    indices = random.sample(range(test_x.shape[0]), num_sample)
    test_x = test_x[indices]
    test_y = test_y[indices]
    
    train_sub(sess, bbox, sub, holdout_x, np.argmax(holdout_y, axis=1))
    
    res = {}
    
    for name in ['FGSM', 'CW', 'PGD']:

        print('Performing {} attack ..'.format(name))
        if name is 'FGSM':
            attack = lambda m, x: my_FGSM(m, x, params=sub.FGSM_params)
        elif name is 'PGD':
            attack = lambda m, x: my_PGD(m, x, params=sub.PGD_params)
        elif name is 'JSMA':
            attack = lambda m, x: my_JSMA(m, x, params=sub.JSMA_params)
        elif name is 'CW':
            attack = lambda m, x: my_CW(m, sess, x, sub.y, params=sub.CW_params)
        else:
            assert False
        adv_x = attack(sub, sub.x)
        # evaluate on model
        logits = bbox.predict(adv_x)
        acc = my_accuracy_wrapper(logits=logits, labels=sub.y)

        # batch
        nbatch = num_sample // batch_size
        for i in range(nbatch):
            start = i * batch_size
            end = (i+1) * batch_size
            batch_x = test_x[start:end]
            batch_y = test_y[start:end]
            print('Batch {} / {}'.format(i, nbatch))

            bbox_acc = sess.run(acc, feed_dict={sub.x: batch_x, sub.y: batch_y})

            if name not in res: res[name] = 0.
            res[name] += bbox_acc
        res[name] /= nbatch
        print('Attack {} black-box acc: {}'.format(name, bbox_acc))
    return res

def __test():
    # training substitute model for blackbox testing
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    # the initial seed for substitute model
    holdout_x = test_x[:150]
    holdout_y = test_y[:150]
    test_x = test_x[150:]
    test_y = test_y[150:]
    
    sess = create_tf_session()
    
    cnn = MNISTModel()
    sub = MNISTModel()
    
    sub = DefenseGAN_c()

    # this cnn is only used to provide shape
    ae = AEModel(cnn)

    advae = A2_Model(cnn, ae)

    tf_init_uninitialized(sess)

    # load cnn and advae weights
    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')
    ae.load_weights(sess, 'saved_models/MNIST-mnistcnn-ae-A2-AdvAE.hdf5')
    # use that as oracle to train the sub model

    # np.argmax(holdout_y, axis=1)
    # train_sub(sess, advae, sub, holdout_x, np.argmax(holdout_y, axis=1))
    train_sub(sess, cnn, sub, holdout_x, np.argmax(holdout_y, axis=1))
    # after getting the sub model, run attack on sub model
    adv_x = my_PGD(sub, sub.x)
    # evaluate on model
    # logits = advae.predict(adv_x)
    logits = cnn.predict(adv_x)
    acc = my_accuracy_wrapper(logits=logits, labels=sub.y)
    bbox_acc = sess.run(acc, feed_dict={sub.x: test_x[:100], sub.y: test_y[:100]})
    print('black-box acc: {}'.format(bbox_acc))
