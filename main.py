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


from utils import *
from model import *
from attacks import *

def train_denoising(model_cls, saved_folder, prefix, advprefix, train_x, train_y, run_adv=True, overwrite=False):
    """Train AdvAE.

    - prefix: CNN and AE ckpt prefix
    - advprefix: AdvAE ckpt prefix
    """
    if not os.path.exists('images'):
        os.makedirs('images')
    print('====== Denoising training for {} ..'.format(advprefix))
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = model_cls()
        
        init = tf.global_variables_initializer()
        sess.run(init)

        # 1. train CNN
        pCNN = os.path.join(saved_folder, '{}-CNN.ckpt'.format(prefix))
        if not os.path.exists(pCNN+'.meta'):
            print('Trianing CNN ..')
            model.train_CNN(sess, train_x, train_y)
            print('Saving model ..')
            model.save_CNN(sess, pCNN)
        else:
            print('Trained, directly loadding ..')
            model.load_CNN(sess, pCNN)
        # print('Testing CNN ..')
        # model.test_CNN(sess, test_x, test_y)

        # 2. train denoiser
        pAE = os.path.join(saved_folder, '{}-AE.ckpt'.format(prefix))
        if not os.path.exists(pAE+'.meta'):
            print('Training AE ..')
            model.train_AE(sess, train_x)
            print('Saving model ..')
            model.save_AE(sess, pAE)
        else:
            print('Trained, directly loadding ..')
            model.load_AE(sess, pAE)
        # print('Testing AE ..')
        # model.test_AE(sess, test_x)
        # model.test_Entire(sess, test_x, test_y)
        
        # 3. train denoiser using adv training, with high level feature guidance
        # acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
        # print('Model accuracy on clean data: {}'.format(acc))
        # model.test_all(sess, val_x, val_y, run_CW=False)

        if run_adv:
            pAdvAE = os.path.join(saved_folder, '{}-AdvAE.ckpt'.format(advprefix))
            # overwrite controls only the AdvAE weights. CNN and AE
            # weights do not need to be retrained because I'm not
            # experimenting with changing training or loss for that.
            if not os.path.exists(pAdvAE+'.meta') or overwrite:
                print('Adv training AdvAE ..')
                my_adv_training(sess, model,
                                train_x, train_y,
                                plot_prefix=advprefix,
                                # DEBUG small num epochs to speed up
                                # debugging, large one to report final
                                # results
                                num_epochs=10)
                print('Saving model ..')
                model.save_AE(sess, pAdvAE)
            else:
                print('Trained, directly loadding ..')
                model.load_AE(sess, pAdvAE)
            # print('Testing AdvAE ..')
            # model.test_Adv_Denoiser(sess, test_x[:10], test_y[:10])

    
def __test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_cifar10_data()
    model = MyResNet()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Trianing CNN ..')
    model.train_CNN(sess, train_x, train_y)
    
    print('Saving model ..')
    model.save_CNN(sess, pCNN)
def main2():
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    train_denoising(MyResNet, 'saved_models', 'resnet', 'Default', train_x, train_y)
    

def main1():
    # tf.random.set_random_seed(0)
    # np.random.seed(0)
    # random.seed(0)
    
    clses = [
        # default model
        # ONE
        A2_Model,
        # A1_Model,
        # A12_Model,
        # TWO
        N0_A2_Model,
        C0_A2_Model,
        # C2_A2_Model,
        # THREE
        C0_N0_A2_Model,
        # C2_N0_A2_Model
    ]
    
    for cls in clses:
        train_denoising(cls, 'saved_models', 'aecnn', cls.name())
    
    # train_denoising(FCAdvAEModel, 'saved_models', 'fccnn', 'FCAdvAE')
    
    # train_denoising(PostNoisy_Adv_Model, 'saved_models', 'aecnn', 'PostNoisy_Adv')

def load_default_model():
    tf.reset_default_graph()
    sess = tf.Session()
    # model = AdvAEModel()
    model = MyResNet()
    init = tf.global_variables_initializer()
    sess.run(init)
    # model.load_CNN(sess, 'saved_models/aecnn-CNN.ckpt')
    model.load_CNN(sess, 'saved_models/resnet-CNN.ckpt')
    # model.load_AE(sess, 'saved_models/A2-AdvAE.ckpt')
    return model, sess

def __test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    clses = [ AdvAEModel, Noisy_Adv_Model, Rec_Adv_Model,
              Rec_Noisy_Adv_Model, CleanAdv_Model, Noisy_CleanAdv_Model,
              High_Model, High_Adv_Model ]
    for cls in clses:
        tf.reset_default_graph()
        sess = tf.Session()
        model = cls()
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load_CNN(sess, 'saved_models/aecnn-CNN.ckpt')
        model.load_AE(sess, 'saved_models/{}-AdvAE.ckpt'.format(cls.name()))
        model.test_all(sess, test_x, test_y, run_CW=True, filename='images/{}-test-result.pdf'.format(cls.name()))
def __test_denoising(path):
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # training
    train_denoising(AdvAEModel, 'saved_models', 'aecnn', 'AdvAE', train_x, train_y)
    
    train_denoising(AdvAEModel_Var1, 'saved_models', 'cnn1', None, run_adv=False)
    train_denoising(AdvAEModel_Var2, 'saved_models', 'cnn2', None, run_adv=False)
    
    train_denoising(FC_CNN, 'saved_models', 'fccnn', 'fccnn')
    
    # only train CNN and AE for these CNN architecture. The AdvAE weights will be loaded.
    
    # testing

    tf.reset_default_graph()
    
    sess = tf.Session()

    # different CNN models
    model = AdvAEModel()
    model = AdvAEModel_Var1()
    model = AdvAEModel_Var2()
    model = MyResNet()

    # init session
    init = tf.global_variables_initializer()
    sess.run(init)
    # init = tf.local_variables_initializer()
    # sess.run(init)

    # load different CNNs
    model.load_CNN(sess, 'saved_models/aecnn-CNN.ckpt')
    model.load_CNN(sess, 'saved_models/1-CNN.ckpt')
    model.load_CNN(sess, 'saved_models/2-CNN.ckpt')
    model.load_CNN(sess, 'saved_models/resnet-CNN.ckpt')

    # load the same AE
    model.load_AE(sess, 'saved_models/aecnn-AE.ckpt')
    
    model.load_AE(sess, 'saved_models/AdvAE-AdvAE.ckpt')
    model.load_AE(sess, 'saved_models/PostNoisy_Adv-AdvAE.ckpt')
    model.load_AE(sess, 'saved_models/resnet-AE.ckpt')

    model.train_CNN(sess, train_x, train_y)

    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    model, sess = load_default_model()
    model.train_AE(sess, train_x, train_y)

    # testing
    model.test_all(sess, test_x, test_y, attacks=[], num_sample=1000)
    
    model.test_all(sess, test_x, test_y, attacks=[])
    model.test_all(sess, test_x, test_y, attacks=['FGSM', 'PGD'])
    model.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'])
    model.test_all(sess, test_x, test_y, attacks=['CW', 'JSMA', 'FGSM', 'PGD'])

    model.save_CNN(sess, 'saved_models/resnet-CNN.ckpt')
    # test_against_attacks(sess, model)

    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    model.test_CW(sess, test_x, test_y)

def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    tf.reset_default_graph()
    
    sess = tf.Session()

    cnn = CNNModel()
    ae = AEModel(cnn)
    adv = AdvAEModel(cnn, ae)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    cnn.train_CNN(sess, train_x, train_y)
    cnn.save_weights(sess, 'tmp/cnn.hdf5')
    cnn.load_weights(sess, 'tmp/cnn.hdf5')
    
    ae.train_AE(sess, train_x, train_y)
    # ae.train_AE_keras(sess, train_x, train_y)
    ae.save_weights(sess, 'tmp/ae.hdf5')
    ae.load_weights(sess, 'tmp/ae.hdf5')

    adv.train_Adv(sess, train_x, train_y)

    adv.test_all(sess, test_x, test_y, attacks=[])
    adv.test_all(sess, test_x, test_y, attacks=['FGSM', 'PGD'])
    adv.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'])

    
    
def __test():
    x=tf.constant([[0.1,0.1,0.2,0.2],
                   [0.1,0.1,0.2,0.2]])
    y=tf.constant([[0.2,0.2,0.1,0.1],
                   [0.2,0.2,0.1,0.1]])
    # sess.run(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=y)))
    sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=y))
    sess.run(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=y))


if __name__ == '__main__':
    main2()
    
def __test():
    # This snippet demonstrate that even if I give a new
    # tf.variable_scope or tf.name_scope, keras will add new suffix to
    # the variables. This is quite annoying when saving and loading
    # model weights.
    def conv_block(inputs, filters, kernel_size, strides, scope):
        '''Create a simple Conv --> BN --> ReLU6 block'''

        with tf.name_scope(scope):
            x = tf.keras.layers.Conv2D(filters, kernel_size, strides)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(tf.nn.relu6)(x)
            return x
    tf.reset_default_graph()
    inputs = tf.keras.Input(shape=[224, 224, 3], batch_size=1, name='inputs')
    hidden = conv_block(inputs, 32, 3, 2, scope='block_1')
    outputs = conv_block(hidden, 64, 3, 2, scope='block_2')
    
    # model = tf.keras.Model(inputs, outputs)
    model.summary()

def foo():
    pass
