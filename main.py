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

def load_model(cnn_cls, ae_cls, advae_cls,
               saved_folder='saved_models',
               cnnprefix='', aeprefix='', advprefix='', load_adv=True):
    """If load_adv = False, try to load ae instead."""
    if not cnnprefix:
        cnnprefix = cnn_cls.NAME()
    if not advprefix:
        advprefix = advae_cls.NAME()
    if not aeprefix:
        aeprefix = ae_cls.NAME()
    tf.reset_default_graph()
    sess = tf.Session()
    
    cnn = cnn_cls()
    ae = ae_cls(cnn)
    adv = advae_cls(cnn, ae)
     
    pCNN = os.path.join(saved_folder, '{}-CNN.hdf5'.format(cnnprefix))
    pAE = os.path.join(saved_folder, '{}-{}-AE.hdf5'.format(cnnprefix, aeprefix))
    pAdvAE = os.path.join(saved_folder, '{}-{}-{}-AdvAE.hdf5'.format(cnnprefix, aeprefix, advprefix))
   
    init = tf.global_variables_initializer()
    sess.run(init)

    cnn.load_weights(sess, pCNN)
    if load_adv:
        ae.load_weights(sess, pAdvAE)
    else:
        ae.load_weights(sess, pAE)
    return adv, sess
    

def train_model(cnn_cls, ae_cls, advae_cls,
                train_x, train_y,
                saved_folder='saved_models',
                cnnprefix='', aeprefix='', advprefix='',
                run_adv=True, overwrite=False):
    """Train AdvAE.

    - prefix: CNN and AE ckpt prefix
    - advprefix: AdvAE ckpt prefix
    """
    if not os.path.exists('images'):
        os.makedirs('images')
    if not cnnprefix:
        cnnprefix = cnn_cls.NAME()
    if not advprefix:
        advprefix = advae_cls.NAME()
    if not aeprefix:
        aeprefix = ae_cls.NAME()
        
    pCNN = os.path.join(saved_folder, '{}-CNN.hdf5'.format(cnnprefix))
    pAE = os.path.join(saved_folder, '{}-{}-AE.hdf5'.format(cnnprefix, aeprefix))
    pAdvAE = os.path.join(saved_folder, '{}-{}-{}-AdvAE.hdf5'.format(cnnprefix, aeprefix, advprefix))

    # DEBUG early return here to avoid the big overhead of creating the graph
    if os.path.exists(pAdvAE):
        print('Already trained AdvAE model {}, break'.format(pAdvAE))
        return
    
    print('====== Denoising training for {} ..'.format(advprefix))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        cnn = cnn_cls()
        ae = ae_cls(cnn)
        adv = advae_cls(cnn, ae)

        # model = model_cls()
        
        init = tf.global_variables_initializer()
        sess.run(init)

        # 1. train CNN
        if not os.path.exists(pCNN):
            print('Trianing CNN ..')
            cnn.train_CNN(sess, train_x, train_y)
            print('Saving model to {} ..'.format(pCNN))
            cnn.save_weights(sess, pCNN)
        else:
            print('Trained, directly loading {} ..'.format(pCNN))
            cnn.load_weights(sess, pCNN)
        # print('Testing CNN ..')
        # model.test_CNN(sess, test_x, test_y)

        # 2. train denoiser
        # FIXME whether this pretraining is useful or not?
        if not os.path.exists(pAE):
            print('Training AE ..')
            ae.train_AE(sess, train_x, train_y)
            print('Saving model to {} ..'.format(pAE))
            ae.save_weights(sess, pAE)
        else:
            print('Trained, directly loading {} ..'.format(pAE))
            ae.load_weights(sess, pAE)
        
        # 3. train denoiser using adv training, with high level feature guidance
        # acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
        # print('Model accuracy on clean data: {}'.format(acc))
        # model.test_all(sess, val_x, val_y, run_CW=False)

        if run_adv:
            # overwrite controls only the AdvAE weights. CNN and AE
            # weights do not need to be retrained because I'm not
            # experimenting with changing training or loss for that.
            if not os.path.exists(pAdvAE) or overwrite:
                print('Trainng AdvAE ..')
                adv.train_Adv(sess, train_x, train_y)
                print('saving to {} ..'.format(pAdvAE))
                ae.save_weights(sess, pAdvAE)
            else:
                print('Already trained {} ..'.format(pAdvAE))
                # adv.load_AE(sess, pAdvAE)
            # print('Testing AdvAE ..')
            # model.test_Adv_Denoiser(sess, test_x[:10], test_y[:10])

    

def main_train():
    
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    # adv training baseline
    train_model(CNNModel, IdentityAEModel, A2_Model, train_x, train_y)

    # train models
    train_model(CNNModel, AEModel, A2_Model, train_x, train_y)
    train_model(CNNModel, AEModel, B2_Model, train_x, train_y)
    train_model(CNNModel, AEModel, N0_A2_Model, train_x, train_y)
    train_model(CNNModel, AEModel, C0_A2_Model, train_x, train_y)
    # train_model(CNNModel, AEModel, C0_N0_A2_Model, train_x, train_y)

    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # adv training baseline
    train_model(MyResNet, IdentityAEModel, A2_Model, train_x, train_y)

    # train models
    train_model(MyResNet, DunetModel, C0_A2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, C2_A2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, A2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, B2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, C0_B2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, A2_B2_C2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, C1_A1_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, C1_A2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, C2_A2_P2_Model, train_x, train_y)
    train_model(MyResNet, DunetModel, P2_Model, train_x, train_y)
    
    # train_model(MyResNet, DunetModel, N0_A2_Model, train_x, train_y)
    # train_model(MyResNet, DunetModel, C0_N0_A2_Model, train_x, train_y)

    # train_model(MyResNet, AEModel, Test_Model, train_x, train_y)

    # train_model(MyResNet, CifarAEModel, A2_Model, train_x, train_y)
    # train_model(MyResNet, DunetModel, A2_Model, train_x, train_y)
    
def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    cnn = MyResNet()
    cnn = MyResNet56()
    cnn = MyResNet110()
    cnn = CNNModel()
    ae = AEModel(cnn)
    ae = CifarAEModel(cnn)
    ae = TestCifarAEModel(cnn)
    ae = DunetModel(cnn)
    adv = A2_Model(cnn, ae)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    cnn.train_CNN(sess, train_x, train_y)
    cnn.train_CNN_keras(sess, train_x, train_y)

    cnn.load_weights(sess, 'saved_models/resnet-CNN.hdf5')
    ae.load_weights(sess, 'saved_models/resnet-cifarae-AE.hdf5')
    
    ae.train_AE(sess, train_x, train_y)
    ae.train_AE_keras(sess, train_x, train_y)
    
    adv.train_Adv(sess, train_x, train_y)

    ae.test_AE(sess, test_x, test_y)


def test_model(cnn_cls, ae_cls, advae_cls,
               test_x, test_y,
               saved_folder='saved_models',
               cnnprefix='', aeprefix='', advprefix='', force=False):


    # FIXME keep consistent with advae.name()
    model_name = '{}-{}-{}'.format(cnn_cls.NAME(), ae_cls.NAME(), advae_cls.NAME())
    filename = 'images/test-result-{}.pdf'.format(model_name)

    if not os.path.exists(filename) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls)
        # model.test_all(sess, test_x, test_y, attacks=[])
        # model.test_all(sess, test_x, test_y, attacks=[], num_sample=1000)
        print('testing {} ..'.format(model.name()))
        model.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'],
                       filename='images/test-result-{}.pdf'.format(model.name()))
        # model.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'])
        # model.test_all(sess, test_x, test_y, attacks=['CW', 'JSMA', 'FGSM', 'PGD'])
    else:
        print('Already tested, see {}'.format(filename))
    
def main_test():

    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    test_model(CNNModel, IdentityAEModel, A2_Model, test_x, test_y)
    
    test_model(CNNModel, AEModel, A2_Model, test_x, test_y)
    test_model(CNNModel, AEModel, B2_Model, test_x, test_y)
    test_model(CNNModel, AEModel, N0_A2_Model, test_x, test_y)
    test_model(CNNModel, AEModel, C0_A2_Model, test_x, test_y)
    # test_model(CNNModel, AEModel, C0_N0_A2_Model, test_x, test_y)

    # (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # test_model(MyResNet, AEModel, A2_Model, test_x, test_y)
    # test_model(MyResNet, AEModel, B2_Model, test_x, test_y)
    # test_model(MyResNet, AEModel, N0_A2_Model, test_x, test_y)
    # test_model(MyResNet, AEModel, C0_A2_Model, test_x, test_y)
    # test_model(MyResNet, AEModel, C0_N0_A2_Model, test_x, test_y)

    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    test_model(MyResNet, IdentityAEModel, A2_Model, test_x, test_y)
    
    test_model(MyResNet, DunetModel, C0_A2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, C2_A2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, A2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, B2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, C0_B2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, A2_B2_C2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, C1_A1_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, C1_A2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, C2_A2_P2_Model, test_x, test_y)
    test_model(MyResNet, DunetModel, P2_Model, test_x, test_y)
    
def main():
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    sess = tf.Session()
    
    cnn = MyResNet()
    # cnn = MyResNet56()
    # cnn = MyResNet110()
    # cnn = CNNModel()
    # ae = AEModel(cnn)
    # ae = CifarAEModel(cnn)
    # ae = TestCifarAEModel(cnn)
    # ae = DunetModel(cnn)
    ae = IdentityAEModel(cnn)
    
    adv = A2_Model(cnn, ae)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    # cnn.train_CNN(sess, train_x, train_y)
    # cnn.train_CNN_keras(sess, train_x, train_y, test_x, test_y)
    # cnn.save_weights(sess, 'saved_models/test-resnet-CNN.hdf5')
    cnn.load_weights(sess, 'saved_models/resnet-CNN.hdf5')
    
    # ae.train_AE(sess, train_x, train_y)
    # ae.save_weights(sess, 'saved_models/test-resnet-ae-AE.hdf5')
    # ae.load_weights(sess, 'saved_models/resnet-dunet-AE.hdf5')
    # test
    # ae.test_AE(sess, test_x, test_y)
    
    adv.train_Adv(sess, train_x, train_y)

if __name__ == '__main__':
    tf.random.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    main_train()
    main_test()
    # main()

def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    model, sess = load_model(CNNModel, AEModel, A2_Model)
    model, sess = load_model(CNNModel, AEModel, B2_Model)
    
    model, sess = load_model(CNNModel, AEModel, B2_Model, load_adv=False)
    model.test_all(sess, test_x, test_y, attacks=['FGSM', 'PGD'],
                   filename='out.pdf')
    
    
def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    cnn = CNNModel()
    ae = AEModel(cnn)
    adv = AdvAEModel(cnn, ae)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    adv.train_Adv(sess, train_x, train_y)
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
