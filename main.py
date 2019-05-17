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
from models import *
from defensegan_models import *
from attacks import *

def compute_names(cnn_cls, ae_cls, advae_cls,
                  saved_folder='saved_models',
                  cnnprefix='', aeprefix='', advprefix='',
                  prefix='',
                  # TODO refactor prefix to dataset_prefix
                  cnnorig='',
                  prefixorig=''):
    if not cnnprefix:
        cnnprefix = cnn_cls.NAME()
    if not advprefix:
        advprefix = advae_cls.NAME()
    if not aeprefix:
        aeprefix = ae_cls.NAME()

    if prefix:
        prefix += '-'
    if prefixorig:
        prefixorig += '-'
    pCNN = os.path.join(saved_folder, '{}{}-CNN.hdf5'.format(prefix, cnnprefix))
    
    if cnnorig:
        # transfer mode. This should only be used when testing
        pAE = os.path.join(saved_folder, '{}{}-{}-AE.hdf5'.format(prefixorig, cnnorig, aeprefix))
        pAdvAE = os.path.join(saved_folder, '{}{}-{}-{}-AdvAE.hdf5'.format(prefixorig, cnnorig, aeprefix, advprefix))
        plot_prefix='{}{}-TransTo-{}-{}-{}'.format(prefixorig, cnnorig, cnnprefix, aeprefix, advprefix)
    else:
        pAE = os.path.join(saved_folder, '{}{}-{}-AE.hdf5'.format(prefix, cnnprefix, aeprefix))
        pAdvAE = os.path.join(saved_folder, '{}{}-{}-{}-AdvAE.hdf5'.format(prefix, cnnprefix, aeprefix, advprefix))
        plot_prefix='{}{}-{}-{}'.format(prefix, cnnprefix, aeprefix, advprefix)
    return pCNN, pAE, pAdvAE, plot_prefix

def load_model(cnn_cls, ae_cls, advae_cls,
               saved_folder='saved_models',
               cnnprefix='', aeprefix='', advprefix='',
               prefix='',
               cnnorig='',
               prefixorig='',
               load_adv=True):
    """If load_adv = False, try to load ae instead.

    cnnorig: the transfer model for CNN to be loaded.
    """
    pCNN, pAE, pAdvAE, _ = compute_names(cnn_cls, ae_cls, advae_cls,
                                         cnnprefix=cnnprefix, aeprefix=aeprefix, advprefix=advprefix,
                                         cnnorig=cnnorig,
                                         prefix=prefix,
                                         prefixorig=prefixorig)
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    cnn = cnn_cls()
    ae = ae_cls(cnn)
    adv = advae_cls(cnn, ae)
     
   
    init = tf.global_variables_initializer()
    sess.run(init)

    print('loading {} ..'.format(pCNN))
    cnn.load_weights(sess, pCNN)
    if load_adv:
        print('loading {} ..'.format(pAdvAE))
        ae.load_weights(sess, pAdvAE)
    else:
        print('loading {} ..'.format(pAE))
        ae.load_weights(sess, pAE)
    return adv, sess

def train_CNN(cnn_cls, train_x, train_y, saved_folder='saved_models', prefix=''):
    """This function currently is exclusively training transfer CNN models."""
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    pCNN, _, _, _ = compute_names(cnn_cls, cnn_cls, cnn_cls, prefix=prefix)
    if os.path.exists(pCNN):
        print('Already trained {}'.format(pCNN))
    else:
        tf.reset_default_graph()
        with tf.Session() as sess:
            cnn = cnn_cls()
            init = tf.global_variables_initializer()
            sess.run(init)

            print('Trianing CNN ..')
            cnn.train_CNN(sess, train_x, train_y, patience=5)
            print('Saving model to {} ..'.format(pCNN))
            cnn.save_weights(sess, pCNN)
    
def train_model(cnn_cls, ae_cls, advae_cls,
                train_x, train_y,
                saved_folder='saved_models',
                cnnprefix='', aeprefix='', advprefix='',
                prefix=''):
    """Train AdvAE.

    """
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    pCNN, pAE, pAdvAE, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                                   cnnprefix=cnnprefix, aeprefix=aeprefix, advprefix=advprefix,
                                                   prefix=prefix)

    # DEBUG early return here to avoid the big overhead of creating the graph
    print('====== Denoising training for {} ..'.format(advprefix))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        cnn = cnn_cls()
        # ae = ae_cls(cnn)
        # adv = advae_cls(cnn, ae)

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
        # if not os.path.exists(pAE):
        #     print('Training AE ..')
        #     ae.train_AE(sess, train_x, train_y)
        #     print('Saving model to {} ..'.format(pAE))
        #     ae.save_weights(sess, pAE)
        # else:
        #     print('Trained, directly loading {} ..'.format(pAE))
        #     ae.load_weights(sess, pAE)
        
        # 3. train denoiser using adv training, with high level feature guidance
        # acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
        # print('Model accuracy on clean data: {}'.format(acc))
        # model.test_all(sess, val_x, val_y, run_CW=False)

        # overwrite controls only the AdvAE weights. CNN and AE
        # weights do not need to be retrained because I'm not
        # experimenting with changing training or loss for that.
        if not os.path.exists(pAdvAE):
            ae = ae_cls(cnn)
            adv = advae_cls(cnn, ae)

            init = tf.global_variables_initializer()
            sess.run(init)

            cnn.load_weights(sess, pCNN)
            print('Trainng AdvAE ..')
            adv.train_Adv(sess, train_x, train_y, plot_prefix=plot_prefix)
            print('saving to {} ..'.format(pAdvAE))
            ae.save_weights(sess, pAdvAE)
        else:
            print('Already trained {}'.format(pAdvAE))
    
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
               cnnprefix='', aeprefix='', advprefix='',
               prefix='',
               cnnorig='',
               prefixorig='',
               force=False):


    # FIXME keep consistent with advae.name()
    model_name = '{}-{}-{}'.format(cnn_cls.NAME(), ae_cls.NAME(), advae_cls.NAME())
    _, _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                         cnnprefix=cnnprefix, aeprefix=aeprefix, advprefix=advprefix,
                                         cnnorig=cnnorig,
                                         prefix=prefix)
    save_prefix = 'test-result-{}'.format(plot_prefix)
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 cnnprefix=cnnprefix, aeprefix=aeprefix, advprefix=advprefix,
                                 prefix=prefix, cnnorig=cnnorig, prefixorig=prefixorig)
        # model.test_all(sess, test_x, test_y, attacks=[])
        # model.test_all(sess, test_x, test_y, attacks=[], num_sample=1000)
        print('testing {} ..'.format(plot_prefix))
        model.test_all(sess, test_x, test_y,
                       attacks=['CW', 'FGSM', 'PGD'],
                       # attacks=['CW', 'FGSM', 'PGD', 'JSMA'],
                       save_prefix=save_prefix)
        # model.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'])
        # model.test_all(sess, test_x, test_y, attacks=['CW', 'JSMA', 'FGSM', 'PGD'])
    else:
        print('Already tested, see {}'.format(filename))

def main_train():
    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    train_model(CNNModel, AEModel, A2_Model, train_x, train_y)
    
    # sess = tf.Session()
    # cnn = CNNModel()
    # ae = AEModel(cnn)
    # adv = A2_Model(cnn, ae)

    # m2 = keras.models.Model(adv.x, ae.AE(adv.x))
    # m2.updates
    
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # cnn.load_weights(sess, 'saved_models/mnistcnn-CNN.hdf5')
    # adv.train_Adv(sess, train_x, train_y)
    # adv.ae_model.AE.summary()
    # adv.ae_model.AE1.summary()
    # train_x.shape
    
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        train_model(CNNModel, AEModel, m, train_x, train_y)
    train_model(CNNModel, IdentityAEModel, A2_Model, train_x, train_y)
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, prefix='DGANMNIST')

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        train_model(FashionCNNModel, AEModel, m, train_x, train_y)
    train_model(FashionCNNModel, IdentityAEModel, A2_Model, train_x, train_y)
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, prefix='DGANFashion')
    
    ##############################
    ## CIFAR10
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # adv training baseline
    for m in [C0_A2_Model, C0_B2_Model]:
        train_model(MyResNet, DunetModel, m, train_x, train_y)
        train_model(MyWideResNet, DunetModel, m, train_x, train_y)
        train_model(MyResNet56, DunetModel, m, train_x, train_y)
    train_model(MyResNet, IdentityAEModel, A2_Model, train_x, train_y)
    
def main_test():

    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    # test_model(CNNModel, AEModel, A2_Model, test_x, test_y, force=True)
    # adv training baseline
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        test_model(CNNModel, AEModel, m, test_x, test_y)
    test_model(CNNModel, IdentityAEModel, A2_Model, test_x, test_y)
    # test whether the model works for other models
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model(m, AEModel, A2_Model,
                   test_x, test_y,
                   prefix='DGANMNIST',
                   prefixorig='',
                   cnnorig=CNNModel.NAME())

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    # adv training baseline
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        test_model(FashionCNNModel, AEModel, m, test_x, test_y)
    test_model(FashionCNNModel, IdentityAEModel, A2_Model, test_x, test_y)
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model(m, AEModel, A2_Model,
                   test_x, test_y,
                   prefix='DGANFashion',
                   prefixorig='',
                   cnnorig=FashionCNNModel.NAME())
    
    ##############################
    ## CIFAR10
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    test_model(MyResNet, IdentityAEModel, A2_Model, test_x, test_y)

    for m in [C0_A2_Model, C0_B2_Model]:
        test_model(MyResNet, DunetModel, m, test_x, test_y)
        test_model(MyWideResNet, DunetModel, m, test_x, test_y)
        test_model(MyResNet56, DunetModel, m, test_x, test_y)
    
def main():
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    sess = tf.Session()
    
    cnn = MyResNet()
    # cnn = MyResNet56()
    # cnn = MyResNet110()
    # cnn = MyWideResNet()
    # cnn = CNNModel()
    # ae = AEModel(cnn)
    # ae = CifarAEModel(cnn)
    # ae = TestCifarAEModel(cnn)
    # ae = DunetModel(cnn)
    # ae = IdentityAEModel(cnn)
    
    # adv = A2_Model(cnn, ae)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    # cnn.train_CNN(sess, train_x, train_y)
    # cnn.train_CNN_keras(sess, train_x, train_y, test_x, test_y)
    # cnn.save_weights(sess, 'saved_models/test-resnet-CNN.hdf5')
    # cnn.load_weights(sess, 'saved_models/resnet-CNN.hdf5')

    # cnn.train_CNN_keras(sess, train_x, train_y)
    
    cnn.train_CNN(sess, train_x, train_y)
    # cnn.save_weights()
    
    # ae.train_AE(sess, train_x, train_y)
    # ae.save_weights(sess, 'saved_models/test-resnet-ae-AE.hdf5')
    # ae.load_weights(sess, 'saved_models/resnet-dunet-AE.hdf5')
    # test
    # ae.test_AE(sess, test_x, test_y)
    
    # adv.train_Adv(sess, train_x, train_y)

if __name__ == '__main__':
    tf.random.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    main_train()
    main_test()
    # main()
def parse_single_whitebox(json_file):
    with open(json_file, 'r') as fp:
        j = json.load(fp)
        return [j[0]['AE clean'],
                j[3]['PGD']['whiteadv_rec'],
                j[2]['FGSM']['whiteadv_rec'],
                j[1]['CW']['whiteadv_rec']]
    

def parse_result(advae_json, advtrain_json, hgd_json):
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)

    res = {}
    res['nodef'] = [advae[0]['CNN clean'],
                    advae[3]['PGD']['obliadv'],
                    advae[2]['FGSM']['obliadv'],
                    advae[1]['CW']['obliadv']]
    
    res['advae obli'] = [-1,
                         advae[3]['PGD']['obliadv_rec'],
                         advae[2]['FGSM']['obliadv_rec'],
                         advae[1]['CW']['obliadv_rec']]
    
    res['advae whitebox'] = parse_single_whitebox(advae_json)
    res['hgd whitebox'] = parse_single_whitebox(hgd_json)
    res['advtrain whitebox'] = parse_single_whitebox(advtrain_json)
    return res
def __test():
    # extracting experiment results

    # MNIST
    res = parse_result('images/test-result-mnistcnn-ae-A2.json',
                       'images/test-result-mnistcnn-identityAE-A2.json',
                       'images/test-result-mnistcnn-ae-B2.json')
    # fashion
    res = parse_result('images/test-result-fashion-ae-A2.json',
                       'images/test-result-fashion-identityAE-A2.json',
                       'images/test-result-fashion-ae-B2.json')
    # cifar
    res = parse_result('images/test-result-resnet-dunet-C0_A2.json',
                       'images/test-result-resnet-identityAE-A2.json',
                       'images/test-result-resnet-dunet-C0_B2.json')
    # transfer models
    res = {}
    for defgan_var in ['a', 'b', 'c', 'd', 'e', 'f']:
        # fname = 'images/test-result-{}-TransTo-DefenseGAN_{}-ae-A2.json'.format('mnist', defgan_var)
        fname = 'images/test-result-{}-TransTo-DefenseGAN_{}-ae-A2.json'.format('fashion', defgan_var)
        res[defgan_var] = parse_single_whitebox(fname)
        
    for i in range(4):
        for k in res:
            print('{:.3f}'.format(res[k][i]), end=',')
        print()

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
