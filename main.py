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
                  dataset_name=''):
    if not dataset_name:
        raise Exception('Need to set dataset_name parameter.')
    cnn_prefix = cnn_cls.NAME()
    ae_prefix = ae_cls.NAME()
    advae_prefix = advae_cls.NAME()

    pCNN = os.path.join(saved_folder, '{}-{}-CNN.hdf5'.format(dataset_name, cnn_prefix))
    pAdvAE = os.path.join(saved_folder, '{}-{}-{}-{}-AdvAE.hdf5'.format(dataset_name, cnn_prefix, ae_prefix, advae_prefix))
    
    plot_prefix = '{}-{}-{}-{}'.format(dataset_name, cnn_prefix, ae_prefix, advae_prefix)
    return pCNN, pAdvAE, plot_prefix

def load_model(cnn_cls, ae_cls, advae_cls,
               saved_folder='saved_models',
               dataset_name=''):
    """If load_adv = False, try to load ae instead.

    cnnorig: the transfer model for CNN to be loaded.
    """
    pCNN, pAdvAE, _ = compute_names(cnn_cls, ae_cls, advae_cls,
                                    dataset_name=dataset_name)
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    cnn = cnn_cls()
    ae = ae_cls(cnn)
    adv = advae_cls(cnn, ae)
     
    init = tf.global_variables_initializer()
    sess.run(init)

    print('loading {} ..'.format(pCNN))
    cnn.load_weights(sess, pCNN)
    print('loading {} ..'.format(pAdvAE))
    ae.load_weights(sess, pAdvAE)
    return adv, sess

def train_CNN(cnn_cls, train_x, train_y, saved_folder='saved_models', prefix=''):
    """This function currently is exclusively training transfer CNN models."""
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    pCNN, _, _, _ = compute_names(cnn_cls, cnn_cls, cnn_cls)
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
                dataset_name=''):
    """Train AdvAE.

    """
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    pCNN, pAdvAE, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                              dataset_name=dataset_name)

    # DEBUG early return here to avoid the big overhead of creating the graph
    print('====== Denoising training for {} ..'.format(pAdvAE))
    if os.path.exists(pCNN) and os.path.exists(pAdvAE):
        print('Already trained {}'.format(pAdvAE))
        return
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        cnn = cnn_cls()

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

        if not os.path.exists(pAdvAE):
            ae = ae_cls(cnn)
            adv = advae_cls(cnn, ae)

            init = tf.global_variables_initializer()
            sess.run(init)

            cnn.load_weights(sess, pCNN)
            print('Trainng AdvAE ..')
            adv.train_Adv(sess, train_x, train_y)
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
               dataset_name='',
               force=False):
    _, _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                         dataset_name=dataset_name)
    save_prefix = 'test-result-{}'.format(plot_prefix)
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
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
def test_model_transfer(cnn_cls, ae_cls, advae_cls, test_x, test_y,
                        dataset_name='', to_cnn_cls=None):
    """Transfer to TO_CNN_CLS."""
    assert to_cnn_cls is not None
    _, _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                         dataset_name=dataset_name)
    save_prefix = 'test-result-{}-TO-{}'.format(plot_prefix, to_cnn_cls.NAME())
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
        print('loading TO_CNN weights ..')
        pCNN, pAdvAE, _ = compute_names(to_cnn_cls, ae_cls, advae_cls,
                                        dataset_name=dataset_name)
        model.cnn_model.load_weights(pCNN)
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

    for m in [A2_Model, B2_Model, C0_A2_Model]:
        train_model(MNISTModel, AEModel, m, train_x, train_y, dataset_name='MNIST')
    train_model(MNISTModel, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='MNIST')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='MNIST')

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        train_model(MNISTModel, AEModel, m, train_x, train_y, dataset_name='Fashion')
    train_model(MNISTModel, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='Fashion')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='Fashion')
    
    ##############################
    ## CIFAR10
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # adv training baseline
    for m in [C0_A2_Model, C0_B2_Model]:
        train_model(MyResNet29, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
        train_model(MyWideResNet, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
        train_model(MyResNet56, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
    train_model(MyResNet29, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='CIFAR10')
    
def main_test():

    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    # test_model(MNISTModel, AEModel, A2_Model, test_x, test_y, force=True)
    # adv training baseline
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        test_model(MNISTModel, AEModel, m, test_x, test_y, dataset_name='MNIST')
    test_model(MNISTModel, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='MNIST')
    # test whether the model works for other models
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model(MNISTModel, AEModel, A2_Model,
                   test_x, test_y,
                   to_cnn_cls=m,
                   dataset_name='MNIST')

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    # adv training baseline
    for m in [A2_Model, B2_Model, C0_A2_Model]:
        test_model(MNISTModel, AEModel, m, test_x, test_y, dataset_name='Fashion')
    test_model(MNISTModel, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='Fashion')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model(MNISTModel, AEModel, A2_Model,
                   test_x, test_y,
                   to_cnn_cls=m,
                   dataset_name='Fashion')
    
    ##############################
    ## CIFAR10
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    test_model(MyResNet, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='CIFAR10')

    for m in [C0_A2_Model, C0_B2_Model]:
        test_model(MyResNet29, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
        test_model(MyWideResNet, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
        test_model(MyResNet56, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
    for m in [MyResNet56, MyWideResNet]:
        test_model(MyResNet29, DunetModel, C0_A2_Model,
                   test_x, test_y,
                   to_cnn_cls=m,
                   dataset_name='CIFAR10')

def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    sess = tf.Session()
    cnn = MNISTModel()
    cnn_a = DefenseGAN_a()
    cnn_b = DefenseGAN_b()
    cnn_c = DefenseGAN_c()

    
    # this cnn is only used to provide shape
    ae = AEModel(cnn)

    advae = A2_Model(cnn, ae)
    advae_a = A2_Model(cnn_a, ae, inputs=advae.x, targets=advae.y)
    advae_b = A2_Model(cnn_b, ae, inputs=advae.x, targets=advae.y)
    advae_c = A2_Model(cnn_c, ae, inputs=advae.x, targets=advae.y)

    init = tf.global_variables_initializer()
    sess.run(init)
    
    cnn.load_weights('saved_models/MNIST-mnistcnn-CNN.hdf5')
    cnn_a.load_weights('saved_models/MNIST-DefenseGAN_a-CNN.hdf5')
    cnn_b.load_weights('saved_models/MNIST-DefenseGAN_b-CNN.hdf5')
    cnn_c.load_weights('saved_models/MNIST-DefenseGAN_c-CNN.hdf5')
    
    # ensemble training all of them
    ensemble_training(sess, [advae, advae_a, advae_b, advae_c], train_x, train_y)

    # TODO I'll test the ensemble on these trained models as well as
    # the untrained models. And compare the results with directly
    # transfer.
    
    
def ensemble_training(sess, advae_models, train_x, train_y):
    """Adv training."""
    (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)

    outputs = keras.layers.Activation('softmax')(self.FC(self.CNN(self.AE(self.x))))
    model = keras.models.Model(self.x, outputs)

    all_losses = []
    all_advacc = []
    all_acc = []
    all_cnnacc = []
    all_obliacc = []
    for advae in advae_models:
        advae.CNN.trainable = False
        advae.FC.trainable = False
        losses.append(advae.adv_loss)
        def advacc(ytrue, ypred): return advae.adv_accuracy
        all_advacc.append(advacc)
        def acc(ytrue, ypred): return advae.accuracy
        all_acc.append(acc)
        def cnnacc(ytrue, ypred): return advae.CNN_accuracy
        all_cnnacc.append(cnnacc)
        def obliacc(ytrue, ypred): return advae.obli_accuracy
        all_obliacc.append(obliacc)

    model = keras.models.Model(inputs=advae_models[0].x,
                               outputs=[m.adv_logits for m in advae_models])
        
    def myloss(ytrue, ypred):
        return tf.reduce_sum(all_losses)

    with sess.as_default():
        callbacks = [get_lr_scheduler(),
                     get_lr_reducer(patience=5),
                     get_es(patience=10),
                     get_mc('best_model.hdf5')]
        model.compile(loss=myloss,
                      # metrics=[cnnacc, acc, obliacc, advacc],
                      metrics=all_advacc,
                      optimizer=keras.optimizers.Adam(lr=1e-3),
                      target_tensors=self.y)
        model.fit(train_x, train_y,
                  # DEBUG can I use validation_split?
                  # validation_split=0.1,
                  validation_data=(val_x, val_y),
                  epochs=100,
                  callbacks=callbacks)
        model.load_weights('best_model.ckpt')

    
    
def main():
    # (train_x, train_y), (test_x, test_y) = load_mnist_data()
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    sess = tf.Session()
    
    # cnn = MyResNet29()
    # cnn = MyResNet56()
    # cnn = MyResNet110()
    cnn = MyWideResNet()
    # cnn = MNISTModel()
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

