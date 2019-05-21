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
from blackbox import test_sub

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
def compute_ensemble_names(cnn_clses, ae_cls, advae_cls,
                           saved_folder='saved_models',
                           dataset_name=''):
    if not dataset_name:
        raise Exception('Need to set dataset_name parameter.')
    cnn_prefixes = [cnn_cls.NAME() for cnn_cls in cnn_clses]
    ae_prefix = ae_cls.NAME()
    advae_prefix = advae_cls.NAME()

    pCNNs = [os.path.join(saved_folder, '{}-{}-CNN.hdf5'.format(dataset_name, cnn_prefix)) for cnn_prefix in cnn_prefixes]
    pAdvAE = os.path.join(saved_folder, '{}-{}-{}-{}-AdvAE.hdf5'.format(dataset_name, cnn_prefixes, ae_prefix, advae_prefix))
    
    plot_prefix = '{}-[{}]-{}-{}'.format(dataset_name, '-'.join(cnn_prefixes), ae_prefix, advae_prefix)
    return pCNNs, pAdvAE, plot_prefix

def load_model(cnn_cls, ae_cls, advae_cls,
               saved_folder='saved_models',
               dataset_name='',
               load_advae=True):
    """If load_adv = False, try to load ae instead.

    cnnorig: the transfer model for CNN to be loaded.
    """
    pCNN, pAdvAE, _ = compute_names(cnn_cls, ae_cls, advae_cls,
                                    dataset_name=dataset_name)
    sess = create_tf_session()
    
    cnn = cnn_cls()
    ae = ae_cls(cnn)
    adv = advae_cls(cnn, ae)

    tf_init_uninitialized(sess)

    print('loading {} ..'.format(pCNN))
    cnn.load_weights(sess, pCNN)
    if load_advae:
        print('loading {} ..'.format(pAdvAE))
        ae.load_weights(sess, pAdvAE)
    return adv, sess
def load_model_transfer(cnn_cls, ae_cls, advae_cls,
                        to_cnn_cls=None,
                        saved_folder='saved_models',
                        dataset_name=''):
    """If load_adv = False, try to load ae instead.

    cnnorig: the transfer model for CNN to be loaded.
    """
    _, pAdvAE, _ = compute_names(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
    pCNN, _, _ = compute_names(to_cnn_cls, ae_cls, advae_cls,
                               dataset_name=dataset_name)
    sess = create_tf_session()
   
    cnn = to_cnn_cls()
    ae = ae_cls(cnn)
    adv = advae_cls(cnn, ae)

    tf_init_uninitialized(sess)

    print('loading {} ..'.format(pCNN))
    cnn.load_weights(sess, pCNN)
    print('loading {} ..'.format(pAdvAE))
    ae.load_weights(sess, pAdvAE)
    return adv, sess

def train_CNN(cnn_cls, train_x, train_y, saved_folder='saved_models', dataset_name=''):
    """This function currently is exclusively training transfer CNN models."""
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    pCNN, _, _ = compute_names(cnn_cls, cnn_cls, cnn_cls, dataset_name=dataset_name)
    if os.path.exists(pCNN):
        print('Already trained {}'.format(pCNN))
    else:
        with create_tf_session() as sess:
            cnn = cnn_cls()

            tf_init_uninitialized(sess)

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
    
    with create_tf_session() as sess:
        cnn = cnn_cls()
        tf_init_uninitialized(sess)

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
            tf_init_uninitialized(sess)

            print('Trainng AdvAE ..')
            adv.train_Adv(sess, train_x, train_y)
            print('saving to {} ..'.format(pAdvAE))
            ae.save_weights(sess, pAdvAE)
        else:
            print('Already trained {}'.format(pAdvAE))
    
def __test():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    sess = create_tf_session()
    
    cnn = MyResNet()
    cnn = MyResNet56()
    cnn = MyResNet110()
    cnn = CNNModel()
    ae = AEModel(cnn)
    ae = CifarAEModel(cnn)
    ae = TestCifarAEModel(cnn)
    ae = DunetModel(cnn)
    adv = A2_Model(cnn, ae)

    tf_init_uninitialized(sess)
    
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
    _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                      dataset_name=dataset_name)
    save_prefix = 'test-result-{}'.format(plot_prefix)
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
        print('testing {} ..'.format(plot_prefix))
        model.test_all(sess, test_x, test_y,
                       attacks=['CW', 'FGSM', 'PGD'],
                       num_sample=1000,
                       save_prefix=save_prefix)
    else:
        print('Already tested, see {}'.format(filename))
def test_model_transfer(cnn_cls, ae_cls, advae_cls, test_x, test_y,
                        dataset_name='', to_cnn_cls=None,
                        saved_folder='saved_models', force=False):
    """Transfer to TO_CNN_CLS."""
    assert to_cnn_cls is not None
    _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                      dataset_name=dataset_name)
    save_prefix = 'test-result-{}-TO-{}'.format(plot_prefix, to_cnn_cls.NAME())
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    print('Testing transfer {} ..'.format(save_prefix))
    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model_transfer(cnn_cls, ae_cls, advae_cls,
                                          to_cnn_cls=to_cnn_cls,
                                          dataset_name=dataset_name)
        
        print('testing {} ..'.format(plot_prefix))
        model.test_all(sess, test_x, test_y,
                       attacks=['CW', 'FGSM', 'PGD'],
                       num_sample=1000,
                       save_prefix=save_prefix)
    else:
        print('Already tested, see {}'.format(filename))
    

def test_model_bbox(cnn_cls, ae_cls, advae_cls, test_x, test_y,
                    dataset_name='', to_cnn_cls=None,
                    saved_folder='saved_models', force=False,
                    save_prefix=''):
    """Transfer to TO_CNN_CLS."""
    assert to_cnn_cls is not None
    _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                      dataset_name=dataset_name)
    if not save_prefix:
        save_prefix = 'test-result-{}-BBOX-{}'.format(plot_prefix, to_cnn_cls.NAME())
    filename = 'images/{}.json'.format(save_prefix)
    
    print('Testing transfer {} ..'.format(save_prefix))
    if not os.path.exists(filename) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
        sub = to_cnn_cls()
        
        # after getting the sub model, run attack on sub model
        res = test_sub(sess, model, sub, test_x, test_y)
        
        datafile = 'images/{}.json'.format(save_prefix)
        print('Result:')
        print(res)
        with open(datafile, 'w') as fp:
            json.dump(res, fp, indent=4)
        print('Done. Saved to {}'.format(datafile))
    else:
        print('Already tested, see {}'.format(filename))


def train_ensemble(cnn_clses, ae_cls, advae_cls,
                   train_x, train_y,
                   saved_folder='saved_models',
                   dataset_name=''):
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    # get cnn weights names
    pCNNs, pAdvAE, plot_prefix  = compute_ensemble_names(cnn_clses, ae_cls, advae_cls, dataset_name=dataset_name)

    # DEBUG early return here to avoid the big overhead of creating the graph
    print('====== Ensemble training for {} ..'.format(plot_prefix))
    for pCNN in pCNNs:
        # TODO I'm asserting pCNN exists here
        assert os.path.exists(pCNN)
    if os.path.exists(pAdvAE):
        print('Already trained {}'.format(pAdvAE))
        return
    
    with create_tf_session() as sess:
        print('creating cnns ..')
        cnns = [cnn_cls() for cnn_cls in cnn_clses]
        print('creating AE ..')
        ae = ae_cls(cnns[0])
        print('creating advae ..')
        adv0 = advae_cls(cnns[0], ae)
        advs = [adv0] + [advae_cls(cnn, ae, inputs=adv0.x, targets=adv0.y) for cnn in cnns[1:]]

        tf_init_uninitialized(sess)
        
        print('loading cnn weights ..')
        for cnn, pCNN in zip(cnns, pCNNs):
            cnn.load_weights(sess, pCNN)

        print('Trainng AdvAE ..')
        ensemble_training_impl(sess, advs, train_x, train_y)
        print('saving to {} ..'.format(pAdvAE))
        ae.save_weights(sess, pAdvAE)
    

def test_ensemble(cnn_clses, ae_cls, advae_cls,
                  test_x, test_y,
                  dataset_name='',
                  to_cnn_cls=None,
                  saved_folder='saved_models', force=False):
    assert to_cnn_cls is not None
    _, pAdvAE, plot_prefix = compute_ensemble_names(cnn_clses, ae_cls, advae_cls,
                                               dataset_name=dataset_name)
    save_prefix = 'test-result-{}-ENSEMBLE-{}'.format(plot_prefix, to_cnn_cls.NAME())
    
    filename = 'images/{}.json'.format(save_prefix)

    print('Testing transfer {} ..'.format(save_prefix))
    if not os.path.exists(filename) or force:
        print('loading model ..')
        model, sess = load_model(to_cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name,
                                 load_advae=False)
        model.ae_model.load_weights(sess, pAdvAE)
        
        print('testing {} ..'.format(plot_prefix))
        model.test_all(sess, test_x, test_y,
                       attacks=['CW', 'FGSM', 'PGD'],
                       num_sample=1000,
                       save_prefix=save_prefix)
    else:
        print('Already tested, see {}'.format(filename))
    
def ensemble_training_impl(sess, advae_models, train_x, train_y):
    """Adv training."""
    print('creating loss and metrics ..')
    all_loss = []
    all_advacc = []
    all_acc = []
    all_cnnacc = []
    all_obliacc = []
    for advae in advae_models:
        advae.CNN.trainable = False
        advae.FC.trainable = False

    metrics = [f for m in advae_models for f in m.metric_funcs()]

    print('creating keras model ..')
    model = keras.models.Model(inputs=advae_models[0].x,
                               # outputs=[m.logits for m in advae_models],
                               outputs=advae_models[0].logits
    )

    print('creating loss ..')
    def myloss(ytrue, ypred):
        return tf.reduce_sum([m.adv_loss for m in advae_models])

    with sess.as_default():
        # DEBUG This is painfully slow, so setting a lower patience
        callbacks = [get_lr_reducer(),
                     get_es()]
        print('compiling model ..')
        model.compile(loss=myloss,
                      # metrics includes 5x overhead
                      # metrics=metrics,
                      optimizer=keras.optimizers.Adam(lr=1e-3),
                      target_tensors=[advae_models[0].y])
        print('fitting model ..')
        model.fit(train_x, train_y,
                  validation_split=0.1,
                  epochs=100,
                  callbacks=callbacks)

    

def main_train():
    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        train_model(MNISTModel, AEModel, m, train_x, train_y, dataset_name='MNIST')
    train_model(MNISTModel, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='MNIST')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='MNIST')

    train_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
                   AEModel, A2_Model, train_x, train_y, dataset_name='MNIST')
    train_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
                   AEModel, A2_Model, train_x, train_y, dataset_name='MNIST')
    train_ensemble([DefenseGAN_c, DefenseGAN_d],
                   AEModel, A2_Model, train_x, train_y, dataset_name='MNIST')

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        train_model(MNISTModel, AEModel, m, train_x, train_y, dataset_name='Fashion')
    train_model(MNISTModel, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='Fashion')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='Fashion')
        
    # train_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
    #                AEModel, A2_Model, train_x, train_y, dataset_name='Fashion')
    # train_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
    #                AEModel, A2_Model, train_x, train_y, dataset_name='Fashion')
    # train_ensemble([DefenseGAN_c, DefenseGAN_d],
    #                AEModel, A2_Model, train_x, train_y, dataset_name='Fashion')
    
    ##############################
    ## CIFAR10
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # adv training baseline
    for m in [A2_Model, C0_A2_Model, C0_B2_Model]:
        train_model(MyResNet29, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
        train_model(MyWideResNet2, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
        train_model(MyResNet56, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
        train_model(MyDenseNet, DunetModel, m, train_x, train_y, dataset_name='CIFAR10')
    train_model(MyResNet29, IdentityAEModel, A2_Model, train_x, train_y, dataset_name='CIFAR10')
    train_ensemble([MyResNet29, MyWideResNet2],
                   DunetModel, C0_A2_Model, train_x, train_y, dataset_name='CIFAR10')
    train_ensemble([MyResNet29, MyWideResNet2, MyDenseNet],
                   DunetModel, C0_A2_Model, train_x, train_y, dataset_name='CIFAR10')
    
def main_test():
    ##############################
    # Experimental
    
    # _, (test_x, test_y) = load_mnist_data()

    # test_model(MNISTModel, AEModel, A2_Model, test_x, test_y, dataset_name='MNIST', force=True)
    # test_model_bbox(MNISTModel, AEModel, A2_Model, test_x, test_y,
    #                 to_cnn_cls=MNISTModel,
    #                 dataset_name='MNIST',
    #                 save_prefix='out',
    #                 force=True)
    # test_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
    #               AEModel, A2_Model,
    #               test_x, test_y,
    #               to_cnn_cls=MNISTModel,
    #               dataset_name='MNIST', force=True)

    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    # adv training baseline
    test_model(MNISTModel, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='MNIST')
    # white box testing
    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        test_model(MNISTModel, AEModel, m, test_x, test_y, dataset_name='MNIST')
    # blackbox testing
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c]:
        test_model_bbox(MNISTModel, AEModel, A2_Model, test_x, test_y,
                        to_cnn_cls=m,
                        dataset_name='MNIST')
    # model transfer testing
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model_transfer(MNISTModel, AEModel, A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='MNIST')
    # ensemble testing
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d]:
        test_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
                      AEModel, A2_Model,
                      test_x, test_y,
                      to_cnn_cls=m,
                      dataset_name='MNIST')

    ##############################
    ## Fashion MNIST
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    # adv training baseline
    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        test_model(MNISTModel, AEModel, m, test_x, test_y, dataset_name='Fashion')
    test_model(MNISTModel, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='Fashion')
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model_transfer(MNISTModel, AEModel, A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='Fashion')
    
    ##############################
    ## CIFAR10
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    test_model(MyResNet29, IdentityAEModel, A2_Model, test_x, test_y, dataset_name='CIFAR10')
    for m in [C0_A2_Model, C0_B2_Model]:
        test_model(MyResNet29, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
        test_model(MyWideResNet2, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
        test_model(MyResNet56, DunetModel, m, test_x, test_y, dataset_name='CIFAR10')
    for m in [MyResNet56, MyWideResNet2]:
        test_model_transfer(MyResNet29, DunetModel, C0_A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='CIFAR10')

def main():
    sess = create_tf_session()
    
    # (train_x, train_y), (test_x, test_y) = load_mnist_data()
    # cnn = MNISTModel()
    
    # ae = AEModel(cnn)
    # ae = IdentityAEModel(cnn)


    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # cnn = MyResNet29()
    # cnn = MyResNet56()
    # cnn = MyResNet110()
    # cnn = MyWideResNet2()
    cnn = MyDenseNet()

    # ae = DunetModel(cnn)
    
    # adv = A2_Model(cnn, ae)
    tf_init_uninitialized(sess)

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
    res = parse_result('images/test-result-MNIST-mnistcnn-ae-A2.json',
                       'images/test-result-MNIST-mnistcnn-identityAE-A2.json',
                       'images/test-result-MNIST-mnistcnn-ae-B2.json')
    # fashion
    res = parse_result('images/test-result-Fashion-mnistcnn-ae-A2.json',
                       'images/test-result-Fashion-mnistcnn-identityAE-A2.json',
                       'images/test-result-Fashion-mnistcnn-ae-B2.json')
    # cifar
    res = parse_result('images/test-result-CIFAR10-resnet29-dunet-C0_A2.json',
                       'images/test-result-CIFAR10-resnet29-identityAE-A2.json',
                       'images/test-result-CIFAR10-resnet29-dunet-C0_B2.json')
    # transfer models
    res = {}
    for defgan_var in ['a', 'b', 'c', 'd', 'e', 'f']:
        # fname = 'images/test-result-{}-TransTo-DefenseGAN_{}-ae-A2.json'.format('mnist', defgan_var)
        # fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('MNIST-mnistcnn-ae-A2', defgan_var)
        # fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('Fashion-mnistcnn-ae-A2', defgan_var)
        fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('CIFAR10-resnet29-dunet-C0_A2', defgan_var)
        res[defgan_var] = parse_single_whitebox(fname)

    # resnet to WRN
    fname = 'images/test-result-{}-TO-{}.json'.format('CIFAR10-resnet29-dunet-C0_A2', 'WRN')
    parse_single_whitebox(fname)
        
    for i in range(4):
        for k in res:
            print('{:.3f}'.format(res[k][i]), end=',')
        print()

