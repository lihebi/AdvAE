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

            print('Training CNN ..')
            cnn.train_CNN(sess, train_x, train_y)
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

    
    # 1. train CNN
    if not os.path.exists(pCNN):
        with create_tf_session() as sess:
            cnn = cnn_cls()
            tf_init_uninitialized(sess)
            print('Training CNN ..')
            cnn.train_CNN(sess, train_x, train_y)
            print('Saving model to {} ..'.format(pCNN))
            cnn.save_weights(sess, pCNN)
    else:
        print('Already trained {} ..'.format(pCNN))
    
    if not os.path.exists(pAdvAE):
        with create_tf_session() as sess:
            # FIXME not the case for It-adv-train baseline
            if 'identity' in ae_cls.NAME():
                print('!!!!!!! ******** Training identityAE, '
                      'setting CNN bath normalization layers trainable')
                cnn = cnn_cls(training=True)
            else:
                cnn = cnn_cls(training=False)
            ae = ae_cls(cnn)
            adv = advae_cls(cnn, ae)
            tf_init_uninitialized(sess)
            
            print('Loading {} ..'.format(pCNN))
            cnn.load_weights(sess, pCNN)
            print('Trainng AdvAE ..')
            adv.train_Adv(sess, train_x, train_y)
            print('saving to {} ..'.format(pAdvAE))
            ae.save_weights(sess, pAdvAE)
    else:
        print('Already trained {}'.format(pAdvAE))

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
def load_dataset(dataset_name):
    if dataset_name == 'MNIST':
        return load_mnist_data()
    elif dataset_name == 'Fashion':
        return load_fashion_mnist_data()
    elif dataset_name == 'CIFAR10':
        return load_cifar10_data()
    else:
        raise Exception()

def run_exp_model(cnn_cls, ae_cls, advae_cls,
                  saved_folder='saved_models',
                  dataset_name='', run_test=True):
    """Both training and testing."""
    # reset seed here
    tf.random.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_name)
    train_model(cnn_cls, ae_cls, advae_cls, train_x, train_y, dataset_name=dataset_name)
    if run_test:
        test_model(cnn_cls, ae_cls, advae_cls, test_x, test_y, dataset_name=dataset_name)

def run_exp_ensemble(cnn_clses, ae_cls, advae_cls,
                     to_cnn_clses=None,
                     saved_folder='saved_models',
                     dataset_name='',
                     run_test=True):
    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_name)
    train_ensemble(cnn_clses, ae_cls, advae_cls, train_x, train_y, dataset_name='MNIST')
    if run_test:
        for m in to_cnn_clses:
            test_ensemble(cnn_clses, ae_cls, advae_cls,
                          test_x, test_y,
                          to_cnn_cls=m,
                          dataset_name='MNIST')

def main_train_cnn():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='MNIST')
        
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        train_CNN(m, train_x, train_y, dataset_name='Fashion')

def main_exp(run_test=True):
    ##############################
    ## MNIST
    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        run_exp_model(MNISTModel, AEModel, m, dataset_name='MNIST', run_test=run_test)
    run_exp_model(MNISTModel, IdentityAEModel, A2_Model, dataset_name='MNIST', run_test=run_test)

    run_exp_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
                     AEModel, A2_Model,
                     to_cnn_clses=[MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
                     dataset_name='MNIST',
                     run_test=run_test)
    run_exp_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
                     AEModel, A2_Model,
                     to_cnn_clses=[MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
                     dataset_name='MNIST',
                     run_test=run_test)
    run_exp_ensemble([DefenseGAN_c, DefenseGAN_d],
                     AEModel, A2_Model,
                     to_cnn_clses=[MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
                     dataset_name='MNIST',
                     run_test=run_test)

    ##############################
    ## Fashion MNIST

    for m in [A2_Model, B2_Model, C0_A2_Model, C0_B2_Model]:
        run_exp_model(MNISTModel, AEModel, m,
                      dataset_name='Fashion', run_test=run_test)
    run_exp_model(MNISTModel, IdentityAEModel, A2_Model,
                  dataset_name='Fashion', run_test=run_test)
    
    # run_exp_ensemble([MNISTModel, DefenseGAN_a, DefenseGAN_b],
    #                  AEModel, A2_Model,
    #                  to_cnn_clses=[MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d],
    #                  dataset_name='Fashion',
    #                  run_test=run_test)
    
    ##############################
    ## CIFAR10
    
    run_exp_model(MyResNet29, IdentityAEModel, A2_Model,
                  dataset_name='CIFAR10', run_test=run_test)

    # I'll probably just use ResNet and some vanila CNN
    for m in [C2_A2_Model, C1_A2_Model, C0_A2_Model, C2_B2_Model]:
        for e in [AEModel, DunetModel]:
            run_exp_model(MyResNet29, e, m,
                          dataset_name='CIFAR10',
                          run_test=run_test)
    run_exp_model(MyResNet29, IdentityAEModel, A2_Model,
                  dataset_name='CIFAR10', run_test=run_test)
    run_exp_model(MyResNet29, IdentityAEModel, C2_A2_Model,
                  dataset_name='CIFAR10', run_test=run_test)

    return
            
    # I have no idea which would work, or wouldn't
    for m in [C2_A2_Model,
              C1_A2_Model,
              C0_A2_Model,
              # C2_B2_Model
    ]:
        for e in [AEModel, DunetModel]:
            run_exp_model(MyResNet29, e, m,
                          dataset_name='CIFAR10',
                          run_test=run_test)
            # DenseNet is so 3X slower than ResNet
            # run_exp_model(MyDenseNetFCN, e, m,
            #               dataset_name='CIFAR10',
            #               run_test=run_test)
            
            # This is not working
            # run_exp_model(MyDenseNet, DunetModel, m,
            #               dataset_name='CIFAR10',
            #               run_test=run_test)
            
            # WRN seems not working for weired reasons, the AE seems
            # not working under it.
            # WRN has even higher overhead. I'm abandening it as well
            run_exp_model(MyWideResNet2, e, m,
                          dataset_name='CIFAR10',
                          run_test=run_test)
    run_exp_model(MyResNet29, IdentityAEModel, C2_A2_Model,
                  dataset_name='CIFAR10', run_test=run_test)

    # These two will be very slow
    # run_exp_ensemble([MyResNet29, MyWideResNet2],
    #                  DunetModel, C0_A2_Model,
    #                  to_cnn_clses=[MyResNet29, MyWideResNet2, MyDenseNetFCN],
    #                  dataset_name='CIFAR10',
    #                  run_test=run_test)
    # run_exp_ensemble([MyResNet29, MyWideResNet2, MyDenseNetFCN],
    #                  DunetModel, C0_A2_Model,
    #                  to_cnn_clses=[MyResNet29, MyWideResNet2, MyDenseNetFCN],
    #                  dataset_name='CIFAR10',
    #                  run_test=run_test)
    
def main_test():
    """Additional tests."""
    ##############################
    ## MNIST
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    # model transfer testing
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model_transfer(MNISTModel, AEModel, A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='MNIST')
    # blackbox testing
    # test_model_bbox(MNISTModel, AEModel, A2_Model, test_x, test_y,
    #                 to_cnn_cls=MNISTModel,
    #                 dataset_name='MNIST', force=True)
    
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c]:
        test_model_bbox(MNISTModel, AEModel, A2_Model, test_x, test_y,
                        to_cnn_cls=m,
                        dataset_name='MNIST')

    ##############################
    ## Fashion MNIST
    # model transfer testing
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist_data()
    for m in [DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d, DefenseGAN_e, DefenseGAN_f]:
        test_model_transfer(MNISTModel, AEModel, A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='Fashion')
    # blackbox testing
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c]:
        test_model_bbox(MNISTModel, AEModel, A2_Model, test_x, test_y,
                        to_cnn_cls=m,
                        dataset_name='Fashion')
    
    ##############################
    ## CIFAR10
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    # model transfer testing
    for m in [MyResNet56, MyWideResNet2, MyDenseNetFCN]:
        test_model_transfer(MyResNet29, DunetModel, C0_A2_Model,
                            test_x, test_y,
                            to_cnn_cls=m,
                            dataset_name='CIFAR10')
    # blackbox testing
    for m in [MyResNet29, MyWideResNet2, MyDenseNetFCN]:
        test_model_bbox(MyResNet29, DunetModel, C0_A2_Model,
                        to_cnn_cls=m,
                        dataset_name='CIFAR10')

def train_them_all():
    """Train both the CNN and the AE. I'm doing it on CIFAR data, as the
MNIST data already showed near 100% accuracy."""
    sess = create_tf_session()
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    cnn = MyResNet29()
    ae = DunetModel(cnn)
    adv = C0_A2_Model(cnn, ae)
    tf_init_uninitialized(sess)
    adv.train_Adv(sess, train_x, train_y, train_all=True)

    print('============================== restore-0517-AdvAE')
    cnn.load_weights(sess, 'back/saved_models-0517/resnet-CNN.hdf5')
    ae.load_weights(sess, 'back/saved_models-0517/resnet-dunet-C0_A2-AdvAE.hdf5')
    adv.test_all(sess, test_x, test_y, attacks=['CW', 'FGSM', 'PGD'], save_prefix='restore-0517-AdvAE')
    pass

    
def mnist_exp():
    # run_exp_model(MNISTModel, AEModel, A2_Model, dataset_name='MNIST', run_test=True)
    # Testing different hyperparameter lambdas
    # this is the same as A2_Model
    run_exp_model(MNISTModel, AEModel, get_lambda_model(0), dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, get_lambda_model(0.2), dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, get_lambda_model(0.5), dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, get_lambda_model(1), dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, get_lambda_model(1.5), dataset_name='MNIST', run_test=True)
    # run_exp_model(MNISTModel, AEModel, get_lambda_model(2), dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, get_lambda_model(5), dataset_name='MNIST', run_test=True)

def cifar10_exp():
    # I probably need to pre-train the AE part
    # run_exp_model(MyResNet29, DunetModel, Pretrained_C0_A2_Model, dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MyResNet29, DunetModel, C0_A2_Model, dataset_name='CIFAR10', run_test=True)
    
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0.2), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0.5), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(1), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(1.5), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(5), dataset_name='CIFAR10', run_test=True)

def transfer_exp():
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    print('training CNNs for transfer models')
    for m in [MNISTModel, DefenseGAN_a, DefenseGAN_b, DefenseGAN_c, DefenseGAN_d]:
        train_CNN(m, train_x, train_y, dataset_name='MNIST')
    print('Testing ..')
    for m in [
            DefenseGAN_a,
            DefenseGAN_b,
            DefenseGAN_c,
            DefenseGAN_d]:
        for l in [0, 0.2, 0.5, 1, 1.5, 5]:
            test_model_transfer(MNISTModel, AEModel, get_lambda_model(l),
                                test_x, test_y,
                                to_cnn_cls=m,
                                dataset_name='MNIST')
    
if __name__ == '__main__':

    # run_exp_model(MNISTModel, AEModel, A2_Model, dataset_name='MNIST', run_test=True)

    # mnist_exp()
    # cifar10_exp()
    # transfer_exp()

    # baseline adversarial training
    # run_exp_model(MNISTModel, IdentityAEModel, A2_Model, dataset_name='MNIST', run_test=True)
    run_exp_model(MNISTModel, AEModel, B2_Model, dataset_name='MNIST', run_test=True)
    
    # run_exp_model(MNISTModel, AEModel, C0_A2_Model, dataset_name='MNIST', run_test=True)

    # Debugging the Cifar10 exps
    # run_exp_model(MyResNet29, DunetModel, A2_Model, dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MyResNet29, DunetModel, C0_A2_Model, dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MNISTModel, IdentityAEModel, A2_Model, dataset_name='MNIST', run_test=True)
    # run_exp_model(MNISTModel, IdentityAEModel, C0_A2_Model, dataset_name='MNIST', run_test=True)
    # rm saved_models/MNIST-mnistcnn-identityAE-A2-AdvAE.hdf5
    # rm images/test-result-MNIST-mnistcnn-identityAE-A2.*
    
    # run_exp_model(MyResNet29, IdentityDunetModel, C0_A2_Model,
    #               dataset_name='CIFAR10', run_test=True)
    # main()
    # main_train_cnn()
    # main_exp(run_test=True)
    # main_test()


def parse_single_whitebox(json_file):
    with open(json_file, 'r') as fp:
        j = json.load(fp)
        return [j[0]['AE clean'],
                j[2]['FGSM']['whiteadv_rec'],
                j[3]['PGD']['whiteadv_rec'],
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


def parse_mnist_table(advae_json, advtrain_json, hgd_json, bbox_json, defgan_json):
    """Parse result for MNIST table.

    Rows (attacks)
    - no attack
    - FGSM
    - PGD
    - CW L2

    Columns (defences):
    - no defence
    - AdvAE oblivious
    - AdvAE blackbox
    - AdvAE whitebox
    - HGD whitebox
    - DefenseGAN whitebox
    - adv training

    """
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)
    with open(bbox_json, 'r') as fp:
        bbox = json.load(fp)
    with open(defgan_json, 'r') as fp:
        defgan = json.load(fp)
    res = {}
    res['nodef'] = [advae[0]['CNN clean'],
                    advae[2]['FGSM']['obliadv'],
                    advae[3]['PGD']['obliadv'],
                    advae[1]['CW']['obliadv']]
    
    res['advae obli'] = [0,
                         # FIXME adjust json format to remove this
                         # indices
                         advae[2]['FGSM']['obliadv_rec'],
                         advae[3]['PGD']['obliadv_rec'],
                         advae[1]['CW']['obliadv_rec']]
    res['advae bbox'] = [0,
                         bbox['FGSM'],
                         bbox['PGD'],
                         bbox['CW']]
    
    res['advae whitebox'] = parse_single_whitebox(advae_json)
    res['hgd whitebox'] = parse_single_whitebox(hgd_json)
    res['defgan whitebox'] = [defgan['CNN clean'],
                              defgan['FGSM'],
                              defgan['PGD'],
                              defgan['CW']]
    res['advtrain whitebox'] = parse_single_whitebox(advtrain_json)
    # print out the table in latex format

    for i, name in enumerate(['No attack', 'FGSM', 'PGD', 'CW $\ell_2$']):
        print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\\\\'
              .format(name, *[res[key][i]*100 for key in res]))
    return


def parse_mnist_transfer_table(trans_jsons, ensemble_jsons):
    """Table for transfer and ensemble models.

    We only evaluate whitebox here.

    Rows (attacks)
    - no attack
    - FGSM
    - PGD
    - CW L2

    Columns (defences setting):
    - X/A
    - X/B
    - X/C
    - X,A,B/A
    - X,A,B/B
    - X,A,B/C
    - X,A,B/D
    """
    trans = []
    ensemble = []
    for tj in trans_jsons:
        trans.append(parse_single_whitebox(tj))
    # for ej in ensemble_jsons:
    #     ensemble.append(parse_single_whitebox(ej))
    for i, name in enumerate(['No attack', 'FGSM', 'PGD', 'CW $\ell_2$']):
        print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & '
              .format(name, *[r[i]*100 for r in trans]), end=' ')
        # print('{:.1f} & {:.1f} & {:.1f} & {:.1f}\\\\'
        #       .format(*[r[i]*100 for r in ensemble]))
        print('')

def parse_lambda_table():
    fmt = 'images/test-result-MNIST-mnistcnn-ae-C0_A2_{}.json'
    fmt2 = 'images/test-result-MNIST-mnistcnn-ae-C0_A2_{}-TO-DefenseGAN_d.json'
    # fmt = 'images/test-result-CIFAR10-resnet29-dunet-C0_A2_{}.json'
    lambdas = [0, 0.2, 0.5, 1, 1.5, 5]
    res = {}
    res['FGSM'] = []
    res['PGD'] = []
    res['PGD X/D'] = []
    res['no'] = []
    res['CW'] = []
    for l in lambdas:
        with open(fmt.format(l), 'r') as fp:
            j = json.load(fp)
            res['no'].append(j[0]['AE clean'])
            res['PGD'].append(j[3]['PGD']['whiteadv_rec'])
            res['FGSM'].append(j[2]['FGSM']['whiteadv_rec'])
            res['CW'].append(j[1]['CW']['whiteadv_rec'])
        with open(fmt2.format(l), 'r') as fp:
            j = json.load(fp)
            res['PGD X/D'].append(j[3]['PGD']['whiteadv_rec'])
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('No attack', *[r*100 for r in res['no']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('PGD', *[r*100 for r in res['PGD']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('FGSM', *[r*100 for r in res['FGSM']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('CW', *[r*100 for r in res['CW']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('PGD X/D', *[r*100 for r in res['PGD X/D']]), end=' ')
    # return res

def __test():
    # extracting experiment results

    parse_lambda_table()
    parse_single_whitebox('images/test-result-MNIST-mnistcnn-ae-C0_A2_1.json')

    # MNIST
    parse_mnist_table(
        'images/test-result-MNIST-mnistcnn-ae-A2.json',
        'images/test-result-MNIST-mnistcnn-identityAE-A2.json',
        'images/test-result-MNIST-mnistcnn-ae-B2.json',
        'images/test-result-MNIST-mnistcnn-ae-A2-BBOX-mnistcnn.json',
        'images/defgan.json')

    parse_mnist_transfer_table(
        ['images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_a.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_b.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_c.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_d.json'],
        ['images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_a.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_b.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_c.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_d.json'
        ])

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

