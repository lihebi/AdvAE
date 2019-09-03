import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import time
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
from exp_utils import *

import warnings

# suppress deprecation warnings. The warning is caused by
# tf.reduce_sum, used in cleverhans
# THIS is not working
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

def run_exp_model(cnn_cls, ae_cls, advae_cls,
                  saved_folder='saved_models',
                  dataset_name='', run_test=True):
    """Both training and testing."""
    # reset seed here
    # tf.random.set_random_seed(0)
    # np.random.seed(0)
    # random.seed(0)
    
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


def main_cifar10_exp():
    # I probably need to pre-train the AE part
    # run_exp_model(MyResNet29, DunetModel, Pretrained_C0_A2_Model, dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MyResNet29, DunetModel, C0_A2_Model, dataset_name='CIFAR10', run_test=True)
    
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0.2), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(0.5), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(1), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(1.5), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(5), dataset_name='CIFAR10', run_test=True)

def main_transfer_exp():
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

def epsilon_exp(ae_cls, advae_cls, num_samples, save_name):
    """Experiment for different epsilon and add advanced blackbox results."""
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    # eps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.55, 0.6, 0.7]
    eps = np.arange(0.01,0.6,0.03).tolist()

    print('#eps {}'.format(len(eps)))
    
    res = {}

    res['eps'] = eps
    # advae
    model, sess = load_model(MNISTModel, ae_cls, advae_cls,
                             dataset_name='MNIST')
    print('Running whitebox PGD ..')
    res['PGD'] = evaluate_attack(sess, model, 'PGD', test_x, test_y,
                                 num_samples=num_samples, eps=eps)
    print(res)
    print('Running blackbox Hop ..')
    res['Hop'] = evaluate_attack(sess, model, 'Hop', test_x, test_y,
                                 num_samples=num_samples, eps=eps)
    print(res)
    print('Saving to {} ..'.format(save_name))
    with open(save_name, 'w') as fp:
        json.dump(res, fp, indent=4)
    return res


def main_epsilon_exp():
    """This is just the main MNIST exp."""
    # res = exp_epsilon(AEModel, get_lambda_model(0), 10, 'images/epsilon-advae-10.json')
    # res = exp_epsilon(AEModel, B2_Model, 10, 'images/epsilon-hgd-10.json')
    # res = exp_epsilon(IdentityAEModel, A2_Model, 10, 'images/epsilon-itadv-10.json')

    # res = epsilon_exp(AEModel, get_lambda_model(0), 100, 'images/epsilon-advae-100.json')
    # res = epsilon_exp(AEModel, B2_Model, 100, 'images/epsilon-hgd-100-fix.json')
    # res = epsilon_exp(IdentityAEModel, A2_Model, 100, 'images/epsilon-itadv-100.json')
    # TODO defgan

    # ItAdvTrain
    # run_exp_model(MNISTModel, IdentityAEModel, A2_Model, dataset_name='MNIST', run_test=True)
    # FIXME these models seems to make HopSkipJumpAttack to return None, so run_test=False for now
    for cnn_cls in [
            # looks like using the PGD with y during training, this
            # thing is not even converging.
            #
            # CNN1AE,
            IdentityAEModel,
            # CNN2AE,
            # CNN3AE
    ]:
        run_exp_model(MNISTModel, cnn_cls, ItAdv_Model, dataset_name='MNIST', run_test=True)
    # AdvAE
    # run_exp_model(MNISTModel, CNN1AE, get_lambda_model(1), dataset_name='MNIST', run_test=True)
    # run_exp_model(MNISTModel, CNN1AE, get_lambda_model(0), dataset_name='MNIST', run_test=True)

    for advae_cls in [get_lambda_model(1), get_lambda_model(0),
                      # HGD
                      B2_Model,
                      # looks like I have to add C2 regularizer, otherwise the clean
                      # AE+CNN has 0.8 accuracy
                      # C2_B2_Model,
                      # seems C0 B2 is not good. Thus probobly C0 is not good? TODO I should try C2
                      C0_B2_Model,
                      # A0?
                      C0_A2_A0_Model, A2_A0_Model,
                      # testing C2
                      # C2_A2_Model,
                      C0_A2_Model]:
        run_exp_model(MNISTModel, CNN1AE, advae_cls, dataset_name='MNIST', run_test=True)
        
def main_lambda_exp():
    # run_exp_model(MNISTModel, AEModel, A2_Model, dataset_name='MNIST', run_test=True)
    # Testing different hyperparameter lambdas
    # this is the same as A2_Model
    # lams = [0, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]
    lams = [0, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]
    for lam in lams:
        run_exp_model(MNISTModel, CNN1AE, get_lambda_model(lam), dataset_name='MNIST', run_test=True)

def main_ae_size():
    for config in [(32,32,32,32),
                   (32,16,16,32),
                   (16,32,32,16),
                   (32,64,64,32)]:
        run_exp_model(MNISTModel,
                      get_wideae_model(config),
                      get_lambda_model(1),
                      dataset_name='MNIST', run_test=True)
    for ae_cls in [CNN1AE, CNN2AE, CNN3AE, FCAE, deepFCAE]:
        run_exp_model(MNISTModel, ae_cls, get_lambda_model(1), dataset_name='MNIST', run_test=True)

def main_new_cifar10():
    # dunet may still be the best
    # dunet has no batch norm layer
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(1), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, IdentityAEModel, ItAdv_Model, dataset_name='CIFAR10', run_test=True)
    # Testing C2
    run_exp_model(MyResNet29, DunetModel, C2_A2_Model, dataset_name='CIFAR10', run_test=True)
    # TODO running B2, the HGD
    # FIXME C0_B2_Model?
    run_exp_model(MyResNet29, DunetModel, B2_Model, dataset_name='CIFAR10', run_test=True)
    
    # testing other lambdas
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(2), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, DunetModel, get_lambda_model(3), dataset_name='CIFAR10', run_test=True)

    # Only Dunet seems to work a little, so I'm not going to even try these
    run_exp_model(MyResNet29, CNN1AE, get_lambda_model(1), dataset_name='CIFAR10', run_test=True)
    run_exp_model(MyResNet29, CNN2AE, get_lambda_model(1), dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MyResNet29, CNN1AE, get_lambda_model(0), dataset_name='CIFAR10', run_test=True)

    run_exp_model(MyResNet29, CNN1AE, ItAdv_Model, dataset_name='CIFAR10', run_test=True)
    # run_exp_model(MyResNet29, CNN2AE, ItAdv_Model, dataset_name='CIFAR10', run_test=True)
    # AdvAE
    # FIXME use some other lambda

def __test():
    m = MNISTModel()
    m = MyResNet29()
    get_wideae_model((32,32,32,32))(m)
    # ae = AEModel(m)
    ae = FCAE(m)
    # ae = deepFCAE(m)
    ae = CNN1AE(m)
    ae = CNN2AE(m)
    ae = CNN3AE(m)
    ae = MNISTAE(m)
    ae = CifarAEModel(m)
    adv = A2_Model(m, ae)
    adv = ItAdv_Model(m, ae)

def test_defgan():
    """Test clean and three attacks on defgan models."""
    sess = create_tf_session()

    from defensegan_test import load_defgan
    defgan = load_defgan(sess)
    cnn = MNISTModel()
    
    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')
    model = DefGanAdvAE(cnn, defgan)
    
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    # shuffle_idx = np.arange(xval.shape[0])
    # idx = shuffle_idx[:num_samples]
    acc = sess.run(model.accuracy, feed_dict={model.x: test_x[:50], model.y: test_y[:50]})
    
    res = test_model_impl(sess, model, test_x, test_y)

if __name__ == '__main__':
    with warnings.catch_warnings():
        # I'm suppressing cleverhans's deprecated usage of reduce_sum
        # This might be dangerous, all warnings.warn will not show up
        warnings.simplefilter("ignore")
        warnings.warn("WARNNNNNNN")
        main_epsilon_exp()
        main_ae_size()
        main_lambda_exp()
        main_new_cifar10()
    
    # main_mnist_exp()
    # main_cifar10_exp()
    # main_transfer_exp()
    # main_epsilon_exp()
