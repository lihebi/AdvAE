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
    # FIXME this CNN should not be used directly
    if 'ItAdv' in advae_cls.NAME():
        print('!!!!!!! ******** computing ItAdv names')
        # FIXME this should be related to the AE name
        pCNN = pCNN + '-' + ae_cls.NAME() + '-ItAdvCNN.hdf5'

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

    # This load model should be for testing, not training
    cnn = cnn_cls(training=False)
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

    cnn.load_weights(sess, pCNN)
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
            cnn = cnn_cls(training=True)

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
    print('-' * 80)
    print('-' * 80)
    print('====== Denoising training for {} ..'.format(pAdvAE))
    if os.path.exists(pCNN) and os.path.exists(pAdvAE):
        print('Already trained {}'.format(pAdvAE))
        return
    
    # 1. train CNN
    # FIXME In case of ItAdv train, the CNN is not pretrained.
    if not os.path.exists(pCNN) and 'ItAdv' not in advae_cls.NAME():
        with create_tf_session() as sess:
            cnn = cnn_cls(training=True)
            cnn.model.summary()
            tf_init_uninitialized(sess)
            print('Training CNN ..')
            cnn.train_CNN(sess, train_x, train_y)
            print('Saving model to {} ..'.format(pCNN))
            cnn.save_weights(sess, pCNN)
    else:
        print('Already trained {} ..'.format(pCNN))
    
    if not os.path.exists(pAdvAE):
        with create_tf_session() as sess:
            # FIXME the training of ItAdv is not stable
            if 'ItAdv' in advae_cls.NAME():
                print('!!!!!!! ******** Training ItAdv, '
                      'setting CNN bath normalization layers trainable')
                # setting BN to trainable
                cnn = cnn_cls(training=True)
                ae = ae_cls(cnn)
                ae.AE.summary()
                adv = advae_cls(cnn, ae)
                tf_init_uninitialized(sess)
                print('Trainng AdvAE ..')
                if dataset_name is 'MNIST':
                    adv.train_Adv(sess, train_x, train_y, augment=False)
                else:
                    adv.train_Adv(sess, train_x, train_y, augment=True)
                # save cnn
                print('Saving model to {} ..'.format(pCNN))
                cnn.save_weights(sess, pCNN)
                print('saving to {} ..'.format(pAdvAE))
                ae.save_weights(sess, pAdvAE)
            else:
                cnn = cnn_cls(training=False)
                ae = ae_cls(cnn)
                ae.AE.summary()
                adv = advae_cls(cnn, ae)
                tf_init_uninitialized(sess)
                # load cnn
                print('Loading {} ..'.format(pCNN))
                cnn.load_weights(sess, pCNN)
                print('Trainng AdvAE ..')
                if dataset_name is 'MNIST':
                    adv.train_Adv(sess, train_x, train_y, augment=False)
                else:
                    adv.train_Adv(sess, train_x, train_y, augment=True)
                print('saving to {} ..'.format(pAdvAE))
                ae.save_weights(sess, pAdvAE)
    else:
        print('Already trained {}'.format(pAdvAE))

def test_model_impl(sess, model, test_x, test_y, dataset_name):
    res = {}
    # FIXME this is total #params. If I want to get trainable
    # ones, I might need to do
    # sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    # print('CNN params: ', model.cnn_model.model.count_params())
    # print('Conv params: ', model.cnn_model.CNN.count_params())
    # print('FC params: ', model.cnn_model.FC.count_params())
    # print('AE params:', model.ae_model.AE.count_params())
    res['CNN params'] = model.cnn_model.model.count_params()
    res['Conv params'] = model.cnn_model.CNN.count_params()
    res['FC params'] = model.cnn_model.FC.count_params()
    if model.ae_model:
        res['AE params'] = model.ae_model.AE.count_params()
    else:
        # currently only for defgan
        res['AE params'] = 0

    # model.test_all(sess, test_x, test_y,
    #                attacks=[
    #                    'CW',
    #                    'FGSM',
    #                    'PGD',
    #                    # 'Hop',
    #                    # 'PGD_BPDA',
    #                    # 'CW_BPDA'
    #                ],
    #                save_prefix=save_prefix)

    # alternative testing
    if dataset_name is 'MNIST':
        eps = np.arange(0.02,0.6,0.04).tolist()
    elif dataset_name is 'CIFAR10':
        # DEBUG more distortion
        eps = (np.arange(2,30,2)/255).tolist()

    # I probably need to use the same set of tests to evaluate these
    # DEBUG number of samples
    indices = random.sample(range(test_x.shape[0]), 100)

    print('testing no atttack CNN ..')
    res["no atttack CNN"] = evaluate_no_attack_CNN(sess, model,
                                                   test_x[indices], test_y[indices])
    print("no atttack CNN", res["no atttack CNN"])

    print('testing no atttack AE ..')
    res["no atttack AE"] = evaluate_no_attack_AE(sess, model,
                                                 test_x[indices], test_y[indices])
    print("no atttack AE", res["no atttack AE"])
    # res["num_samples"] = 50
    
    res['epsilon'] = eps
    
    print('Running whitebox FGSM ..')
    fgsm_res = evaluate_attack(sess, model, 'FGSM',
                               test_x[indices], test_y[indices],
                               eps=eps)
    res['FGSM'] = fgsm_res
    
    print('Running whitebox PGD ..')
    pgd_res = evaluate_attack(sess, model, 'PGD',
                              test_x[indices], test_y[indices],
                              eps=eps)
    res['PGD'] = pgd_res
    
    print('Running CW ..')
    cw_res = evaluate_attack(sess, model, 'CW',
                             test_x[indices], test_y[indices])
    res["CW"] = cw_res
    print(res["CW"])

    # this is extremely slow, so setting to run 20

    print('Running blackbox Hop ..')
    hop_res = evaluate_attack(sess, model, 'Hop',
                              test_x[indices][:20], test_y[indices][:20],
                              eps=eps)
    res["Hop"] = hop_res

    return res

def test_model(cnn_cls, ae_cls, advae_cls,
               test_x, test_y,
               saved_folder='saved_models',
               dataset_name='',
               force=False):
    _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                      dataset_name=dataset_name)
    save_prefix = 'test-result-{}'.format(plot_prefix)
    
    # filename = 'images/{}.pdf'.format(save_prefix)
    filename = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
        print('testing {} ..'.format(plot_prefix))
        res = test_model_impl(sess, model, test_x, test_y, dataset_name)
        print('Saving to {} ..'.format(filename))
        with open(filename, 'w') as fp:
            json.dump(res, fp, indent=4)
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


def test_model_bpda(cnn_cls, ae_cls, advae_cls,
                    test_x, test_y,
                    saved_folder='saved_models',
                    dataset_name='',
                    force=False):
    _, _, plot_prefix = compute_names(cnn_cls, ae_cls, advae_cls,
                                      dataset_name=dataset_name)
    save_prefix = 'test-result-bpda-{}'.format(plot_prefix)
    
    filename = 'images/{}.pdf'.format(save_prefix)
    filename2 = 'images/{}.json'.format(save_prefix)

    if not os.path.exists(filename2) or force:
        print('loading model ..')
        model, sess = load_model(cnn_cls, ae_cls, advae_cls,
                                 dataset_name=dataset_name)
        print('testing {} ..'.format(plot_prefix))

        print('Testing FGSM ..')
        res = {}
        adv_x = my_PGD_BPDA(sess, model.ae_model.evaluate_np,
                            model.cnn_model, model.cnn_model.x, model.cnn_model.y,
                            loss_func='xent', epsilon=0.3, max_steps=1, step_size=0.3, rand=False)
        res['FGSM'] = eval_with_adv(sess, model.AE,
                                    model.cnn_model, model.cnn_model.x, model.cnn_model.y,
                                    adv_x, test_x, test_y)
        print(res['FGSM'])
        print('Testing PGD ..')
        adv_x = my_PGD_BPDA(sess, model.ae_model.evaluate_np,
                            model.cnn_model, model.cnn_model.x, model.cnn_model.y,
                            loss_func='xent')
        res['PGD'] = eval_with_adv(sess, model.AE,
                                   model.cnn_model, model.cnn_model.x, model.cnn_model.y,
                                   adv_x, test_x, test_y)
        print(res['PGD'])
        print(res)

    else:
        print('Already tested, see {}'.format(filename))
def __test():
    # tf.random.set_random_seed(0)
    # np.random.seed(0)
    # random.seed(0)
    dataset_name='MNIST'
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    
    test_model_bpda(MNISTModel, AEModel, get_lambda_model(0),
                    test_x, test_y,
                    dataset_name=dataset_name)

