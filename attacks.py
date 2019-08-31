import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cleverhans.model
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import HopSkipJumpAttack
from bpda import CarliniWagnerL2_BPDA

import foolbox


CLIP_MIN = 0.
CLIP_MAX = 1.

def generate_victim(test_x, test_y):
    inputs = []
    labels = []
    targets = []
    nlabels = 10
    for i in range(nlabels):
        for j in range(1000):
            x = test_x[j]
            y = test_y[j]
            if i == np.argmax(y):
                inputs.append(x)
                labels.append(y)
                onehot = np.zeros(nlabels)
                onehot[(i+1) % 10] = 1
                targets.append(onehot)
                break
    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    return inputs, labels, targets

def my_FGSM(model, x, y=None, params=dict()):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'y': y,
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX
    }
    fgsm_params.update(params)
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    return tf.stop_gradient(adv_x)
    # return adv_x
def my_PGD(model, x, y=None, params=dict()):
    pgd = ProjectedGradientDescent(model)
    pgd_params = {'eps': 0.3,
                  # CAUTION I need this, otherwise the adv acc data is
                  # not accurate
                  'y': y,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': CLIP_MIN,
                  'clip_max': CLIP_MAX}
    pgd_params.update(params)
    adv_x = pgd.generate(x, **pgd_params)
    return tf.stop_gradient(adv_x)

def my_HopSkipJump(sess, model, x, params=dict()):
    assert False
    attack = HopSkipJumpAttack(model, sess=sess)
    # TODO add allowed pertubation parameter
    default_params = {
        # 'stepsize_search': 0.01,
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX,
        # 'num_iterations': 40,
        # anything that is not l2
        'constraint': 'linf',
        'verbose': False,
    }
    # FIXME the generate method of HopSkipJumpAttack is problematic,
    # it uses x[0] instead of x, i.e. it only generates adv for one
    # value, the adv_x and x is not the same shape (although it
    # pretended to be by setting the shape at the end).
    adv_x = attack.generate(x, **default_params)
    return tf.stop_gradient(adv_x)
def my_HopSkipJump_np(sess, model, xval, params=dict()):
    assert False
    attack = HopSkipJumpAttack(model, sess=sess)
    # TODO add allowed pertubation parameter
    default_params = {
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX,
        # anything that is not l2
        'constraint': 'linf',
        # 'verbose': False,
    }
    adv_xval = attack.generate_np(xval, **default_params)
    return adv_xval

def my_HSJA_foolbox_np(sess, model, xval, yval):
    with sess.as_default():
        fbmodel = foolbox.models.TensorFlowModel(model.x, model.logits, (0, 1))
        # fbmodel = foolbox.models.TensorFlowModel(model.cnn_model.x, model.cnn_model.logits, (0, 1))
        attack = foolbox.attacks.BoundaryAttackPlusPlus(fbmodel, distance=foolbox.distances.Linfinity)
        print('Single attacking ..')
        # adversarial = attack(test_x[0], np.argmax(test_y[0]), max_epsilon=0.3)

        res = []
        for i, (x,y) in enumerate(zip(xval, yval)):
            print('foolbox HSJA attacking {}-th sample ..'.format(i))
            adversarial = attack(x, np.argmax(y), log_every_n_steps=20)
            # FIXME NOW this might be None
            # print(adversarial.shape)
            assert adversarial is not None
            res.append(adversarial)
        return np.array(res)

def my_JSMA(model, x, params=dict()):
    jsma = SaliencyMapMethod(model)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': CLIP_MIN, 'clip_max': CLIP_MAX,
                   'y_target': None}
    jsma_params.update(params)
    return jsma.generate(x, **jsma_params)
def my_CW(sess, model, x, y, targeted=False, params=dict()):
    """When targeted=True, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess=sess)
    yname = 'y_target' if targeted else 'y'
    cw_params = {'binary_search_steps': 1,
                 'y': y,
                 'max_iterations': 1000,
                 'learning_rate': 0.2,
                 'batch_size': 50,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    cw_params.update(params)
    adv_x = cw.generate(x, **cw_params)
    return adv_x
def my_CW_BPDA(sess, pre_model, post_model, x, y, targeted=False, params=dict()):
    cw = CarliniWagnerL2_BPDA(pre_model, post_model, sess=sess)
    yname = 'y_target' if targeted else 'y'
    cw_params = {'binary_search_steps': 1,
                 yname: y,
                 # 3 sec per iteration
                 'max_iterations': 10,
                 'learning_rate': 0.2,
                 # to match the defgan model
                 'batch_size': 50,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    cw_params.update(params)
    adv_x = cw.generate(x, **cw_params)
    return tf.stop_gradient(adv_x)


    
def evaluate_attack_PGD(sess, model, attack_name, xval, yval, eps):
    """PGD likes. The attack will take eps as argument."""
    accs = []
    for e in eps:
        if attack_name is 'PGD':
            adv = my_PGD(model, model.x, y=model.y, params={'eps': e})
        elif attack_name is 'FGSM':
            adv = my_FGSM(model, model.x, y=model.y, params={'eps': e})
        else:
            assert False
        adv_val = sess.run(adv, feed_dict={model.x: xval, model.y: yval})
        acc = sess.run(model.accuracy, feed_dict={model.x: adv_val, model.y: yval})
        accs.append(acc)
    return np.array(accs).tolist()

def evaluate_attack_CW(sess, model, xval, yval):
    """This is L2, so no eps, no thresholding."""
    adv = my_CW(model, xval)
    adv_val = sess.run(adv, feed_dict={model.x: xval, model.y: yval})
    acc = sess.run(model.accuracy, feed_dict={model.x: adv_val, model.y: yval})
    return float(acc)

def evaluate_attack_Hop(sess, model, attack_name, xval, yval, eps):
    # assuming attack name is 'Hop'
    if attack_name is 'Hop':
        # adv_val = my_HopSkipJump_np(sess, model, xval)
        adv_val = my_HSJA_foolbox_np(sess, model, xval, yval)
    else:
        assert False
    # filter about distortion level
    # get indices that are within eps
    # print(adv_val)
    # print(adv_val.shape)
    # print(xval.shape)
    diff = adv_val - xval

    accs = []
    for e in eps:
        # The problem of np.linalg.norm is that, it sum the row/column for 2D matrix
        # np.linalg.norm(diff, ord=np.inf, axis=(1,2))

        diffnorm = np.max(np.abs(diff), axis=(1,2))
        # round because PGD is giving 0.3000004
        # .round(decimals=4)
        # print(diffnorm)
        idx = diffnorm <= e
        idx = idx.reshape(-1)

        print('All: {}, valid: {}'.format(idx.shape[0], np.sum(idx)))

        # probably just replace invalid ones with xval
        adv_val_copy = adv_val.copy()
        adv_val_copy[np.logical_not(idx)] = xval[np.logical_not(idx)]

        acc = sess.run(model.accuracy, feed_dict={model.x: adv_val_copy, model.y: yval})
        accs.append(acc)
    return np.array(accs).tolist()

def evaluate_attack(sess, model, attack_name, xval, yval, num_samples=100, eps=[]):
    assert len(eps) > 0
    
    shuffle_idx = np.arange(xval.shape[0])
    idx = shuffle_idx[:num_samples]

    if attack_name in ['PGD', 'FGSM']:
        return evaluate_attack_PGD(sess, model, attack_name, xval[idx], yval[idx], eps=eps)
    elif attack_name in ['Hop']:
        return evaluate_attack_Hop(sess, model, attack_name, xval[idx], yval[idx], eps=eps)
    elif attack_name in ['CW']:
        return evaluate_attack_CW(sess, model, xval[idx], yval[idx])
    else:
        assert False


def evaluate_no_attack_CNN(sess, model, xval, yval, num_samples=100):
    shuffle_idx = np.arange(xval.shape[0])
    idx = shuffle_idx[:num_samples]
    acc = sess.run(model.CNN_accuracy, feed_dict={model.x: xval[idx], model.y: yval[idx]})
    return acc.tolist()

def evaluate_no_attack_AE(sess, model, xval, yval, num_samples=100):
    shuffle_idx = np.arange(xval.shape[0])
    idx = shuffle_idx[:num_samples]
    acc = sess.run(model.accuracy, feed_dict={model.x: xval[idx], model.y: yval[idx]})
    return acc.tolist()
