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
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX
    }
    fgsm_params.update(params)
    # FIXME Cleverhans has a bug in Attack.get_or_guess_labels: it
    # only checks y is in the key, but didn't check whether the value
    # is None. fgsm.generate calls that function
    if y is not None:
        fgsm_params.update({'y': y})
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    return tf.stop_gradient(adv_x)

def my_PGD(model, x, y=None, params=dict()):
    # DEBUG I would always want to supply the y manually
    # assert y is not None
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
    # DEBUG do I need to stop gradients here?
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
            if adversarial is None:
                adversarial = np.zeros_like(x)
                print('WARNING: HSJA adversarial is None. Setting to 0s.')
            assert adversarial is not None
            res.append(adversarial)
        return np.array(res)

def my_JSMA(model, x, params=dict()):
    jsma = SaliencyMapMethod(model)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': CLIP_MIN, 'clip_max': CLIP_MAX,
                   'y_target': None}
    jsma_params.update(params)
    return tf.stop_gradient(jsma.generate(x, **jsma_params))
def my_CW(sess, model, x, y, params=dict()):
    """When targeted=True, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess=sess)
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
    return tf.stop_gradient(adv_x)
def my_CW_BPDA(sess, pre_model, post_model, x, y, params=dict()):
    cw = CarliniWagnerL2_BPDA(pre_model, post_model, sess=sess)
    cw_params = {'binary_search_steps': 1,
                 'y': y,
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
        print('Running on eps ', e)
        if attack_name is 'PGD':
            # setting mainly the niter and step_size
            params = model.cnn_model.PGD_params
            params.update({'eps': e})
            adv = my_PGD(model, model.x,
                         y=model.y,
                         params=params)
        elif attack_name is 'FGSM':
            params = model.cnn_model.FGSM_params
            params.update({'eps': e})
            adv = my_FGSM(model, model.x,
                          y=model.y,
                          params=params)
        else:
            assert False
        def foo(batch_x, batch_y):
            adv_val = sess.run(adv, feed_dict={model.x: batch_x, model.y: batch_y})
            acc = sess.run(model.accuracy, feed_dict={model.x: adv_val, model.y: batch_y})
            return acc
        acc = run_on_batch_light(foo, xval, yval, 100)
        accs.append(sum(acc) / len(acc))
    return np.array(accs).tolist()

def evaluate_attack_CW(sess, model, xval, yval):
    """This is L2, so no eps, no thresholding."""
    def foo(x,y):
        adv = my_CW(sess, model, model.x, model.y)
        adv_val = sess.run(adv, feed_dict={model.x: x, model.y: y})
        acc = sess.run(model.accuracy, feed_dict={model.x: adv_val, model.y: y})
        return acc
    res = run_on_batch_light(foo, xval, yval, 100)
    return float(sum(res) / len(res))

def evaluate_attack_Hop(sess, model, attack_name, xval, yval, eps):
    assert xval.shape[0] <= 100
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

def evaluate_attack(sess, model, attack_name, xval, yval, eps=[]):
    if attack_name in ['PGD', 'FGSM']:
        return evaluate_attack_PGD(sess, model, attack_name, xval, yval, eps=eps)
    elif attack_name in ['Hop']:
        return evaluate_attack_Hop(sess, model, attack_name, xval, yval, eps=eps)
    elif attack_name in ['CW']:
        return evaluate_attack_CW(sess, model, xval, yval)
    else:
        assert False

def evaluate_no_attack_CNN(sess, model, xval, yval):
    # FIXME does lexical scope work?
    def foo(batch_x, batch_y):
        return sess.run(model.CNN_accuracy, feed_dict={model.x: batch_x, model.y: batch_y})
    accs = run_on_batch_light(foo, xval, yval, 100)
    # FIXME not accurate if cannot divide equally
    return sum(accs) / len(accs)

def evaluate_no_attack_AE(sess, model, xval, yval, num_samples=100):
    def foo(batch_x, batch_y):
        return sess.run(model.accuracy, feed_dict={model.x: batch_x, model.y: batch_y})
    accs = run_on_batch_light(foo, xval, yval, 100)
    return sum(accs) / len(accs)

def run_on_batch_light(f, npx, npy, batch_size):
    nbatch = npx.shape[0] // batch_size
    print('running on {} batches: '.format(nbatch), end='', flush=True)
    allres = []
    for i in range(nbatch):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_x = npx[start:end]
        batch_y = npy[start:end]
        npres = f(batch_x, batch_y)
        allres.append(npres)
        print('.', end='', flush=True)
    print('')
    return allres
