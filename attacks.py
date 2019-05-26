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
from bpda import CarliniWagnerL2_BPDA


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

def my_FGSM(model, x, params=dict()):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX
    }
    fgsm_params.update(params)
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    return tf.stop_gradient(adv_x)
    # return adv_x
def my_PGD(model, x, params=dict()):
    pgd = ProjectedGradientDescent(model)
    pgd_params = {'eps': 0.3,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': CLIP_MIN,
                  'clip_max': CLIP_MAX}
    pgd_params.update(params)
    adv_x = pgd.generate(x, **pgd_params)
    return tf.stop_gradient(adv_x)
    # return adv_x
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
                 yname: y,
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


