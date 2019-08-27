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

import foolbox

from utils import *
from tf_utils import *
from models import *
from defensegan_models import *
from attacks import *
from pgd_bpda import *
from blackbox import test_sub

from main import compute_names, load_model

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

def evaluate_attack(sess, model, attack_name, xval, yval, num_samples=100, eps=[]):
    assert len(eps) > 0
    
    shuffle_idx = np.arange(xval.shape[0])
    idx = shuffle_idx[:num_samples]

    if attack_name in ['PGD', 'FGSM']:
        return evaluate_attack_PGD(sess, model, attack_name, xval[idx], yval[idx], eps=eps)
    elif attack_name in ['CW', 'Hop']:
        return evaluate_attack_Hop(sess, model, attack_name, xval[idx], yval[idx], eps=eps)
    else:
        assert False
    
def evaluate_attack_PGD(sess, model, attack_name, xval, yval, eps):
    """PGD likes. The attack will take eps as argument."""
    accs = []
    for e in eps:
        if attack_name is 'PGD':
            adv_val = my_PGD_np(sess, model, xval, {'eps': e})
        elif attack_name is 'FGSM':
            adv_val = my_FGSM_np(sess, model, xval, {'eps': e})
        else:
            assert False
        acc = sess.run(model.accuracy, feed_dict={model.x: adv_val, model.y: yval})
        accs.append(acc)
    return np.array(accs).tolist()

def my_HSJA_foolbox_np(sess, model, xval, yval):
    with sess.as_default():
        fbmodel = foolbox.models.TensorFlowModel(model.x, model.logits, (0, 1))
        # fbmodel = foolbox.models.TensorFlowModel(model.cnn_model.x, model.cnn_model.logits, (0, 1))
        attack = foolbox.attacks.BoundaryAttackPlusPlus(fbmodel, distance=foolbox.distances.Linfinity)
        print('Single attacking ..')
        # adversarial = attack(test_x[0], np.argmax(test_y[0]), max_epsilon=0.3)

        res = []
        for x,y in zip(xval, yval):
            adversarial = attack(x, np.argmax(y))
            res.append(adversarial)
        return np.array(res)

def evaluate_attack_Hop(sess, model, attack_name, xval, yval, eps):
    # assuming attack name is 'Hop'
    if attack_name is 'CW':
        adv_val = my_CW_np(sess, model, xval)
    elif attack_name is 'Hop':
        # adv_val = my_HopSkipJump_np(sess, model, xval)
        adv_val = my_HSJA_foolbox_np(sess, model, xval, yval)
    else:
        assert False
    # filter about distortion level
    # get indices that are within eps
    diff = adv_val - xval

    accs = []
    for e in eps:
        # The problem of np.linalg.norm is that, it sum the row/column for 2D matrix
        # np.linalg.norm(diff, ord=np.inf, axis=(1,2))

        diffnorm = np.max(np.abs(diff), axis=(1,2))
        # round because PGD is giving 0.3000004
        # .round(decimals=4)
        print(diffnorm)
        idx = diffnorm <= e
        idx = idx.reshape(-1)

        print('All: {}, valid: {}'.format(idx.shape[0], np.sum(idx)))

        # probably just replace invalid ones with xval
        adv_val_copy = adv_val.copy()
        adv_val_copy[np.logical_not(idx)] = xval[np.logical_not(idx)]

        acc = sess.run(model.accuracy, feed_dict={model.x: adv_val_copy, model.y: yval})
        accs.append(acc)
    return np.array(accs).tolist()

def exp_epsilon(ae_cls, advae_cls, num_samples, save_name):
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

def __test_plot():
    myplot()

def myplot():

    # TODO plot
    with open('images/epsilon-advae-100.json') as fp:
        advae = json.load(fp)
    with open('images/epsilon-hgd-100-fix.json') as fp:
        hgd = json.load(fp)
    with open('images/epsilon-itadv-100.json') as fp:
        itadv = json.load(fp)

    fig = plt.figure(
        # figsize=(fig_width, fig_height),
        dpi=300)
    
    # fig.canvas.set_window_title('My Grid Visualization')
    plt.plot(hgd['eps'], hgd['PGD'], 'x-', color='green', markersize=4, label='PGD / HGD')
    plt.plot(hgd['eps'], hgd['Hop'], 'x', dashes=[2, 3],  color='green', markersize=4, label='HSJA / HGD')
    plt.plot(itadv['eps'], itadv['PGD'], '^-', color='brown', markersize=4, label='PGD / ItAdv')
    plt.plot(itadv['eps'], itadv['Hop'], '^', dashes=[2,3], color='brown', markersize=4, label='HSJA / ItAdv')
    plt.plot(advae['eps'], advae['PGD'], 'o-', color='blue', markersize=4,
             label='PGD / ' + r"$\bf{" + 'AdvAE' + "}$")
    plt.plot(advae['eps'], advae['Hop'], 'o', dashes=[2,3], color='blue', markersize=4,
             label='HSJA / ' + r"$\bf{" + 'AdvAE' + "}$")
    # HopSkipJump
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    # plt.title('Accuracy against PGD and HSJA on different distortions')
    
    # if titles is None:
    #     titles = [''] * len(images)

    # for image, ax, title in zip(images, axes.reshape(-1), titles):
    #     # ax.set_axis_off()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     # TITLE has to appear on top. I want it to be on bottom, so using xlabel
    #     ax.set_title(title, loc='left')
    #     # ax.set_xlabel(title)
    #     # ax.imshow(convert_image_255(image), cmap='gray')
    #     # ax.imshow(image)
    #     ax.imshow(image, cmap='gray')
    
    # plt.subplots_adjust(hspace=0.5)
    plt.savefig('test.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    # res = exp_epsilon(AEModel, get_lambda_model(0), 10, 'images/epsilon-advae-10.json')
    # res = exp_epsilon(AEModel, B2_Model, 10, 'images/epsilon-hgd-10.json')
    # res = exp_epsilon(IdentityAEModel, A2_Model, 10, 'images/epsilon-itadv-10.json')

    res = exp_epsilon(AEModel, get_lambda_model(0), 100, 'images/epsilon-advae-100.json')
    res = exp_epsilon(AEModel, B2_Model, 100, 'images/epsilon-hgd-100-fix.json')
    res = exp_epsilon(IdentityAEModel, A2_Model, 100, 'images/epsilon-itadv-100.json')
    # TODO defgan

if __name__ == '__main__':
    main()
    

def __test():
    model, sess = load_model(MNISTModel, AEModel, get_lambda_model(0),
                             dataset_name='MNIST')
    model, sess = load_model(MNISTModel, AEModel, B2_Model,
                             dataset_name='MNIST')
    model, sess = load_model(MNISTModel, IdentityAEModel, A2_Model,
                             dataset_name='MNIST')
    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    xval = test_x[:10]
    yval = test_y[:10]

    sess.run(model.accuracy, feed_dict={model.x: a, model.y: yval})

    a = my_HSJA_foolbox_np(sess, model, xval, yval)

    eps = np.arange(0.2,0.5,0.03)

    eps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.55, 0.6, 0.7]

    a = evaluate_attack(sess, model, 'PGD', test_x, test_y,
                    num_samples=10, eps=eps)
    
    with open('test.json', 'w') as fp:
        json.dump(np.array(a).tolist(), fp, indent=4)
    evaluate_attack(sess, model, 'Hop', test_x, test_y,
                    num_samples=10, eps=eps)
    
    images, _, _, data = model.test_attack(sess, test_x, test_y, 'Hop', num_samples=100)

    adv = my_HopSkipJump(sess, model, model.x)
    adv_val = sess.run(adv, feed_dict={model.x: test_x[:10], model.y: test_y[:10]})

    # cleverhans.model.CallableModelWrapper()

    adv_val = my_HopSkipJump_np(sess, model, test_x[:20])

    adv_val = my_HopSkipJump_np(sess, model.cnn_model, test_x[:5])
    
    i2, _, _, data = model.test_attack(sess, test_x, test_y, 'PGD', num_samples=100)
    _, _, _, data = model.test_attack(sess, test_x, test_y, 'CW', num_samples=100)
    print(data)

    model.test_all(sess, test_x, test_y,
                   # attacks=['CW', 'FGSM', 'PGD', 'Hop'],
                   # attacks=['FGSM', 'PGD'],
                   attacks=['Hop'],
                   save_prefix='test',
                   num_samples=100)

    # testing foolbox attacks
    with sess.as_default():
        fbmodel = foolbox.models.TensorFlowModel(model.x, model.logits, (0, 1))
        # fbmodel = foolbox.models.TensorFlowModel(model.cnn_model.x, model.cnn_model.logits, (0, 1))
        # criterion = foolbox.criteria.TargetClassProbability('ostrich', p=0.99)
        # attack = foolbox.attacks.FGSM(fbmodel)
        # attack = foolbox.attacks.BoundaryAttack(fbmodel)
        # attack = foolbox.attacks.CarliniWagnerL2Attack(fbmodel)
        # attack = foolbox.attacks.PGD(fbmodel, distance=foolbox.distances.Linfinity)

        attack = foolbox.attacks.BoundaryAttackPlusPlus(fbmodel, distance=foolbox.distances.Linfinity)
        # attack = foolbox.attacks.BoundaryAttackPlusPlus(fbmodel)
        # foolbox.attacks.boundary_attack

        print('Single attacking ..')
        # adversarial = attack(test_x[0], np.argmax(test_y[0]), max_epsilon=0.3)
        
        # If binary search is not set, the epsilon is increased by 1.5
        # each time in a loop, this does not make sense if I want to
        # experiment with a fixed distortion level. But it probably
        # makes sense to search for hyperparameter optimization?
        # adversarial = attack(test_x[0], np.argmax(test_y[0]), epsilon=0.3, binary_search=False)
        adversarial = attack(test_x[0], np.argmax(test_y[0]))
        
        # adversarial = attack(test_x[0], np.argmax(test_y[0]))
        # print(np.argmax(fbmodel.forward_one(image)))

        # batching
        
        # print('Batching attacking ..')
        # criterion = foolbox.criteria.Misclassification()
        # attack_create_fn = foolbox.batch_attacks.PGD
        # advs = foolbox.run_parallel(attack_create_fn, fbmodel, criterion,
        #                             test_x[:100], np.argmax(test_y[:100], axis=1))

    np.argmax(sess.run(model.logits,
                       feed_dict={
                           # model.x: [test_x[0]]
                           model.x: [adversarial]
                       }))

    np.argmax(sess.run(model.logits,
                       feed_dict={
                           # model.x: test_x[:100]
                           # model.x: [advs[8].perturbed]
                           model.x: [test_x[8]]
                           # model.x: [adversarial]
                       }), axis=1)

    np.max(np.abs(adversarial - test_x[0]))

    grid_show_image([[test_x[0], adversarial]])
