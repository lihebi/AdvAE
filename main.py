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


from utils import *
from model import DenoisedCNN, DenoisedCNN_Var1, DenoisedCNN_Var2
from train import my_adv_training
from attacks import *

def train_denoising(model_cls, saved_folder, prefix, advprefix, run_adv=True, overwrite=False):
    """Train AdvAE.

    - prefix: CNN and AE ckpt prefix
    - advprefix: AdvAE ckpt prefix
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        # model = DenoisedCNN()
        # model1 = DenoisedCNN_Var1()
        # model2 = DenoisedCNN_Var2()
        model = model_cls()
        
        init = tf.global_variables_initializer()
        sess.run(init)

        (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
        
        # 1. train CNN
        pCNN = os.path.join(saved_folder, '{}-CNN.ckpt'.format(prefix))
        if not os.path.exists(pCNN+'.meta'):
            print('Trianing CNN ..')
            model.train_CNN(sess, train_x, train_y)
            print('Saving model ..')
            model.save_CNN(sess, pCNN)
        else:
            print('Trained, directly loadding ..')
            model.load_CNN(sess, pCNN)
        print('Testing CNN ..')
        model.test_CNN(sess, test_x, test_y)

        # 2. train denoiser
        pAE = os.path.join(saved_folder, '{}-AE.ckpt'.format(prefix))
        if not os.path.exists(pAE+'.meta'):
            print('Training AE ..')
            model.train_AE(sess, train_x)
            print('Saving model ..')
            model.save_AE(sess, pAE)
        else:
            print('Trained, directly loadding ..')
            model.load_AE(sess, pAE)
        print('Testing AE ..')
        model.test_AE(sess, test_x)
        model.test_Entire(sess, test_x, test_y)
        
        # 3. train denoiser using adv training, with high level feature guidance
        acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
        print('Model accuracy on clean data: {}'.format(acc))


        # my_adv_training(sess, model, model.ce_loss, [],
        #                 model.adv_train_step)
        # my_adv_training(sess, model, model.rec_loss, [],
        #                 model.adv_rec_train_step)
        if run_adv:
            pAdvAE = os.path.join(saved_folder, '{}-AdvAE.ckpt'.format(advprefix))
            # overwrite controls only the AdvAE weights. CNN and AE
            # weights do not need to be retrained because I'm not
            # experimenting with changing training or loss for that.
            if not os.path.exists(pAdvAE+'.meta') or overwrite:
                print('Adv training AdvAE ..')
                my_adv_training(sess, model, model.unified_adv_loss,
                                [model.rec_loss, model.ce_loss,
                                 model.clean_rec_loss,model.clean_ce_loss,
                                 model.noisy_rec_loss, model.noisy_ce_loss],
                                model.unified_adv_train_step,
                                num_epochs=50)
                print('Saving model ..')
                model.save_AE(sess, pAdvAE)
            else:
                print('Trained, directly loadding ..')
                model.load_AE(sess, pAdvAE)
            print('Testing AdvAE ..')
            model.test_Adv_Denoiser(sess, test_x[:10], test_y[:10])

    
def test_model_against_attack(sess, model, attack_func, inputs, labels, targets, prefix=''):
    """
    testing against several attacks. Return the atatck accuracy and L2 distortion.
    """
    # generate victims
    adv_x = attack_func(sess, model)
    # adv_x = my_CW(sess, model)
    adv_preds = model.predict(adv_x)
    if targets is None:
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: labels})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: labels})
        adv_acc = model.compute_accuracy(sess, model.x, model.y, adv_preds, inputs, labels)
    else:
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: targets})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: targets})
        adv_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, labels)

    l2 = np.mean(mynorm(inputs, adv_x_concrete, 2))

    pngfile = prefix + '-out.png'
    print('Writing adv examples to {} ..'.format(pngfile))
    grid_show_image(adv_x_concrete[:10], 5, 2, pngfile)
    
    print('True label: {}'.format(np.argmax(labels, 1)))
    print('Predicted label: {}'.format(np.argmax(adv_preds_concrete, 1)))
    print('Adv input accuracy: {}'.format(adv_acc))
    print("L2: {}".format(l2))
    if targets is not None:
        target_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, targets)
        print('Targeted label: {}'.format(np.argmax(targets, 1)))
        print('Targeted accuracy: {}'.format(target_acc))
    return adv_acc, l2

def test_against_attacks(sess, model):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
     # all 10000 inputs
    inputs, labels = test_x, test_y
    # random 100 inputs
    indices = random.sample(range(test_x.shape[0]), 100)
    inputs = test_x[indices]
    labels = test_y[indices]

    print('testing FGSM ..')
    acc, l2 = test_model_against_attack(sess, model, my_FGSM, inputs, labels, None, prefix='FGSM')
    print('{} acc: {}, l2: {}'.format('FGSM', acc, l2))
    
    print('testing PGD ..')
    acc, l2 = test_model_against_attack(sess, model, my_PGD, inputs, labels, None, prefix='PGD')
    print('{} acc: {}, l2: {}'.format('PGD', acc, l2))
    
    print('testing JSMA ..')
    acc, l2 = test_model_against_attack(sess, model, my_JSMA, inputs, labels, None, prefix='JSMA')
    print('{} acc: {}, l2: {}'.format('JSMA', acc, l2))
    
    print('testing CW ..')
    # FIXME CW l2 is a bit problematic, because in case it fails, it
    # will output the same value, and thus 0 l2. In general, I need to
    # calculate L2 only for successful attacks.
    acc, l2 = test_model_against_attack(sess, model, my_CW, inputs, labels, None, prefix='CW')
    print('{} acc: {}, l2: {}'.format('CW', acc, l2))

def main():
    # training
    train_denoising(DenoisedCNN, 'saved_model/AdvDenoiser', '0', '0')
    train_denoising(DenoisedCNN, 'saved_model/AdvDenoiser', '0', '4')
    # only train CNN and AE for these CNN architecture. The AdvAE weights will be loaded.
    train_denoising(DenoisedCNN_Var1, 'saved_model/AdvDenoiser', '1', '1', run_adv=False)
    train_denoising(DenoisedCNN_Var2, 'saved_model/AdvDenoiser', '2', '2', run_adv=False)
    # testing
    # experiments

def __test_denoising(path):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()

    tf.reset_default_graph()
    
    sess = tf.Session()

    # different CNN models
    model = DenoisedCNN()
    model = DenoisedCNN_Var1()
    model = DenoisedCNN_Var2()

    # init session
    init = tf.global_variables_initializer()
    sess.run(init)

    # load different CNNs
    model.load_CNN(sess, 'saved_model/AdvDenoiser/0-CNN.ckpt')
    model.load_CNN(sess, 'saved_model/AdvDenoiser/1-CNN.ckpt')
    model.load_CNN(sess, 'saved_model/AdvDenoiser/2-CNN.ckpt')

    # load the same AE
    model.load_AE(sess, 'saved_model/AdvDenoiser/0-AdvAE.ckpt')
    model.load_AE(sess, 'saved_model/AdvDenoiser/3-AdvAE.ckpt')
    model.load_AE(sess, 'saved_model/AdvDenoiser/4-AdvAE.ckpt')

    # testing
    model.test_CNN(sess, test_x, test_y)
    model.test_AE(sess, test_x)
    model.test_Entire(sess, test_x, test_y)
    model.test_Adv_Denoiser(sess, test_x[:10], test_y[:10])
    
    acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
    print('Model accuracy on clean data: {}'.format(acc))

    # testing against adv
    test_against_attacks(sess, model)
