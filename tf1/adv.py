import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Activation
from keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from keras import Sequential
from keras.optimizers import Adam
import numpy as np

from tqdm import tqdm
import time

from data_utils import load_mnist, sample_and_view

from model import get_Madry_model, get_LeNet5, train_CNN
from model import dense_AE, CNN_AE, train_AE, test_AE

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2

def clear_tqdm():
    if '_instances' in dir(tqdm):
        tqdm._instances.clear()

def FGSM(model, x, y=None, params=dict()):
    # FGSM attack
    assert y != None
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    fgsm_params.update(params)
    if y is not None:
        fgsm_params.update({'y': y})
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    return tf.stop_gradient(adv_x)

def PGD(model, x, y=None, params=dict()):
    assert y != None
    pgd = ProjectedGradientDescent(model)
    pgd_params = {'eps': 0.3,
                  'y': y,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': 0.,
                  'clip_max': 1.}
    pgd_params.update(params)
    adv_x = pgd.generate(x, **pgd_params)
    return tf.stop_gradient(adv_x)

def my_softmax_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))
def my_accuracy_wrapper(logits=None, labels=None):
    return tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1),
        tf.argmax(labels, 1)), dtype=tf.float32))

def wrap_cleverhans(model, attack_fn):
    """This should return (newmodel, (x,y,loss)) where newmodel has input connected x

    """
    class Dummy(cleverhans.model.Model):
        def __init__(self, model):
            self.model = model
        def predict(self, x):
            return self.model(x)
        def fprop(self, x, **kwargs):
            logits = self.predict(x)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

    x = keras.Input(shape=(28,28,1))
    # connect mm with x
    assert 'softmax' in model.layers[-1].name
    mm = keras.Model(x, Sequential(model.layers[:-1])(x))
    # this y is not connected. I would need to feed it, and use it in loss
    y = keras.Input(shape=(10,))
    # wrap in cleverhans Model
    mmm = Dummy(mm)

    adv = attack_fn(mmm, x, y)
    # IMPORTANT: this should be pre_softmax, thus should be mm instead of
    # model. The differences is the accuracy improves slowly and cannot reach a
    # high value.
    adv_logits = mm(adv)
    adv_xent = my_softmax_xent(logits=adv_logits, labels=y)
    # I also need the nat loss
    nat_xent = my_softmax_xent(logits=mm(x), labels=y)

    # DEBUG using adv_xent is tricky to converge, lr=1e-4 does, lr=1e-3 doesn't
    #
    # loss = adv_xent + nat_xent
    loss = adv_xent

    advacc = my_accuracy_wrapper(logits=adv_logits, labels=y)
    return mm, (x, y, adv, loss, advacc)

def advtrain(model, data):
    xs, ys = data
    model, (x,y,_,loss, advacc) = wrap_cleverhans(model, PGD)

    def advacc_fn(a,b):
        return advacc

    model.compile(
        loss=lambda a,b: loss,
        # IMPORTANT 1e-3 does not converge!!! This is the root cause of
        # performance difference with Madry when using only adv training
        # data. Setting Madry's lr=1e-3 won't converge as well. Thus for stable
        # and fast training, I would probably just use both clean and adv data.
        optimizer=Adam(1e-4),
        metrics=['accuracy', advacc_fn],
        target_tensors=y)
    model.fit(xs, ys, epochs=5, batch_size=50)

def evaluate_attack(model, attack_fn, ds):
    npx, npy = ds
    mm, (x, y, adv, loss, _) = wrap_cleverhans(model, attack_fn)

    print('attacking 10 samples ..')
    npadv = K.get_session().run(adv, feed_dict={x: npx[:10], y: npy[:10]})
    print('sample:')
    sample_and_view(npadv)
    nppred = model.predict(npadv)
    print('preds: ', np.argmax(nppred, 1))
    print('labels:', np.argmax(npy[:10], 1))

    batch_size = 64
    nbatch = npx.shape[0] // batch_size

    correct = 0
    total = 0
    clear_tqdm()
    for j in tqdm(range(nbatch)):
        start = j * batch_size
        end = (j+1) * batch_size
        batch_x = npx[start:end]
        batch_y = npy[start:end]

        npadv = K.get_session().run(adv, feed_dict={x: batch_x, y: batch_y})
        logits = mm.predict(npadv)
        correct += np.sum(np.argmax(logits, 1) == np.argmax(batch_y, 1))
        total += logits.shape[0]

        if j % 20 == 0:
            print('current acc:', correct / total)
    print('Acc:', correct / total)

def do_evaluate_attack(model, dl):
    print('evaluting clean model ..')
    evaluate_attack(model, lambda m,x,y: x, dl)
    print('evaluating FGSM ..')
    evaluate_attack(model, FGSM, dl)
    print('evaluating PGD ..')
    evaluate_attack(model, PGD, dl)

def test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    # cnn = get_LeNet5()
    cnn = get_Madry_model()

    train_CNN(cnn, (train_x, train_y))

    cnn.evaluate(test_x, test_y)


    advtrain(cnn, (train_x, train_y))

    evaluate_attack(cnn, FGSM, (test_x, test_y))
    evaluate_attack(cnn, PGD, (test_x, test_y))
    do_evaluate_attack(cnn, (test_x, test_y))

    evaluate_attack(cnn, PGD, (train_x[:1000], train_y[:1000]))

    ae = CNN_AE()
    test_AE(ae, test_x, test_y)
    train_AE(ae, train_x)

    # FIXME would this work? Seems to work but the cnn weights also seem to change
    cnn.trainable = False
    ae.trainable = True
    advtrain(Sequential(ae.layers + cnn.layers), (train_x, train_y))

    evaluate_attack(Sequential(ae.layers + cnn.layers), PGD, (test_x, test_y))
