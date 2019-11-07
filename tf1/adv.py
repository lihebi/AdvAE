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

from model import get_Madry_model, get_LeNet5, dense_AE, train_AE, train_CNN

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2

def clear_tqdm():
    if '_instances' in dir(tqdm):
        tqdm._instances.clear()

def FGSM(model, x, y=None, params=dict()):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    fgsm_params.update(params)
    # FIXME Cleverhans has a bug in Attack.get_or_guess_labels: it
    # only checks y is in the key, but didn't check whether the value
    # is None. fgsm.generate calls that function
    if y is not None:
        fgsm_params.update({'y': y})
    fgsm = FastGradientMethod(model)
    adv_x = fgsm.generate(x, **fgsm_params)
    # return tf.stop_gradient(adv_x)
    return adv_x

def PGD(model, x, y=None, params=dict()):
    # DEBUG I would always want to supply the y manually
    # assert y is not None

    # FIXME cleverhans requires the model to be instantiated from
    # cleverhans.Model class. This is ugly. I resist to do that. The tf2
    # implementation fixed that, so I'd try tf2 version first. Stopping here
    pgd = ProjectedGradientDescent(model)
    pgd_params = {'eps': 0.3,
                  # CAUTION I need this, otherwise the adv acc data is
                  # not accurate
                  'y': y,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': 0.,
                  'clip_max': 1.}
    pgd_params.update(params)
    adv_x = pgd.generate(x, **pgd_params)
    # DEBUG do I need to stop gradients here?
    # return tf.stop_gradient(adv_x)
    return adv_x

def do_FGSM(sess, model, dx, dy):
    x, y = model.inputs[0], model.outputs[0]
    # FIXME NOW Again, this needs to inherit from cleverhans Model
    adv = FGSM(model, x, y)
    adv_np = sess.run(adv, feed_dict={x: dx, y: dy})
    loss, acc = model.evaluate(adv_np, dy)
    print('acc:', acc)

def get_wrapperd_adv_model(model):
    # FIXME this does not work
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
    # perform PGD
    adv = PGD(mmm, x, y)
    # FIXME this throws error
    res = keras.models.Model(x, mm(adv))
    # FIXME I'll probably need to compose a loss function?
    return res

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
    # perform PGD
    adv = attack_fn(mmm, x, y)
    # compute loss
    adv_logits = model(adv)
    adv_xent = my_softmax_xent(logits=adv_logits, labels=y)
    # I also need the nat loss
    nat_xent = my_softmax_xent(logits=model(x), labels=y)
    # loss = adv_xent + nat_xent
    # DEBUG
    loss = adv_xent
    return mm, (x, y, adv, loss)

def my_sigmoid_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))
def my_l2loss(x1, x2):
    return tf.reduce_mean(tf.square(x1-x2))

def my_softmax_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))

def _advmodel(model):
    # I'll need to use functional API
    x = keras.Input(shape=(28,28,1))
    adv = PGD(model, x)
    # FIXME maybe no need to stop gradients?
    adv = tf.stop_gradient(adv)
    y = model(adv)
    return keras.Model(x, y)

def advtrain(model, data):
    xs, ys = data
    # need a custom loss function
    # If using tf1, basically two ways:
    # 1. I need to have a very special training loop, that generate adv examples in the loop
    # 2. I need to wrap adv generation as a tensor, stop gradients, and use that in loss function
    #
    # generate adv

    # model = _advmodel(model)
    # model = get_wrapperd_adv_model(model)

    model, (x,y,_,loss) = wrap_cleverhans(model, PGD)

    model.compile(
        loss=lambda a,b: loss,
        # loss=keras.losses.CategoricalCrossentropy(),
        # loss='categorical_crossentropy',
        optimizer=Adam(0.001),
        # TODO I also want valid adv acc
        metrics=['accuracy'],
        target_tensors=y)
    model.fit(xs, ys, epochs=5, batch_size=32)

def evaluate_model(model, ds):
    # Prepare the training dataset.
    batch_size = 64
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    clear_tqdm()
    for step, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
        logits = model(x)
        acc_metric.update_state(y, logits)
        if step % 200 == 0:
            acc = acc_metric.result()
            print('Acc so far: %s' % (float(acc),))
    acc = acc_metric.result()
    # FIXME I don't need to reset
    acc_metric.reset_states()
    print('Acc of all data: %s' % (float(acc),))

def evaluate_attack(model, attack_fn, ds):
    npx, npy = ds
    mm, (x, y, adv, loss) = wrap_cleverhans(model, attack_fn)

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

    cnn = get_LeNet5()
    # cnn = get_Madry_model()

    train_CNN(cnn, (train_x, train_y))

    # cnn.predict(val_x).shape
    cnn.evaluate(test_x, test_y)

    advtrain(cnn, (train_x, train_y))

    _, (x,y,adv,loss) = wrap_with_loss(cnn)

    sess = keras.backend.get_session()
    # TODO batch and get all adv
    adv_x = sess.run(adv, feed_dict={x: test_x[:10], y: test_y[:10]})
    sample_and_view(adv_x)
    cnn.compile(optimizer=Adam(0.001),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    cnn.evaluate(adv_x, test_y[:10])
    np.argmax(cnn.predict(adv_x), 1)
    np.argmax(test_y[:10], 1)

    evaluate_attack(cnn, FGSM, (test_x, test_y))
    evaluate_attack(cnn, PGD, (test_x, test_y))
    evaluate_attack(cnn, PGD, (train_x[:1000], train_y[:1000]))
