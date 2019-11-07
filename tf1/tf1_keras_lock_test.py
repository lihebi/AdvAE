# This is kind of a MWE for the bug of tf.keras with old cleverhans API. The
# cause seems to be the tf.while loop. Using tf.keras will block at that while
# loop for some lock.

# Thus, I'm not going to continue develop on tf1.0 and old cleverhans API,
# because keras is deprecating in favor of tf.keras, and thus I'm not using keras.

import keras
import tensorflow as tf
import cleverhans
from cleverhans.model import Model
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.attacks import ProjectedGradientDescent, SaliencyMapMethod
import tensorflow as tf

class Dummy(cleverhans.model.Model):
    def __init__(self, model):
        self.model = model
        # FIXME are these shared layers?
        # self.presoftmax = keras.Sequential(model.layers[:-1])
    def predict(self, x):
        return self.model(x)
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: logits}

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
    return tf.stop_gradient(adv_x)

def test_keras_success():
    print('=== This should run through.')
    x = keras.Input(shape=(28,28,1))
    y = keras.layers.Flatten()(x)
    y = keras.layers.Dense(10)(y)
    m = keras.models.Model(x, y)
    mm = Dummy(m)
    adv = PGD(mm, x)
    adv = tf.stop_gradient(adv)
    # y = model(adv)

def test_keras_block():
    print('=== This should block forever on a lock.')
    x = tf.keras.Input(shape=(28,28,1))
    y = tf.keras.layers.Flatten()(x)
    y = tf.keras.layers.Dense(10)(y)
    m = tf.keras.models.Model(x, y)
    mm = Dummy(m)
    adv = PGD(mm, x)
    adv = tf.stop_gradient(adv)
    # y = model(adv)

test_keras_success()
test_keras_block()
