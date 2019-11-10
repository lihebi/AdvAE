import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

from cleverhans.future.tf2.attacks import fast_gradient_method, projected_gradient_descent, spsa

from tqdm import tqdm
import time
import datetime

from data_utils import load_mnist, sample_and_view, load_cifar10, load_cifar10_ds
from model import get_Madry_model, get_LeNet5, dense_AE, CNN_AE, train_AE, train_CNN, test_AE

import sys
sys.path.append('./tf_models')
from official.vision.image_classification import resnet_cifar_model

from train import custom_train, custom_advtrain, evaluate_model, evaluate_attack


def keras_train(model, opt, ds, steps_per_epoch, train_epochs):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=(['categorical_accuracy']))
    model.fit(ds,
              epochs=train_epochs,
              steps_per_epoch=train_steps,
              # callbacks=callbacks,
              # validation_steps=num_eval_steps,
              # validation_data=validation_data,
              # validation_freq=flags_obj.epochs_between_evals,
              verbose=1)


# LR_SCHEDULE = [(0.1, 91), (0.01, 136), (0.001, 182)]
# lr_schedule = [(0, 0.1), (40000, 0.01), (60000, 0.001)]
lr_schedule = [(0, 0.1), (10000, 0.01), (20000, 0.001)]

def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate
def test_resnet():
    # (train_x, train_y), (test_x, test_y) = load_cifar10()
    # sample_and_view(train_x)

    batch_size = 128

    # dataset has batch information, and has no len(). Thus, I need to know
    # steps_per_epoch.
    ds, eval_ds, train_steps, test_steps = load_cifar10_ds(batch_size)

    next(iter(ds))[1].shape
    # TensorShape([128, 10])
    next(iter(ds))[0].shape
    # TensorShape([128, 32, 32, 3])
    # so this is a iterable dataset

    boundaries = [10000, 20000]
    values = [0.1, 0.01, 0.001]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    # step = tf.Variable(0, trainable=False)
    # learning_rate = learning_rate_fn(step)

    # FIXME Adam?
    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    # FIXME Evaluation steps in training loop will cause the BN layers
    # statistics to change. In my manual training loop, this should not be
    # handled automatically
    model = resnet_cifar_model.resnet20(training=None)

    # keras training
    keras_train(model, opt, ds, steps_steps, 100)

    # TODO use my own training loop
    # batch 128
    # 390 steps/epoch
    # train_steps
    custom_train(model, opt, ds)

    # num_eval_steps = (cifar_preprocessing.NUM_IMAGES['validation'] //
    #                   batch_size)
    # model.evaluate(eval_ds, steps=num_eval_steps)

    # FIXME I need to recreate model with training=False to evaluate
    evaluate_model(model, eval_ds)


def get_default_config():
    config = {
        'lr': 1e-3,
        'mixing_fn': lambda nat, adv: 0,
        'logname': 'default',

        # TODO run training incrementally
        'max_steps': 2000,

        # CAUTION: batch size will affect the scale of number of steps
        'batch_size': 50,

        # FIXME: setting this to 1000 because dataset state is not restored.
        # 'num_ckpt_step': 1000,
        'num_ckpt_step': 100,

        'num_train_summary_step': 20,
        # NOTE: running full loop of test data evaluation
        'num_test_summary_step': 200,
    }
    return config


def exp_train(config):
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    # a bread new cnn
    cnn = get_Madry_model()
    # dummy call to instantiate weights
    cnn(train_x[:10])
    # do the training
    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y),
                    config)

def test_free():
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    # a bread new cnn
    cnn = get_Madry_model()
    # dummy call to instantiate weights
    cnn(train_x[:10])
    # do the training
    config = get_default_config()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config['logname'] = 'free-{}'.format(current_time)

    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y),
                    config)



def test():
    # FIXME remove validation data
    (train_x, train_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    # DEBUG try lenet5
    # cnn = get_LeNet5()
    cnn = get_Madry_model()

    custom_train(cnn, (train_x, train_y))
    evaluate_model(cnn, (test_x, test_y))

    cnn(train_x[:10])
    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y))

    do_evaluate_attack(cnn, (test_x, test_y))
    # on training data
    do_evaluate_attack(cnn, (train_x[:1000], train_y[:1000]))


    ## AdvAE

    # ae = dense_AE()
    ae = CNN_AE()

    train_AE(ae, train_x)
    test_AE(ae, test_x, test_y)

    custom_advtrain(tf.keras.Sequential([ae, cnn]),
                    ae.trainable_variables,
                    (train_x, train_y), (test_x, test_y))

    evaluate_model(cnn, (test_x, test_y))

    evaluate_model(tf.keras.Sequential([ae, cnn]), (test_x, test_y))
    do_evaluate_attack(tf.keras.Sequential([ae, cnn]), (test_x, test_y))


if __name__ == '__main__':
    tfinit()
    # test()
    exp_all()
    test_free()
