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

import tensorflow.keras.backend as K

from data_utils import load_mnist, sample_and_view, load_mnist_ds, load_cifar10_ds
from model import get_Madry_model, get_LeNet5, dense_AE, CNN_AE, train_AE, train_CNN, test_AE
from utils import tfinit

import sys
sys.path.append('./tf_models')
from official.vision.image_classification import resnet_cifar_model

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

def test_keras_train():
    batch_size = 128
    ds, eval_ds, train_steps, test_steps = load_cifar10_ds(batch_size)
    model = resnet_cifar_model.resnet20(training=None)
    # keras training
    keras_train(model, opt, ds, steps_steps, 100)

def train_metric_step(step, metrics):
    print('[step {}] loss: {:.5f}, acc: {:.5f}'
          .format(step,
                  metrics['train_loss'].result(),
                  metrics['train_acc'].result()))
    # about 0.0035
    tf.summary.scalar('loss', metrics['train_loss'].result(), step=step)
    tf.summary.scalar('accuracy', metrics['train_acc'].result(), step=step)
    tf.summary.scalar('time', metrics['train_time'].result(), step=step)
    metrics['train_loss'].reset_states()
    metrics['train_acc'].reset_states()

def test_metric_step(step, metrics):
    tf.summary.scalar('loss', metrics['test_loss'].result(), step=step)
    tf.summary.scalar('accuracy', metrics['test_acc'].result(), step=step)
    # used only to monitor the time wasted
    tf.summary.scalar('time', metrics['test_time'].result(), step=step)
    print('[test in step {}] loss: {}, acc: {}'.format(step,
                                                       metrics['test_loss'].result(),
                                                       metrics['test_acc'].result()))
    metrics['test_loss'].reset_states()
    metrics['test_acc'].reset_states()

def create_ckpt(model, opt, ID):
    # FIXME keyword arguments names are random?
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, model=model)
    # use logname as path
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts/' + ID, max_to_keep=3)
    # FIXME sanity check for optimizer state restore
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    return ckpt, manager

def create_metrics():
    metrics = {
        'train_loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
        'train_acc': tf.keras.metrics.CategoricalAccuracy('train_accuracy'),
        'train_time': tf.keras.metrics.Sum('train time', dtype=tf.float32),
        'test_loss': tf.keras.metrics.Mean('test_loss', dtype=tf.float32),
        'test_acc': tf.keras.metrics.CategoricalAccuracy('test_accuracy'),
        'test_time': tf.keras.metrics.Sum('test time', dtype=tf.float32)}
    return metrics

def create_summary_writers(ID):
    train_log_dir = 'logs/gradient_tape/' + ID + '/train'
    test_log_dir = 'logs/gradient_tape/' + ID + '/test'
    # This will append instead of overwrite
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer

def FGSM(model, x, y):
    adv = fast_gradient_method(model, x,
                               0.3, np.inf,
                               clip_min=0., clip_max=1.,
                               y=np.argmax(y, 1))
    return adv

def PGD(model, x, y):
    # FIXME rand_init rand_minmax
    return projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf,
                                      clip_min=0., clip_max=1.,
                                      y=np.argmax(y, 1),
                                      # FIXME this is throwing errors
                                      sanity_checks=False)
def clean_train_step(model, params, opt, loss_fn, x, y, metrics):
    t = time.time()
    with tf.GradientTape() as tape:
        with K.learning_phase_scope(1):
            logits = model(x)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, params)
    opt.apply_gradients(zip(gradients, params))
    metrics['train_time'].update_state(time.time() - t)
    metrics['train_acc'].update_state(y, logits)
    metrics['train_loss'].update_state(loss)

def clean_test_step(model, loss_fn, x, y, metrics):
    t = time.time()

    with K.learning_phase_scope(0):
        logits = model(x)
        loss = loss_fn(y, logits)

    metrics['test_loss'].update_state(loss)
    metrics['test_acc'].update_state(y, logits)
    metrics['test_time'].update_state(time.time() - t)

def get_adv_train_step():
    attack_fn = PGD
    def step_fn(model, params, opt,
                loss_fn, x, y, metrics):
        t = time.time()
        # FIXME Do not collect statistics during attack? Or maybe do, because we're
        # using the same minibatch
        with K.learning_phase_scope(0):
            adv = attack_fn(model, x, y)
        with tf.GradientTape() as tape:
            with K.learning_phase_scope(1):
                adv_logits = model(adv)
                nat_logits = model(x)
            adv_loss = loss_fn(y, adv_logits)
            nat_loss = loss_fn(y, nat_logits)
            λ = 1
            loss = adv_loss + λ * nat_loss
        gradients = tape.gradient(loss, params)
        opt.apply_gradients(zip(gradients, params))
        metrics['train_time'].update_state(time.time() - t)
        metrics['train_loss'].update_state(adv_loss)
        metrics['train_acc'].update_state(y, adv_logits)
    return step_fn

def get_adv_test_step():
    attack_fn = PGD
    def step_fn(model, loss_fn, x, y, metrics):
        t = time.time()

        with K.learning_phase_scope(0):
            adv = attack_fn(model, x, y)
            adv_logits = model(adv)
            adv_loss = loss_fn(y, adv_logits)

        metrics['test_loss'].update_state(adv_loss)
        metrics['test_acc'].update_state(y, adv_logits)
        metrics['test_time'].update_state(time.time() - t)
    return step_fn

def free_train_stepS(model, params, opt, loss_fn, x, y, metrics):
    # CAUTION this updates opt model m=30 times, so technically it is m
    # steps. However, the running time cost is one step
    m = 30
    step_size = 0.01
    ε = 0.3
    # initial
    δ = 0
    adv = x
    for i in range(m):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(adv)
            with K.learning_phase_scope(1):
                logits = model(adv)
            loss = loss_fn(logits, y)
        # FIXME this gradient computation might be slow. Consider using tf.function to wrap it
        # update model
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        # update adv
        x_grads = tape.gradient(loss, adv)
        # FIXME this should not be step_size
        δ += tf.sign(x_grads) * ε
        # δ += tf.sign(x_grads) * step_size
        δ = tf.clip_by_value(δ, -ε, ε)
        # this will create new tensors
        adv = tf.clip_by_value(x + δ, 0, 1)

        metrics['train_loss'].update_state(loss)
        # CAUTION FIXME since we do not have a attack here, we don't have
        # training adv accuracy. We can only get validation accuracy
        metrics['train_acc'].update_state(y, logits)
        # FIXME I should not have to del this explicitly
        del tape

def custom_train(model, params, opt,
                 train_step_fn, test_step_fn,
                 ds, test_ds,
                 ID, config):
    """
    FIXME params = model.trainable_variables
    TODO optimizer
    """
    assert isinstance(ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    # FIXME opt should be created elsewhere
    # opt = Adam(0.001)

    ckpt, manager = create_ckpt(model, opt, ID)
    train_summary_writer, test_summary_writer = create_summary_writers(ID)
    metrics = create_metrics()

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for x, y in ds:
        if int(ckpt.step) > config['max_steps']:
            print('Finished {} steps larger than max step {}. Returning ..'
                  .format(ckpt.step, config['max_steps']))
            return
        train_step_fn(model, params, opt, loss_fn, x, y, metrics)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % config['num_train_summary_step'] == 0:
            with train_summary_writer.as_default():
                train_metric_step(int(ckpt.step), metrics)
        if int(ckpt.step) % config['num_test_summary_step'] == 0:
            # about 20 seconds
            print('performing test phase ..')
            # TODO since test_ds runs only once, add tqdm support
            for x,y in test_ds:
                test_step_fn(model, loss_fn, x, y, metrics)
            with test_summary_writer.as_default():
                test_metric_step(int(ckpt.step), metrics)
        # save check point at the end, to make sure the above branches are
        # always performed completely. If interrupted, it will get re-performed.
        #
        # FIXME if interrupted, summary steps might be re-run, and will have overlap
        if int(ckpt.step) % config['num_ckpt_step'] == 0:
            # FIXME the dataset status is not saved
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


def mnist_train(model, ID, config):
    batch_size = config['batch_size']
    ds, eval_ds, train_steps, test_steps = load_mnist_ds(batch_size)
    # FIXME get from config, or related to batch_size
    opt = Adam(1e-3)
    custom_train(model, model.trainable_variables, opt,
                 clean_train_step, clean_test_step,
                 ds, eval_ds,
                 ID, config)

def mnist_advtrain(model, ID, config):
    batch_size = config['batch_size']
    ds, eval_ds, train_steps, test_steps = load_mnist_ds(batch_size)
    # FIXME 1e-3 seems to work as well this time
    opt = Adam(1e-4)
    custom_train(model, model.trainable_variables, opt,
                 get_adv_train_step(), get_adv_test_step(),
                 ds, eval_ds,
                 ID, config)

def cifar10_train(model, ID, config):
    batch_size = config['batch_size']
    ds, eval_ds, train_steps, test_steps = load_cifar10_ds(batch_size)

    # LR_SCHEDULE = [(0.1, 91), (0.01, 136), (0.001, 182)]
    # lr_schedule = [(0, 0.1), (40000, 0.01), (60000, 0.001)]
    # lr_schedule = [(0, 0.1), (10000, 0.01), (20000, 0.001)]
    boundaries = [10000, 20000]
    values = [0.1, 0.01, 0.001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    # step = tf.Variable(0, trainable=False)
    # learning_rate = learning_rate_fn(step)

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    custom_train(model, model.trainable_variables, opt,
                 clean_train_step, clean_test_step,
                 ds, eval_ds,
                 ID, config)

def cifar10_advtrain(model, ID, config):
    batch_size = config['batch_size']
    ds, eval_ds, train_steps, test_steps = load_cifar10_ds(batch_size)

    boundaries = [10000, 20000]
    values = [0.1, 0.01, 0.001]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    custom_train(model, model.trainable_variables, opt,
                 get_adv_train_step(), get_adv_test_step(),
                 ds, eval_ds,
                 ID, config)

def evaluate_model(model, ds):
    # Prepare the training dataset.
    batch_size = 64
    # FIXME size of dataset
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    clear_tqdm()
    for step, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
        # in evaluation mode, set training=False
        logits = model(x, training=False)
        acc_metric.update_state(y, logits)
        if step % 200 == 0:
            acc = acc_metric.result()
            print('Acc so far: %s' % (float(acc),))
    acc = acc_metric.result()
    # FIXME I don't need to reset
    acc_metric.reset_states()
    print('Acc of all data: %s' % (float(acc),))

def evaluate_attack(model, attack_fn, ds):
    x, y = ds
    print('attacking 10 samples ..')
    x_adv = attack_fn(model, x[:10], y[:10])
    print('sample:')
    sample_and_view(x_adv)
    print('labels:', np.argmax(model(x_adv), 1))
    # print('evaluating all models')
    # evaluate_model(model, (x_adv, y))

    batch_size = 64
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    clear_tqdm()
    for step, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
        adv = attack_fn(model, x, y)
        logits = model(adv)
        acc_metric.update_state(y, logits)
        if step % 20 == 0:
            acc = acc_metric.result()
            print('Acc so far: %s' % (float(acc),))
    acc = acc_metric.result()
    # FIXME I don't need to reset
    acc_metric.reset_states()
    print('Acc of all data: %s' % (float(acc),))

def do_evaluate_attack(model, dl):
    print('evaluting clean model ..')
    evaluate_attack(model, lambda m,x,y: x, dl)
    print('evaluating FGSM ..')
    evaluate_attack(model, FGSM, dl)
    print('evaluating PGD ..')
    evaluate_attack(model, PGD, dl)


def get_default_config():
    return {
        # 'lr': 1e-3,
        # 'mixing_fn': lambda nat, adv: 0,

        'max_steps': 5000,

        # CAUTION: batch size will affect the scale of number of steps
        'batch_size': 50,

        # FIXME: setting this to 1000 because dataset state is not restored.
        # 'num_ckpt_step': 1000,
        'num_ckpt_step': 100,

        'num_train_summary_step': 20,
        # NOTE: running full loop of test data evaluation
        'num_test_summary_step': 200}

def test_mnist():
    # batch_size = 50
    # ds, eval_ds, train_steps, test_steps = load_mnist_ds(batch_size)

    tfinit()

    model = get_Madry_model()
    model.build((None,28,28,1))
    config = get_default_config()
    config['batch_size'] = 50

    # ID should be related to:
    # - batch size
    # - learning rate. However, learning rate has schedule, so ...
    # - seed
    mnist_train(model, 'mnist_clean', config)
    mnist_advtrain(model, 'mnist_adv', config)

    # evaluate_model(model, eval_ds)


def test_resnet():
    # batch_size = 128
    # ds, eval_ds, train_steps, test_steps = load_cifar10_ds(batch_size)

    tfinit()

    model = resnet_cifar_model.resnet20(training=None)
    model.build((None,32,32,3))

    config = get_default_config()
    config['batch_size'] = 128

    cifar10_train(model, 'cifar10_clean', config)
    cifar10_advtrain(model, 'cifar10_adv', config)

    # evaluate_model(model, eval_ds)
