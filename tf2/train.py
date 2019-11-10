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

from data_utils import load_mnist, sample_and_view
from model import get_Madry_model, get_LeNet5, dense_AE, CNN_AE, train_AE, train_CNN, test_AE


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


def custom_train(model, opt, ds):
    # FIXME this should be in data
    assert isinstance(ds, tf.data.Dataset)
    # FIXME opt should be created elsewhere
    # opt = Adam(0.001)

    # TODO summary writer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(3):
        for i, (x, y) in enumerate(ds):
            with tf.GradientTape() as tape:
                with K.learning_phase_scope(1):
                    logits = model(x)
                loss = loss_fn(y, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

            train_acc_metric.update_state(y, logits)
            if i % 20 == 0:
                print('step {}, loss: {:.5f}'.format(i, loss))
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

def free_train_stepS(model, opt, loss_fn, x, y, metrics):
    # configuration
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


def train_step(model, params, opt, attack_fn, loss_fn, mixing_fn,
               x, y,
               metrics):
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

        # FIXME if mixing=0, will there be extra computation cost?
        # TODO I want to monitor nat/adv loss and this lambda value
        λ = mixing_fn(nat_loss, adv_loss)

        # FIXME sum of ratio should equal to 1?
        # loss = (1-λ) * adv_loss + λ * nat_loss
        loss = adv_loss + λ * nat_loss
        # loss = adv_loss + nat_loss
        # loss = adv_loss
    # apply gradients
    # params = model.trainable_variables
    gradients = tape.gradient(loss, params)
    opt.apply_gradients(zip(gradients, params))
    # metrics
    metrics['train_time'].update_state(time.time() - t)
    # DEBUG using adv_loss instead of combined loss
    metrics['train_loss'].update_state(adv_loss)
    # TODO log lambda value
    metrics['train_acc'].update_state(y, adv_logits)

def test_step(model, attack_fn, loss_fn, x, y, metrics):
    t = time.time()

    with K.learning_phase_scope(0)
        adv = attack_fn(model, x, y)
        adv_logits = model(adv)
        adv_loss = loss_fn(y, adv_logits)

    metrics['test_loss'].update_state(adv_loss)
    metrics['test_acc'].update_state(y, adv_logits)
    metrics['test_time'].update_state(time.time() - t)

def train_metric_step(step, metrics):
    print('[step {}] loss: {}, acc: {}'.format(step,
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

def create_ckpt(model, opt, config):
    # FIXME keyword arguments names are random?
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, model=model)
    # use logname as path
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts/' + config['logname'], max_to_keep=3)
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

def create_summary_writers(config):
    train_log_dir = 'logs/gradient_tape/' + config['logname'] + '/train'
    test_log_dir = 'logs/gradient_tape/' + config['logname'] + '/test'
    # This will append instead of overwrite
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer

def custom_advtrain(model, params, train_ds, test_ds, config):
    # TODO different batch size
    batch_size = config['batch_size']
    if isinstance(train_ds, tf.data.Dataset):
        ds = train_ds
    else:
        ds = tf.data.Dataset.from_tensor_slices(train_ds)\
                            .shuffle(buffer_size=1024)\
                            .batch(batch_size)\
                            .repeat()


    # TODO learning rate decay?
    opt = Adam(config['lr'])

    ckpt, manager = create_ckpt(model, opt, config)

    attack_fn = PGD
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metrics = create_metrics()

    train_summary_writer, test_summary_writer = create_summary_writers(config)

    for x,y in ds:
        if int(ckpt.step) > config['max_steps']:
            print('Finished {} steps larger than max step {}. Returning ..'
                  .format(ckpt.step, config['max_steps']))
            return
        if 'free' in config['logname']:
            # CAUTION this updates opt model m=30 times, so technically it is m
            # steps. However, the running time cost is one step
            free_train_stepS(model, opt, loss_fn, x, y, metrics)
            ckpt.step.assign_add(1)
        else:
            # about 40 seconds/100 step = 0.4
            train_step(model, params, opt,
                       attack_fn, loss_fn, config['mixing_fn'],
                       x, y,
                       metrics)
            ckpt.step.assign_add(1)


        # FIXME configure all the number of steps
        if int(ckpt.step) % config['num_train_summary_step'] == 0:
            with train_summary_writer.as_default():
                train_metric_step(int(ckpt.step), metrics)

        if int(ckpt.step) % config['num_test_summary_step'] == 0:
            # about 20 seconds
            print('performing test phase ..')
            # FIXME move batch size into configuration (and log name)
            for x,y in tf.data.Dataset.from_tensor_slices(test_ds).batch(config['batch_size'] * 2):
                test_step(model, attack_fn, loss_fn,
                          x, y,
                          metrics)

            with test_summary_writer.as_default():
                test_metric_step(int(ckpt.step), metrics)
        # save check point at the end, to make sure the above branches are
        # always performed completely. If interrupted, it will get re-performed.
        if int(ckpt.step) % config['num_ckpt_step'] == 0:
            # FIXME the dataset status is not saved
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


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
