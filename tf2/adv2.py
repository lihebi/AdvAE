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

from data_utils import load_mnist, sample_and_view
from model import get_Madry_model, get_LeNet5, dense_AE, CNN_AE, train_AE, train_CNN, test_AE


def clear_tqdm():
    if '_instances' in dir(tqdm):
        tqdm._instances.clear()

def tfinit():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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


def custom_train(model, ds):
    batch_size = 64
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    opt = Adam(0.001)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(3):
        clear_tqdm()
        # tqdm cannot show bar for tf.data.Dataset, because it has no len()
        for i, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

            train_acc_metric.update_state(y, logits)
            if i % 200 == 0:
                print('')
                print('step {}, loss: {:.5f}'.format(i, loss))
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # TODO Run a validation loop at the end of each epoch.
        # for x_batch_val, y_batch_val in val_dataset:
        #     val_logits = model(x_batch_val)
        #     # Update val metrics
        #     val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        # print('Validation acc: %s' % (float(val_acc),))

def test_time():
    t = time.time()
    time.sleep(1)
    time.time() - t

def train_step(model, params, opt, attack_fn, loss_fn, mixing_fn,
               x, y,
               train_loss, train_acc, train_time):
    t = time.time()
    adv = attack_fn(model, x, y)
    with tf.GradientTape() as tape:
        adv_logits = model(adv)
        adv_loss = loss_fn(y, adv_logits)

        nat_logits = model(x)
        nat_loss = loss_fn(y, nat_logits)

        # FIXME if mixing=0, will there be extra computation cost?
        # TODO I want to monitor nat/adv loss and this lambda value
        λ = mixing_fn(nat_loss, adv_loss)
        loss = adv_loss + λ * nat_loss
        # loss = adv_loss + nat_loss
        # loss = adv_loss
    # apply gradients
    # params = model.trainable_variables
    gradients = tape.gradient(loss, params)
    opt.apply_gradients(zip(gradients, params))
    # metrics
    train_time.update_state(time.time() - t)
    train_loss.update_state(loss)
    train_acc.update_state(y, adv_logits)

def test_step(model, attack_fn, loss_fn, x, y, loss_metric, acc_metric):
    adv = attack_fn(model, x, y)
    adv_logits = model(adv)
    adv_loss = loss_fn(y, adv_logits)

    loss_metric.update_state(adv_loss)
    acc_metric.update_state(y, adv_logits)


def custom_advtrain(model, params, ds, test_ds,
                    lr=1e-3, mixing_fn=lambda nat, adv: 0, logname=''):
    # TODO different batch size
    batch_size = 50
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size).repeat()

    # TODO learning rate decay?
    opt = Adam(lr)

    attack_fn = PGD
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_acc = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
    train_time = tf.keras.metrics.Sum('train time', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_acc = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
    test_time = tf.keras.metrics.Sum('test time', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # add meaningful experimental setting in name
    train_log_dir = 'logs/gradient_tape/' + logname + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + logname + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for i, (x,y) in enumerate(ds):
        if i > 3000:
            print('Finished {} steps. Returning ..'.format(i))
            return
        # about 40 seconds/100 step = 0.4
        train_step(model, params, opt,
                   attack_fn, loss_fn, mixing_fn,
                   x, y,
                   train_loss, train_acc, train_time)

        if i % 20 == 0:
            print('[step {}] loss: {}, acc: {}'.format(i, train_loss.result(), train_acc.result()))
        # FIXME about 0.0035, so I'm just logging all data
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i)
            tf.summary.scalar('accuracy', train_acc.result(), step=i)
            tf.summary.scalar('time', train_time.result(), step=i)
        train_loss.reset_states()
        train_acc.reset_states()


        if i % 200 == 0:
            t = time.time()
            # about 20 seconds
            print('performing test phase ..')
            for x,y in tf.data.Dataset.from_tensor_slices(test_ds).batch(500):
                test_step(model, attack_fn, loss_fn,
                          x, y, test_loss, test_acc)
            test_time.update_state(time.time() - t)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=i)
                tf.summary.scalar('accuracy', test_acc.result(), step=i)
                # used only to monitor the time wasted
                tf.summary.scalar('time', test_time.result(), step=i)
            print('[test in step {}] loss: {}, acc: {}'.format(i, test_loss.result(), test_acc.result()))
            test_loss.reset_states()
            test_acc.reset_states()
    return


def exp_all():
    mixing_fns = {
        'f0': lambda nat, adv: 0,
        # simply mixing
        'f1': lambda nat, adv: 1,
        # FIXME (σ(nat) - 0.5) * 2
        'σ(nat)': lambda nat, adv: tf.math.sigmoid(nat),
        # TODO linear. This function is essentially nat^2
        'nat': lambda nat, adv: nat,
        # relative value
        # FIXME will this be slow to compute gradient?
        # TODO σ(nat) * natloss + σ(adv) * advloss?
        # FIXME divide by zero
        # FIXME this should not be stable
        'σ(nat/adv)': lambda nat, adv: tf.math.sigmoid(nat / adv)
        # TODO functions other than σ?
        # TODO use training iterations? This should be a bad idea.
    }

    lrs = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]

    for fname in mixing_fns:
        for lr in lrs:
            # logname = '{}-{:.0e}'.format(fname, lr)
            logname = '{}-{}'.format(fname, lr)
            print('Exp:', logname)
            fn = mixing_fns[fname]
            # TODO set a step limit
            exp_train(fn, lr, logname)

def exp_train(mixing_fn, lr, logname):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)
    cnn = get_Madry_model()
    cnn(train_x[:10])
    custom_advtrain(cnn, cnn.trainable_variables,
                    (train_x, train_y), (test_x, test_y),
                    lr=lr, mixing_fn=mixing_fn, logname=logname)



def test():
    # FIXME remove validation data
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
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
                    (train_x, train_y), (val_x, val_y))

    evaluate_model(cnn, (test_x, test_y))

    evaluate_model(tf.keras.Sequential([ae, cnn]), (test_x, test_y))
    do_evaluate_attack(tf.keras.Sequential([ae, cnn]), (test_x, test_y))


if __name__ == '__main__':
    tfinit()
    # test()
    exp_all()
