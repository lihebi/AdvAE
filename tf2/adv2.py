import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Softmax, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

from cleverhans.future.tf2.attacks import fast_gradient_method, projected_gradient_descent, spsa

from tqdm import tqdm
import time

from data_utils import load_mnist, sample_and_view
from model import get_Madry_model, get_LeNet5, dense_AE, train_AE, train_CNN
from model import get_LeNet5_functional

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
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    clear_tqdm()
    for step, (x, y) in enumerate(tqdm(ds)):
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
    # TODO use Dataset
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


def custom_advtrain(model, ds, valid_ds):
    batch_size = 64
    tqdm_total = len(ds[0]) // batch_size
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)

    vx, vy = valid_ds
    vx = vx[:50]
    vy = vy[:50]

    opt = Adam(0.003)

    attack_fn = PGD
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    train_advacc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    acc_fn = tf.keras.metrics.categorical_accuracy

    # FIXME NOW IMPORTANT why my previous keras training loop achieve high
    # validation advacc so quickly? Why previous advae converges so quickly?
    for epoch in range(3):
        clear_tqdm()
        for i, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
            adv = attack_fn(model, x, y)
            with tf.GradientTape() as tape:
                adv_logits = model(adv)
                adv_loss = loss_fn(y, adv_logits)

                nat_logits = model(x)
                nat_loss = loss_fn(y, nat_logits)

                # FIXME NOW IMPORTANT why I have to add nat_loss? The original
                # adversarial training does not need that
                loss = adv_loss + nat_loss
                # loss = adv_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            # accumulate acc results
            train_acc_metric.update_state(y, nat_logits)
            train_advacc_metric.update_state(y, adv_logits)
            if i % 20 == 0:
                print('')
                print('step {}, loss: {:.5f}'.format(i, loss))
                nat_acc = train_acc_metric.result()
                adv_acc = train_advacc_metric.result()
                print('nat acc: {:.5f}, adv acc: {:.5f}'.format(nat_acc, adv_acc))
                # reset here because I don't want the train loss to be delayed so much
                train_acc_metric.reset_states()
                train_advacc_metric.reset_states()
                # I also want to monitor the validation accuracy
                v_logits = model(vx)
                valid_nat_acc = acc_fn(v_logits, vy)
                adv_vx = attack_fn(model, vx, vy)
                v_logits = model(adv_vx)
                valid_adv_acc = acc_fn(v_logits, vy)
                print('valid nat acc: {:.5f}, valid adv acc: {:.5f}'
                      .format(valid_nat_acc.numpy().mean(), valid_adv_acc.numpy().mean()))

        # Display metrics at the end of each epoch.
        nat_acc = train_acc_metric.result()
        adv_acc = train_advacc_metric.result()
        print('nat acc: {:.5f}, adv acc: {:.5f}'.format(nat_acc, adv_acc))
        train_acc_metric.reset_states()
        train_advacc_metric.reset_states()

        # TODO Run a validation loop at the end of each epoch.
        # for x_batch_val, y_batch_val in val_dataset:
        #     val_logits = model(x_batch_val)
        #     # Update val metrics
        #     val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        # print('Validation acc: %s' % (float(val_acc),))


def test():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist()
    sample_and_view(train_x)

    cnn = get_LeNet5()
    cnn = get_Madry_model()

    custom_train(cnn, (train_x, train_y))
    evaluate_model(cnn, (test_x, test_y))

    custom_advtrain(cnn, (train_x, train_y), (val_x, val_y))
    do_evaluate_attack(cnn, (test_x, test_y))
    # on training data
    do_evaluate_attack(cnn, (train_x[:1000], train_y[:1000]))

if __name__ == '__main__':
    tfinit()
    test()
