import logging
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
MODEL_PATH = os.path.join('models', 'mnist')
TARGETED = True

def load_data():
    # Get MNIST test data
    mnist = MNIST(train_start=0, train_end=60000,
                  test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')
    # Obtain Image Parameters
    # (nrows, ncols, nchannels, nlabels)
    return (x_train, y_train), (x_test, y_test)

def get_cnn_model(nrows, ncols, nchannels, nlabels):
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, nrows, ncols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nlabels))
    nb_filters = 64
    # Define TF model graph
    model = ModelBasicCNN('model1', nlabels, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")
    return model, loss


def train_cnn_model(model, x_train, y_train):
    # tf.set_random_seed(1234)
    sess = tf.Session()
    # set_log_level(logging.DEBUG)

    (x_train, y_train), (x_test, y_test) = load_data()
    nrows, ncols, nchannels = x_train.shape[1:4]
    nlabels = y_train.shape[1]
    model, loss = get_cnn_model(nrows, ncols, nchannels, nlabels)

    # Train an MNIST model
    train_params = {
        'nb_epochs': NB_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'filename': os.path.split(MODEL_PATH)[-1]
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(MODEL_PATH + ".meta"):
        print('loading from ', MODEL_PATH + ".meta")
        tf_model_load(sess, MODEL_PATH)
    else:
        print('training a new model ..')
        train(sess, loss, x_train, y_train, args=train_params, rng=rng)
        saver = tf.train.Saver()
        saver.save(sess, MODEL_PATH)
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': BATCH_SIZE}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    assert x_test.shape[0] == 10000 - 0, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

def cw_model(model, x_test, y_test):
    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, sess=sess)
    
    assert SOURCE_SAMPLES == nb_classes
    idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0]
            for i in range(nb_classes)]
    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                  nchannels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # adversarial inputs
    adv_inputs = np.array(
        [[instance] * nb_classes for instance in x_test[idxs]],
        dtype=np.float32)

    one_hot = np.zeros((nb_classes, nb_classes))
    one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

    adv_inputs = adv_inputs.reshape(
        (SOURCE_SAMPLES * nb_classes, img_rows, img_cols, nchannels))
    adv_ys = np.array([one_hot] * SOURCE_SAMPLES,
                      dtype=np.float32).reshape((SOURCE_SAMPLES *
                                                 nb_classes, nb_classes))
    yname = "y_target"
    
    cw_params_batch_size = SOURCE_SAMPLES * nb_classes
    cw_params = {'binary_search_steps': 1,
                 yname: adv_ys,
                 'max_iterations': ATTACK_ITERATIONS,
                 'learning_rate': CW_LEARNING_RATE,
                 'batch_size': cw_params_batch_size,
                 'initial_const': 10}

    # generate only based on adv_inputs. The yname is also passed
    # through the params
    adv = cw.generate_np(adv_inputs,
                         **cw_params)
    eval_params = {'batch_size': np.minimum(nb_classes, SOURCE_SAMPLES)}
    adv_accuracy = model_eval(
        sess, x, y, preds, adv, adv_ys, args=eval_params)
    for j in range(nb_classes):
        for i in range(nb_classes):
            grid_viz_data[i, j] = adv[i * nb_classes + j]

    print(grid_viz_data.shape)

    print('--------------------------------------')
    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
    

def __test():
    report = AccuracyReport()


    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1)
    print('Crafting ' + str(SOURCE_SAMPLES) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Close TF session
    sess.close()
    _ = grid_visual(grid_viz_data)
    return report
