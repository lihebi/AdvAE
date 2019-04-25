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

BATCH_SIZE = 128
NUM_EPOCHS = 100

class DenoisingEncoder():
    """Implement a denoising auto encoder.
    """
    def __init__(self):
        self.x = tf.placeholder(shape=self.x_shape(), dtype='float32')
        with tf.variable_scope("encoder"):
            self.code_digits = self.auto_encoder(self.x_image)
            self.reconstructed = tf.nn.sigmoid(self.code_digits)
        # self.code_loss = tf.reduce_mean((self.reconstructed - self.x_image)**2)
        self.code_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                        (labels=self.x_image, logits=self.code_digits))

        self.encoder_variables = tf.trainable_variables()[pre_var_num:]
    def auto_encoder(self, noise_x, clean_x):
        """From noise_x to x."""
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(noise_x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (7, 7, 32)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        # first convolutional layer
        with tf.variable_scope("conv1"):
            h_conv1 = tf.nn.relu(self._conv(input_images, [3, 3, 1, 32], [32]))
        h_pool1 = self._max_pool_2x2(h_conv1)
        # second convolutional layer
        with tf.variable_scope("conv2"):
            h_conv2 = tf.nn.relu(self._conv(h_pool1, [3, 3, 32, 32], [32]))
        encoded = self._max_pool_2x2(h_conv2)

        # at this point the representation is (7, 7, 32)
        with tf.variable_scope("conv3"):
            h_conv3 = tf.nn.relu(self._conv(encoded, [3, 3, 32, 32], [32]))
        upsampled = self._upsample(h_conv3)
        with tf.variable_scope("conv4"):
            h_conv4 = tf.nn.relu(self._conv(upsampled, [3, 3, 32, 32], [32]))
        upsampled2 = self._upsample(h_conv4)
        decoded = self._conv(upsampled2, [3, 3, 32, 1], [1])
        return decoded
        
        pass

class CNNModel(cleverhans.model.Model):
    def __init__(self):
        self.setup_model()
    # abstract interfaces
    def x_shape(self):
        assert(False)
        return None
    def y_shape(self):
        assert(False)
        return None
    def rebuild_model(self):
        assert(False)
        return None
    # general methods
    def setup_model(self):
        shape = self.x_shape()
        # DEBUG I'm trying the input layer vs. placeholders
        self.x = keras.layers.Input(shape=self.x_shape(), dtype='float32')
        self.y = tf.placeholder(shape=self.y_shape(), dtype='float32')
        self.logits = self.rebuild_model(self.x)
        self.preds = tf.argmax(self.logits, axis=1)
        self.probs = tf.nn.softmax(self.logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))
        # FIXME num_classes
        # self.num_classes = np.product(self.y_shape[1:])
        self.ce_grad = tf.gradients(self.cross_entropy(), self.x)[0]
    def cross_entropy_with(self, y):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y))
    def logps(self):
        """Symbolic TF variable returning an Nx1 vector of log-probabilities"""
        return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)
    def predict(self, x):
        return keras.models.Model(self.x, self.logits)(x)
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}


    @staticmethod
    def compute_accuracy(sess, x, y, preds, x_test, y_test):
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
            dtype=tf.float32))
        return sess.run(acc, feed_dict={x: x_test, y: y_test})

    # loss functions. Using different loss function would result in
    # different model to compare
    def certainty_sensitivity(self):
        """Symbolic TF variable returning the gradients (wrt each input
        component) of the sum of log probabilities, also interpretable
        as the sensitivity of the model's certainty to changes in `X`
        or the model's score function

        """
        crossent_w_1 = self.cross_entropy_with(tf.ones_like(self.y) / self.num_classes)
        return tf.gradients(crossent_w_1, self.x)[0]
    def l1_certainty_sensitivity(self):
        """Cache the L1 loss of that product"""
        return tf.reduce_mean(tf.abs(self.certainty_sensitivity()))
    def l2_certainty_sensitivity(self):
        """Cache the L2 loss of that product"""
        return tf.nn.l2_loss(self.certainty_sensitivity())
    def cross_entropy(self):
        """Symbolic TF variable returning information distance between the
        model's predictions and the true labels y

        """
        return self.cross_entropy_with(self.y)
    def cross_entropy_grads(self):
        """Symbolic TF variable returning the input gradients of the cross
        entropy.  Note that if you pass in y=(1^{NxK})/K, this returns
        the same value as certainty_sensitivity.

        """
        # return tf.gradients(self.cross_entropy(), self.x)[0]
        # FIXME why [0]
        return tf.gradients(self.cross_entropy(), self.x)

    def l1_double_backprop(self):
        """L1 loss of the sensitivity of the cross entropy"""
        return tf.reduce_sum(tf.abs(self.cross_entropy_grads()))

    def l2_double_backprop(self):
        """L2 loss of the sensitivity of the cross entropy"""
        # tf.sqrt(tf.reduce_mean(tf.square(grads)))
        return tf.nn.l2_loss(self.cross_entropy_grads())
    def gradients_group_lasso(self):
        grads = tf.abs(self.cross_entropy_grads())
        return tf.reduce_mean(tf.sqrt(
            keras.layers.AvgPool3D(4, 4, 'valid')(tf.square(grads))
            + 1e-10))

    # training
    def evaluate_np(self, sess, test_x, test_y):
        return sess.run([accuracy()],
                        feed_dict={self.x: test_x, self.y: test_y})
    def predict_np(self, sess, test_x):
        # FIXME can I just supply only one placeholder?
        return sess.run(self.preds, feed_dict={self.x: test_x})

    def save(self):
        raise NotImplementedError()
        pass
    def load(self):
        raise NotImplementedError()
        pass
    # attacks
    def my_FGSM(self, sess, x, y, eps):
        clip_min = np.min(x)
        clip_max = np.max(x)
        feed = { self.x: x, self.y: y }
        grads = sess.run(self.cross_entropy_grads(), feed_dict=feed)
        x = np.clip(x + np.sign(grads)*eps, clip_min, clip_max)
        return x
    def my_PGD(self, sess, x, y):
        pass
    def my_jsma(self, sess, x, y):
        pass
    def my_cw(self, sess, x, y):
        pass
    

def my_adv_training(sess, model, loss, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    """Adversarially training the model. Each batch, use PGD to generate
    adv examples, and use that to train the model.

    Print clean and adv rate each time.

    Early stopping when the clean rate is good and adv rate does not
    improve.

    """
    # FIXME refactor this with train.
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    best_loss = math.inf
    best_epoch = 0
    # DEBUG
    PATIENCE = 2
    patience = PATIENCE
    print_interval = 20

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=10)
    for i in range(num_epochs):
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // batch_size
        ct = 0
        for j in range(nbatch):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            clean_dict = {model.x: batch_x, model.y: batch_y}

            # generate adv examples

            # Using cleverhans library to generate PGD intances. This
            # approach is constructing a lot of graph strucutres, and
            # it is not clear what it does. The runnning is quite slow.
            
            # adv_x = my_PGD(sess, model)
            # adv_x_concrete = sess.run(adv_x, feed_dict=clean_dict)

            # Using a homemade PGD adapted from mnist_challenge is
            # much faster. But there is a caveat. See below.
            
            adv_x_concrete = my_fast_PGD(sess, model, batch_x, batch_y)
            
            adv_dict = {model.x: adv_x_concrete, model.y: batch_y}
            # evaluate current accuracy
            clean_acc = sess.run(model.accuracy, feed_dict=clean_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            # actual training
            #
            # clean training turns out to be important when using
            # homemade PGD. Otherwise the training does not
            # progress. Using cleverhans PGD won't need a clean data,
            # and still approaches the clean accuracy fast. However
            # since the running time of cleverhans PGD is so slow, the
            # overall running time is still far worse.
            #
            _, l, a = sess.run([train_step, loss, model.accuracy],
                               feed_dict=clean_dict)
            _, l, a = sess.run([train_step, loss, model.accuracy],
                               feed_dict=adv_dict)
            ct += 1
            if ct % print_interval == 0:
                print('{} / {}: clean acc: {:.5f}, \tadv acc: {:.5f}, \tloss: {:.5f}'
                      .format(ct, nbatch, clean_acc, adv_acc, l))
        print('EPOCH ends, calculating total loss')
        # evaluate the loss for whole epoch
        adv_x_concrete = my_fast_PGD(sess, model, val_x, val_y)
        adv_dict = {model.x: adv_x_concrete, model.y: val_y}
        adv_acc, l = sess.run([model.accuracy, loss], feed_dict=adv_dict)

        # save the weights
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        print('EPOCH {}: loss: {:.5f}, adv acc: {:.5f}' .format(i, l, adv_acc))
        
        if best_loss < l:
            patience -= 1
            if patience <= 0:
                # restore the best model
                saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
                # verify if the best model is restored
                adv_x_concrete = my_fast_PGD(sess, model, val_x, val_y)
                adv_dict = {model.x: adv_x_concrete, model.y: val_y}
                adv_acc, l = sess.run([model.accuracy, loss], feed_dict=adv_dict)
                print('Best loss: {}, restored loss: {}'.format(l, best_loss))
                # FIXME this is not equal, as there's randomness in PGD?
                # assert abs(l - best_loss) < 0.001
                print('Early stopping .., best loss: {}'.format(best_loss))
                break
        else:
            best_loss = l
            best_epoch = i
            patience = PATIENCE
    

def train(sess, model, loss):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    best_loss = math.inf
    best_epoch = 0
    patience = 5
    print_interval = 500

    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(max_to_keep=10)
    for i in range(NUM_EPOCHS):
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // BATCH_SIZE
        
        for j in range(nbatch):
            start = j * BATCH_SIZE
            end = (j+1) * BATCH_SIZE
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            _, l, a = sess.run([train_step, loss, model.accuracy],
                               feed_dict={model.x: batch_x,
                                          model.y: batch_y})
        print('EPOCH ends, calculating total loss')
        l, a = sess.run([loss, model.accuracy],
                        feed_dict={model.x: val_x,
                                   model.y: val_y})
        print('EPOCH {}: loss: {:.5f}, acc: {:.5f}' .format(i, l, a,))
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        if best_loss < l:
            patience -= 1
            if patience <= 0:
                # restore the best model
                saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
                print('Early stopping .., best loss: {}'.format(best_loss))
                # verify if the best model is restored
                l, a = sess.run([loss, model.accuracy],
                                feed_dict={model.x: val_x,
                                           model.y: val_y})
                assert l == best_loss
                break
        else:
            best_loss = l
            best_epoch = i
            patience = 5

    
class MNIST_CNN(CNNModel):
    def x_shape(self):
        # return [None, 28*28]
        return (28,28,1,)
    def y_shape(self):
        return [None, 10]
        # return (10,)
    def rebuild_model(self, xin):
        # FIXME may not need reshape layer
        x = keras.layers.Reshape([28, 28, 1])(xin)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        return logits


def generate_victim(test_x, test_y):
    inputs = []
    labels = []
    targets = []
    nlabels = 10
    for i in range(nlabels):
        for j in range(1000):
            x = test_x[j]
            y = test_y[j]
            if i == np.argmax(y):
                inputs.append(x)
                labels.append(y)
                onehot = np.zeros(nlabels)
                onehot[(i+1) % 10] = 1
                targets.append(onehot)
                break
    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    return inputs, labels, targets

def my_FGSM(sess, model):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': -0.5,
        'clip_max': 0.5
    }
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(model.x, **fgsm_params)
    return adv_x

def my_PGD(sess, model):
    pgd = ProjectedGradientDescent(model, sess=sess)
    pgd_params = {'eps': 0.3,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': -0.5,
                  'clip_max': 0.5}
    adv_x = pgd.generate(model.x, **pgd_params)
    return adv_x

def my_fast_PGD(sess, model, x, y):
    epsilon = 0.3
    k = 40
    a = 0.01
    
    # loss = model.cross_entropy()
    # grad = tf.gradients(loss, model.x)[0]
    
    # FGSM, PGD, JSMA, CW
    # without random
    # 0.97, 0.97, 0.82, 0.87
    # using random, the performance improved a bit:
    # 0.99, 0.98, 0.87, 0.93
    #
    # adv_x = np.copy(x)
    adv_x = x + np.random.uniform(-epsilon, epsilon, x.shape)
    
    for i in range(k):
        g = sess.run(model.ce_grad, feed_dict={model.x: adv_x, model.y: y})
        adv_x += a * np.sign(g)
        adv_x = np.clip(adv_x, x - epsilon, x + epsilon)
        # adv_x = np.clip(adv_x, 0., 1.)
        adv_x = np.clip(adv_x, -0.5, 0.5)
    return adv_x

def my_JSMA(sess, model):
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': -0.5, 'clip_max': 0.5,
                   'y_target': None}
    return jsma.generate(model.x, **jsma_params)
    

def my_CW(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 # 'y_target': model.y,
                 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 100,
                 'initial_const': 10,
                 'clip_min': -0.5,
                 'clip_max': 0.5}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

def my_CW_targeted(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 'y_target': model.y,
                 # 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 10,
                 'initial_const': 10,
                 'clip_min': -0.5,
                 'clip_max': 0.5}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

        
def mynorm(a, b, p):
    size = a.shape[0]
    delta = a.reshape((size,-1)) - b.reshape((size,-1))
    return np.linalg.norm(delta, ord=p, axis=1)

def test_model_against_attack(sess, model, attack_func, inputs, labels, targets, prefix=''):
    """
    testing against several attacks
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

def train_double_backprop(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        c=100
        # TODO verify the results in their paper
        loss = model.cross_entropy() + c * model.l2_double_backprop()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_group_lasso(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        # group lasso
        c=100
        loss = model.cross_entropy() + c * model.gradients_group_lasso()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_ce(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_adv(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        my_adv_training(sess, model, loss, batch_size=128, num_epochs=50)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def __test():
    sess = tf.Session()
    model = MNIST_CNN()
    loss = model.cross_entropy()
    my_adv_training(sess, model, loss)

        

def restore_model(path):
    tf.reset_default_graph()
    sess = tf.Session()
    model = MNIST_CNN()
    saver = tf.train.Saver()
    saver.restore(sess, path)
    print("Model restored.")
    return sess, model
    

def train_models_main():
    # training of CNN models
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "saved_model/double-backprop.ckpt")
        
    
def my_distillation():
    """1. train a normal model, save the weights
    
    2. teacher model: load the weights, train the model with ce loss
    function but replace: logits=predicted/train_temp(10)
    
    3. predicted = teacher.predict(data.train_data)
       y = sess.run(tf.nn.softmax(predicted/train_temp))
       data.train_labels = y
    
    4. Use the updated data to train the student. The student should
    load the original weights. Still use the loss function with temp.

    https://github.com/carlini/nn_robust_attacks

    """
    pass

def __test():
    train_double_backprop("saved_model/double-backprop.ckpt")
    train_group_lasso("saved_model/group-lasso.ckpt")
    train_ce("saved_model/ce.ckpt")
    train_adv("saved_model/adv.ckpt")
    # train_adv("saved_model/adv2.ckpt")
    
    sess, model = restore_model("saved_model/double-backprop.ckpt")
    sess, model = restore_model("saved_model/group-lasso.ckpt")
    sess, model = restore_model("saved_model/ce.ckpt")
    sess, model = restore_model("saved_model/adv.ckpt")
    # sess, model = restore_model("saved_model/adv2.ckpt")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    # testing model accuracy on clean data
    model.compute_accuracy(sess, model.x, model.y, model.logits, test_x, test_y)
    
    # 10 targeted inputs for demonstration
    inputs, labels, targets = generate_victim(test_x, test_y)
    # all 10000 inputs
    inputs, labels = test_x, test_y
    # random 100 inputs
    indices = random.sample(range(test_x.shape[0]), 100)
    inputs = test_x[indices]
    labels = test_y[indices]
    
    test_model_against_attack(sess, model, my_FGSM, inputs, labels, None, prefix='FGSM')
    test_model_against_attack(sess, model, my_PGD, inputs, labels, None, prefix='PGD')
    test_model_against_attack(sess, model, my_JSMA, inputs, labels, None, prefix='JSMA')
    # test_model_against_attack(sess, model, my_CW_targeted, inputs, labels, targets, prefix='CW_targeted')
    test_model_against_attack(sess, model, my_CW, inputs, labels, None, prefix='CW')

    # adv_x_concrete = my_fast_PGD(sess, model, inputs, labels)
    # my_PGD(sess, model)
    # adv_x = my_CW(sess, model)
    # adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: labels})
    # adv_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, labels)
    

def __test():

    grid_show_image(inputs, 5, 2, 'orig.png')
    np.argmax(labels, 1)
    np.argmax(targets, 1)
    

    # accuracy
    grid_show_image(inputs, 5, 2)
    
    sess.run([model.accuracy],
             feed_dict={model.x: inputs, model.y: targets})

    with tf.Session() as sess:
        train(sess, model, loss)


