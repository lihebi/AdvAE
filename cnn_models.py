import keras
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

# VISUAL_LAMBDA = 100
class VisualCNNModel():
    # model.image_size: size of the image (e.g., 28 for MNIST, 32 for CIFAR)
    # model.num_channels: 1 for greyscale, 3 for color images
    # model.num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)
    def __init__(self, loss_type, grad_type, visual_lambda):
        assert loss_type in ['l2', 'l1', 'l0', 'group_lasso', None]
        assert grad_type in ['xent', 'logit']
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10
        # img = tf.placeholder(dtype='float32', shape=(None,28,28,1))

        # FIXME seems using keras input layers slows down the training a lot
        self.img = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        x = keras.layers.Conv2D(32, 3)(self.img)
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
        self.logits = keras.layers.Dense(10)(x)
        self.preds = tf.argmax(self.logits, axis=1)
        self.label = tf.placeholder(dtype='float32', shape=(None,10))
        self.ce_loss = tf.losses.softmax_cross_entropy(self.label, self.logits)

        if grad_type == 'logit':
            grads = tf.abs(tf.gradients(tf.math.reduce_max(self.logits), self.img))
        else:
            grads = tf.abs(tf.gradients(self.ce_loss, self.img))

        # L2 loss
        # self.vi_loss = tf.square(grads)
        # define many templates
        # mast the loss through the templates
        if loss_type == 'l1':
            # L1 loss
            # TODO add setting to 0 when less than a threshold
            self.vi_loss = tf.reduce_mean(grads)
        elif loss_type == 'group_lasso':
            # Group lasso
            # tf.sqrt(tf.nn.ave_pool(tf.square(h)))
            # tf.sqrt(keras.layers.AvgPool3D(4, 4, 'valid')(tf.square(grads)))
            # tf.sqrt(tf.nn.pool(tf.square(grads), 4, 'AVG', 'VALID'))
            # tf.sqrt(tf.nn.pool(tf.square(grads), 4, 'AVG', 'VALID'))
            # tf.nn.pool(tf.square(grads), 4, 'AVG', 'VALID')
            self.vi_loss = tf.reduce_mean(
                tf.sqrt(keras.layers.AvgPool3D(4, 4, 'valid')(tf.square(grads))
                        + 1e-10))
        elif loss_type == 'l0':
            # FIXME L0 loss
            # The problems of L0: what is the grads?
            self.vi_loss = tf.reduce_mean(
                tf.to_float(
                    tf.logical_not(
                        tf.equal(grads, tf.zeros_like(grads)))))
        elif loss_type == 'l2':
            self.vi_loss = tf.sqrt(tf.reduce_mean(tf.square(grads)))
        else:
            # no visual loss term
            #
            # FIXME this is not used. However setting it to 0 seems to
            # cause error.
            self.vi_loss = tf.reduce_mean(grads)
            # self.vi_loss = 0
        if loss_type is None:
            self.loss = self.ce_loss
        else:
            self.loss = self.ce_loss + self.vi_loss * visual_lambda
        # tf.metrics.accuracy
        self.acc = tf.reduce_mean(
            tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
                                 tf.argmax(self.label, axis=1))))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        # FIXME this model is only used in self.predict
        self.model = keras.models.Model(self.img, self.logits)
    def predict(self, data):
        return self.model(data)
    def evaluate_np(self, sess, test_x, test_y):
        return sess.run([self.acc, self.loss],
                        feed_dict={self.img: test_x, self.label: test_y})
    def predict_np(self, sess, test_x):
        # FIXME can I just supply only one placeholder?
        return sess.run(self.preds, feed_dict={self.img: test_x})
    def train(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
        best_loss = 1000
        patience = 5
        for i in range(NUM_EPOCHS):
            shuffle_idx = np.arange(train_x.shape[0])
            np.random.shuffle(shuffle_idx)
            nbatch = train_x.shape[0] // BATCH_SIZE
            sum_acc, sum_loss, sum_vl, sum_xl = 0,0,0,0
            for j in range(nbatch):
                start = j * BATCH_SIZE
                end = (j+1) * BATCH_SIZE
                batch_x = train_x[shuffle_idx[start:end]]
                batch_y = train_y[shuffle_idx[start:end]]
                _, l, a, vl, xl = sess.run([self.train_step, self.loss,
                                            self.acc, self.vi_loss, self.ce_loss],
                                           feed_dict={self.img: batch_x,
                                                      self.label: batch_y})
                sum_acc += a
                sum_loss += l
                sum_vl += vl
                sum_xl += xl
            if best_loss < (sum_loss / nbatch):
                patience -= 1
                if patience <= 0:
                    print('Early stopping .., best loss: {}'.format(best_loss))
                    break
            else:
                best_loss = sum_loss / nbatch
                patience = 5
            print('EPOCH {}: loss: {:.5f}, acc: {:.5f}, vl: {:.10f}, xl: {:.10f}'
                  .format(i+1, sum_loss / nbatch, sum_acc / nbatch,
                          sum_vl / nbatch, sum_xl / nbatch))
        l, a, vl = sess.run([self.loss, self.acc, self.vi_loss],
                            feed_dict={self.img: test_x,
                                       self.label: test_y})
        print('testing loss: {}, acc: {}, vi_loss: {}'.format(l, a, vl))

sys.path.append('/home/hebi/github/reading/nn_robust_attacks')
import l2_attack
import l0_attack

def generate_victim(test_x, test_y):
    inputs = []
    targets = []
    nlabels = 10
    for i in range(nlabels):
        for j in range(1000):
            x = test_x[j]
            y = test_y[j]
            if i == np.argmax(y):
                inputs.append(x)
                onehot = np.zeros(nlabels)
                onehot[(i+1) % 10] = 1
                targets.append(onehot)
                break
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def __test():
    visual_defense_exp(None, 'logit', visual_lambda=0, fname='cwl2-orig.png')
    for loss_type in ['group_lasso', 'l2', 'l1']:
        for grad_type in ['xent', 'logit']:
            for visual_lambda in [1, 10, 50, 200, 500, 1000]:
                print('======', loss_type, visual_lambda)
                fname = 'cwl2-{}-{}-{:=04}.png'.format(loss_type, grad_type, visual_lambda)
                visual_defense_exp(loss_type, grad_type,
                                   visual_lambda=visual_lambda,
                                   fname=fname)
    
def visual_defense_exp(loss_type, grad_type, visual_lambda, fname):
    # loss_type=group_lasso
    # loss_type=l2
    # loss_type=None
    # visual_lambda=100
    model = VisualCNNModel(loss_type=loss_type,
                           grad_type=grad_type,
                           visual_lambda=visual_lambda)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    # sess = tf.Session()
    with tf.Session() as sess:
        model.train(sess)
        # Evaluate this model
        # sess.run([model.acc], feed_dict={model.img: test_x, model.label: test_y})
        # pred = model.predict_np(sess, test_x)
        # print('Model prediction: {}'.format(pred))
        res = model.evaluate_np(sess, test_x, test_y)
        model_acc = res[0]
        print('Model accuracy: {:.5f}'.format(model_acc))

        # victimis
        inputs, targets = generate_victim(test_x, test_y)

        pred = model.predict_np(sess, inputs)
        print('Victim prediction: {}'.format(pred))
        res = model.evaluate_np(sess, inputs, targets)  # should be 0 accuracy
        print('Victim accuracy (should be 0): {:.5f}'.format(res[0]))

        # adverserial attacks
        attack = l2_attack.CarliniL2(sess, model, batch_size=10,
                                     max_iterations=1000, confidence=0)
        # attack = l0_attack.CarliniL0(sess, model, max_iterations=1000,
        #                              initial_const=10, largest_const=15)
        # inputs, targets = generate_data_2(data)
        adv_l2 = attack.attack(inputs, targets)
        grid_show_image(inputs, 10, 1, 'images/orig-mnist.png')
        grid_show_image(adv_l2, 10, 1, 'images/'+fname)
        print('outputed adv example to', 'images/'+fname)

        pred = model.predict_np(sess, adv_l2)
        res = model.evaluate_np(sess, adv_l2, targets)
        attack_acc = res[0]
        print('Victim prediction: {}'.format(pred))
        print('Victim accuracy (should be 1 if attack succeeds): {:.5f}'
              .format(res[0]))

        def mynorm(a, b, p, t=None):
            delta = a.reshape(10,-1) - b.reshape((10,-1))
            if t is not None:
                delta[delta < t] = 0
            return np.linalg.norm(delta, ord=p, axis=1)
        # calculate norm
        l2_dist = mynorm(adv_l2, inputs, 2).mean()
        l1_dist = mynorm(adv_l2, inputs, 1).mean()
        l0_dist = mynorm(adv_l2, inputs, 0).mean()
        # TODO print li_dist
        li_dist = mynorm(adv_l2, inputs, np.inf).mean()
        
        # save the draw of the distortion
        # mynorm(adv_l2, inputs, 2, 0.1)
        print('saving distortion plot ..')
        
        # plt.figure()
        # plt.hist((adv_l2.reshape(10,-1) - inputs.reshape(10,-1))[0])
        # plt.savefig('images/'+fname + '.dist.hist.png')

        plt.figure()
        # plt.imshow((adv_l2-inputs)[0].reshape((28,28)), cmap='hot')
        # TODO Add more figures
        sns.heatmap(np.abs((adv_l2-inputs)[0]).reshape((28,28)),
                    linewidth=0.5, cmap='Reds', vmin=0, vmax=0.8)
        # plt.hist((adv_l2.reshape(10,-1) - inputs.reshape(10,-1))[0])
        setting_report = 'Lambda: {}, loss: {}, grad: {}'.format(visual_lambda, loss_type, grad_type)
        acc_report = 'Model Acc: {:.4f} ASR: {:.1f}' .format(model_acc, attack_acc)
        distort_report = 'distortion: L2: {:.2f}, L1: {:.2f}, L0: {:.2f}, Li: {:.2f}'.format(l2_dist, l1_dist, l0_dist, li_dist)
        full_report = setting_report + '\n' + acc_report + '\n' + distort_report

        print(full_report)
        with open('log.txt', 'a') as f:
            f.write(setting_report + '\n')
            f.write('\t' + acc_report + '\n')
            f.write('\t' + distort_report + '\n')
        
        plt.title(full_report)
        plt.savefig('images/'+fname + '.dist.heat.png')

