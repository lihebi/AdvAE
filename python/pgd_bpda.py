import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import cleverhans.model
import time

from cleverhans.attacks import Attack

from utils import *
from tf_utils import *

class MyPgdBpdaAttack():
    def __init__(self, sess, np_purifier, cnn, x, y,
                 epsilon=0.3, max_steps=40, step_size=0.01, loss_func='xent', rand=True):
        self.np_purifier = np_purifier
        self.cnn = cnn
        self.x = x
        self.y = y
        self.sess = sess
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.step_size = step_size
        self.rand = rand
        
        # calculating grads
        self.logits = cnn.predict(x)
        if loss_func == 'xent':
            loss = my_softmax_xent(self.logits, y)
        elif loss_func == 'cw':
            # label_mask = tf.one_hot(cnn.y, 10, on_value=1.0, off_value=0.0, dtype=tf.float32)
            label_mask = cnn.y
            correct_logit = tf.reduce_sum(label_mask * self.logits, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * self.logits, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            raise NotImplementedError()
        self.grads = tf.gradients(loss, x)[0]
    def attack(self, npx, npy):
        """BPDA attack.

        From test_x, mutate and obtain adversarial examples.
        We assume model has two methods:
        - np_purifier()
        - cnn.prediction

        Then we can do PGD attack like this:
        - purify: npx' = purify(npx)
        - compute gradient: p, g = run([cnn.pred(x), grad(cnn.logits, x)], feed={x: npx'})
        - update *npx*: npx += lr * np.sign(g), and clip

        Do this until the prediction is different or reach a maximum steps.

        How about CW (TODO)

        """

        lower = np.clip(npx - self.epsilon, 0., 1.)
        upper = np.clip(npx + self.epsilon, 0., 1.)

        if self.rand:
            npx = npx + np.random.uniform(self.epsilon, self.epsilon, npx.shape)
        else:
            npx = np.copy(npx)

        for i in range(self.max_steps):
            npx_prime = self.np_purifier(self.sess, npx)
            # baseline
            # npx_prime = npx
            p, g = self.sess.run([self.logits, self.grads],
                            {self.x: npx_prime, self.y: npy})
            # FIXME all different?? I should stop each one individually
            ypred = np.argmax(p, axis=1)
            # print('ypred: {}'.format(ypred))
            ytrue = np.argmax(npy, axis=1)
            # print('ytrue: {}'.format(ytrue))
            wrong = (ypred != ytrue).astype(int).sum()
            total = npy.shape[0]
            print('Step {} / {}, number of wrong {} / {}'.format(i, self.max_steps, wrong, total))
            if wrong == total:
                print('done')
                break
            # print(g.shape)
            npx += self.step_size * np.sign(g)
            npx = np.clip(npx, lower, upper)
        return npx
        
def my_PGD_BPDA(sess, np_purifier, cnn, x, y,
                epsilon=0.3, max_steps=40, step_size=0.01, loss_func='xent', rand=True):
    attack = MyPgdBpdaAttack(sess, np_purifier, cnn, x, y,
                             epsilon=epsilon, max_steps=max_steps, step_size=step_size,
                             loss_func=loss_func, rand=rand)
    
    def my_wrap(npx, npy):
        res = attack.attack(npx, npy)
        return np.array(res, dtype=np.float32)

    wrap = tf.py_func(my_wrap, [x, y], tf.float32)
    wrap.set_shape(x.get_shape())
    return wrap

def eval_with_adv(sess, rec_model, cnn, x, y, adv_x, test_x, test_y, batch_size=50):
    """Evaluate the accuracy using batch_size=50.

    - test_x/y: 1000-sampled from original test set
    - adv_x: the adv example tensor built upon x
    """
    num_samples = test_x.shape[0]
    nbatch = num_samples // batch_size
    
    preds = cnn.predict(rec_model(adv_x))
    acc = my_accuracy_wrapper(logits=preds, labels=cnn.y)

    res = 0.
    
    for i in range(nbatch):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_x = test_x[start:end]
        batch_y = test_y[start:end]

        t = time.time()
        print('Batch {} / {}'.format(i, nbatch))
        tmp = sess.run(acc, feed_dict={cnn.x: batch_x, cnn.y: batch_y,
                                       keras.backend.learning_phase(): 0})
        print('Attack done. Accuracy: {}, time: {:.3f}'.format(tmp, time.time()-t))
        res += tmp
    res /= nbatch
    print('Total accuracy: {:.3f}'.format(res))
    return res

