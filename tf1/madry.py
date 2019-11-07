import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# minimum working example (MWE) of Madry's mnist_challenge

class Model(object):
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
        self.y_input = tf.placeholder(tf.int64, shape = [None])

        self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

        # first convolutional layer
        W_conv1 = self._weight_variable([5,5,1,32])
        b_conv1 = self._bias_variable([32])

        h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        # second convolutional layer
        W_conv2 = self._weight_variable([5,5,32,64])
        b_conv2 = self._bias_variable([64])

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # first fully connected layer
        W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self._bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # output layer
        W_fc2 = self._weight_variable([1024,10])
        b_fc2 = self._bias_variable([10])

        self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

        self.xent = tf.reduce_sum(y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME')

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                        - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 1) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return x

def train(model, attack, mnist, batch_size=50):
    # Setting up the optimizer
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                       global_step=global_step)
    with tf.Session() as sess:
        # Initialize the summary writer, global variables, and our time counter.
        sess.run(tf.global_variables_initializer())

        # Main training loop
        for ii in range(10000):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            # Compute Adversarial Perturbations
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            # Output to stdout
            if ii % 20 == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                print('Step {}:'.format(ii))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            # Actual training step
            sess.run(train_step, feed_dict=adv_dict)

def test():
    batch_size = 50
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    model = Model()

    # Set up adversary
    attack = LinfPGDAttack(model, 0.3, 40, 0.01, True, 'xent')

    train(model, attack, mnist, batch_size=50)

if __name__ == '__main__':
    test()
