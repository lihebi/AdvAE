"""Test defense gan
"""

import keras
import logging

from attacks import *
from cnn_models import *
from defensegan_models import *
from models import *

from tf_utils import *


sys.path.append('/home/hebi/github/reading/')

import defensegan
from defensegan.utils import config
from defensegan import whitebox
from defensegan.models.gan import MnistDefenseGAN


if 'simple-prompt' not in tf.app.flags.FLAGS:
    tf.app.flags.DEFINE_boolean('simple-prompt', False, '')
if 'i' not in tf.app.flags.FLAGS:
    tf.app.flags.DEFINE_boolean('i', False, '')

def setup_flags():
    flags = tf.app.flags
    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing. [False]")
    # flags.DEFINE_boolean('simple_prompt', False, '')

    # Other flags
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697.')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_string('rec_path', None, 'Path to reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the classifier.')
    flags.DEFINE_boolean("online_training", False,
                         "Train the base classifier on reconstructions.")
    flags.DEFINE_string("defense_type", "none", "Type of defense [none|defense_gan|adv_tr]")
    flags.DEFINE_string("attack_type", "none", "Type of attack [fgsm|cw|rand_fgsm]")
    flags.DEFINE_string("results_dir", None, "The final subdirectory of the results.")
    flags.DEFINE_boolean("same_init", False, "Same initialization for z_hats.")
    flags.DEFINE_string("model", "F", "The classifier model.")
    flags.DEFINE_string("debug_dir", "temp", "The debug directory.")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("override", False, "Overriding the config values of reconstruction "
                                            "hyperparameters. It has to be true if either "
                                            "`--rec_rr`, `--rec_lr`, or `--rec_iters` is passed "
                                            "from command line.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the classifier on the reconstructed samples "
                         "using Defense-GAN.")

def load_defgan(sess, cfgpath='/home/hebi/github/reading/defensegan/gans/mnist/'):
    # setup_flags()
    # cfg = config.load_config('/home/hebi/tmp/defensegan_tmp/output/gans/mnist/')
    cfg = config.load_config(cfgpath)
    gan = defensegan.models.gan.MnistDefenseGAN(cfg=cfg, test_mode=True)
    # gan.purify = lambda npx: gan.sess.run(gan.reconstruct(x))
    gan.set_session(sess)
    # gan.load_generator('/home/hebi/tmp/defensegan_tmp/output/gans/mnist/')
    gan.load_generator()
    return gan

def train_defgan():
    setup_flags()
    FLAGS = tf.app.flags.FLAGS
    FLAGS.is_train = True
    gan = defensegan.models.gan.MnistDefenseGAN(cfg=cfg)
    gan.train()

def __test():
    """Run DefenseGAN's white box test."""
    FLAGS = tf.app.flags.FLAGS
    # FLAGS.is_train = True
    FLAGS.nb_epochs = 1
    FLAGS.num_tests = 400
    FLAGS.attack_type = 'fgsm'
    FLAGS.defense_type = 'defense_gan'
    FLAGS.model = 'A'
    FLAGS.results_dir = 'whitebox'
    # (train_x, train_y), (test_x, test_y) = load_mnist_data_nocat()
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    data = train_x, train_y, test_x, test_y
    whitebox.whitebox(gan, online_training=True, hebi_data=data)
    whitebox.whitebox(gan)
    whitebox.main(cfg)
    
    accuracies = whitebox.whitebox(
        gan,
        # rec_data_path=FLAGS.rec_path,
        # batch_size=FLAGS.batch_size,
        # learning_rate=FLAGS.learning_rate,
        nb_epochs=FLAGS.nb_epochs,
        # eps=FLAGS.fgsm_eps,
        # online_training=FLAGS.online_training,
        defense_type=FLAGS.defense_type,
        # num_tests=FLAGS.num_tests,
        # attack_type=FLAGS.attack_type,
        # num_train=FLAGS.num_train,
        hebi_data=data
    )

def __test():
    """Testing the clean rec accuracy."""
    sess = create_tf_session()
    defgan = load_defgan(sess)
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    x = keras.layers.Input(shape=(28,28,1), dtype='float32')
    rec = defgan.reconstruct(x, batch_size=100)
    tf_init_uninitialized(sess)
    out = sess.run(rec, feed_dict={x: test_x[:100]})
    grid_show_image([test_x[:10], out[:10]])
    preds = sess.run(cnn.predict(cnn.x), feed_dict={cnn.x: out[:50], K.learning_phase(): 0})
    preds = sess.run(cnn.predict(cnn.x), feed_dict={cnn.x: test_x[:50], K.learning_phase(): 0})
    # np.argmax(preds, axis=1).astype(int)

    (np.argmax(preds, axis=1) == np.argmax(test_y[:50], axis=1)).astype(int).sum()


class MyDefGan():
    def __init__(self, defgan):
        self.defgan = defgan
        self.sess = defgan.sess
        self.x = keras.layers.Input(shape=(28,28,1), dtype='float32')
        self.batch_size = 50
        self.rec = self.defgan.reconstruct(self.x, batch_size=self.batch_size)
        tf_init_uninitialized(self.sess)
    def purify_np(self, npx):
        return self.sess.run(self.rec, feed_dict={self.x: npx, K.learning_phase(): 0})
    def __call__(self, x):
        rec = self.defgan.reconstruct(x, batch_size=self.batch_size)
        tf_init_uninitialized(self.sess)
        return rec
    
def my_PGD_BPDA(sess, defgan, cnn, npx, npy,
                epsilon=0.3, max_steps=40, step_size=0.01, loss='xent'):
    """BPDA attack.

    From test_x, mutate and obtain adversarial examples.
    We assume model has two methods:
    - defgan.purify_np()
    - cnn.prediction

    Then we can do PGD attack like this:
    - purify: npx' = purify(npx)
    - compute gradient: p, g = run([cnn.pred(x), grad(cnn.logits, x)], feed={x: npx'})
    - update *npx*: npx += lr * np.sign(g), and clip

    Do this until the prediction is different or reach a maximum steps.

    How about CW (TODO)

    """
    mydefgan = MyDefGan(defgan)
    
    lower = np.clip(npx - epsilon, 0., 1.)
    upper = np.clip(npx + epsilon, 0., 1.)
    
    npx = npx + np.random.uniform(epsilon, epsilon, npx.shape)
    # npx = np.copy(npx)

    # calculating grads
    logits = cnn.predict(cnn.x)
    if loss == 'xent':
        loss = my_softmax_xent(logits, cnn.y)
    elif loss == 'cw':
        # label_mask = tf.one_hot(cnn.y, 10, on_value=1.0, off_value=0.0, dtype=tf.float32)
        label_mask = cnn.y
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits, axis=1)
        loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
        raise NotImplementedError()
    grads = tf.gradients(loss, cnn.x)[0]
    
    for i in range(max_steps):
        npx_prime = mydefgan.purify_np(npx)
        # baseline
        # npx_prime = npx
        p, g = sess.run([logits, grads],
                        {cnn.x: npx_prime, cnn.y: npy})
        # FIXME all different?? I should stop each one individually
        ypred = np.argmax(p, axis=1)
        # print('ypred: {}'.format(ypred))
        ytrue = np.argmax(npy, axis=1)
        # print('ytrue: {}'.format(ytrue))
        wrong = (ypred != ytrue).astype(int).sum()
        total = npy.shape[0]
        print('Step {} / {}, number of wrong {} / {}'.format(i, max_steps, wrong, total))
        if wrong == total:
            print('done')
            break
        # print(g.shape)
        npx += step_size * np.sign(g)
        npx = np.clip(npx, lower, upper)
    return npx

def __test():
    """Testing the PGD BPDA attack."""
    sess = create_tf_session()
    defgan = load_defgan(sess)

    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    cnn = MNISTModel()
    # cnn = DefenseGAN_f()

    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')
    # cnn.load_weights(sess, 'saved_models/MNIST-DefenseGAN_f-CNN.hdf5')
    
    adv_x = my_PGD_BPDA(sess, defgan, cnn, test_x[:50], test_y[:50], loss='xent')
    adv_x = my_PGD_BPDA(sess, defgan, cnn, test_x[:50], test_y[:50], loss='cw')

    grid_show_image([test_x[:10], adv_x[:10]])

    my_bpda_defgan()


def __test():
    """Test CW L2 BPDA attack."""
    sess = create_tf_session()
    defgan = load_defgan(sess)
    mydefgan = MyDefGan(defgan)

    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    cnn = MNISTModel()

    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')

    adv_x = my_CW_BPDA(mydefgan, cnn, sess, cnn.x, cnn.y, params={'max_iterations': 30})
    # tf_init_uninitialized(sess)
    with cleverhans.utils.TemporaryLogLevel(logging.INFO, "cleverhans.attacks.bpda"):
        adv_x_concrete = sess.run(adv_x, feed_dict={cnn.x: test_x[:50], cnn.y: test_y[:50]})

    adv_x_concrete = sess.run(adv_x, feed_dict={cnn.x: test_x[:50], cnn.y: test_y[:50]})

    # logger = cleverhans.utils.create_logger("cleverhans.attacks.hebi")
    # logger.setLevel(logging.INFO)
    # with cleverhans.utils.TemporaryLogLevel(logging.INFO, "cleverhans.attacks.hebi"):
    #     logger.info('hello')


    preds = cnn.predict(mydefgan(cnn.x))
    
    # pred = model.predict(adv_x)
    
    nppred = sess.run(preds, feed_dict={cnn.x: test_x[:50]})
    nppred = sess.run(preds, feed_dict={cnn.x: adv_x_concrete})

    yy = np.argmax(nppred, 1)
    yyy = np.argmax(test_y[:50], 1)
    (yy != yyy).astype(int).sum()

    grid_show_image([test_x[:10], adv_x_concrete[:10]])
    
if __name__ == '__main__':
    # train_defgan()
    pass

