"""Test defense gan
"""

import keras
import logging

from attacks import *
from cnn_models import *
from defensegan_models import *
from models import *

from tf_utils import *


sys.path.append('/home/XXX/github/reading/')

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

def load_defgan(sess, cfgpath='/home/XXX/github/reading/defensegan/gans/mnist/'):
    # setup_flags()
    # cfg = config.load_config('/home/XXX/tmp/defensegan_tmp/output/gans/mnist/')
    cfg = config.load_config(cfgpath)
    gan = defensegan.models.gan.MnistDefenseGAN(cfg=cfg, test_mode=True)
    # gan.purify = lambda npx: gan.sess.run(gan.reconstruct(x))
    gan.set_session(sess)
    # gan.load_generator('/home/XXX/tmp/defensegan_tmp/output/gans/mnist/')
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
    whitebox.whitebox(gan, online_training=True, XXX_data=data)
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
        XXX_data=data
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
        return self.sess.run(self.rec, feed_dict={self.x: npx})
    def __call__(self, x):
        rec = self.defgan.reconstruct(x, batch_size=self.batch_size)
        tf_init_uninitialized(self.sess)
        return rec
    
# def my_PGD_BPDA_np(sess, mydefgan, cnn, logits, grads,
#                    npx, npy,
#                    epsilon=0.3, max_steps=40, step_size=0.01, loss='xent', rand=True):

class MyPgdBpdaAttack():
    def __init__(self, sess, mydefgan, cnn, x, y,
                 epsilon=0.3, max_steps=40, step_size=0.01, loss_func='xent', rand=True):
        self.mydefgan = mydefgan
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
        - defgan.purify_np()
        - cnn.prediction

        Then we can do PGD attack like this:
        - purify: npx' = purify(npx)
        - compute gradient: p, g = run([cnn.pred(x), grad(cnn.logits, x)], feed={x: npx'})
        - update *npx*: npx += lr * np.sign(g), and clip

        Do this until the prediction is different or reach a maximum steps.

        How about CW (TODO)

        """
        # mydefgan = MyDefGan(defgan)

        lower = np.clip(npx - self.epsilon, 0., 1.)
        upper = np.clip(npx + self.epsilon, 0., 1.)

        if self.rand:
            npx = npx + np.random.uniform(self.epsilon, self.epsilon, npx.shape)
        else:
            npx = np.copy(npx)

        for i in range(self.max_steps):
            npx_prime = self.mydefgan.purify_np(npx)
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
        
def my_PGD_BPDA(sess, mydefgan, cnn, x, y,
                epsilon=0.3, max_steps=40, step_size=0.01, loss_func='xent', rand=True):
    attack = MyPgdBpdaAttack(sess, mydefgan, cnn, x, y,
                             epsilon=epsilon, max_steps=max_steps, step_size=step_size,
                             loss_func=loss_func, rand=rand)
    
    def my_wrap(npx, npy):
        res = attack.attack(npx, npy)
        return np.array(res, dtype=np.float32)

    wrap = tf.py_func(my_wrap, [x, y], tf.float32)
    wrap.set_shape(x.get_shape())
    return wrap

def __test():
    """Testing the PGD BPDA attack."""
    sess = create_tf_session()
    with sess.as_default():
        defgan = load_defgan(sess)

        (train_x, train_y), (test_x, test_y) = load_mnist_data()

        cnn = MNISTModel()
        # cnn = DefenseGAN_f()

        cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')
        # cnn.load_weights(sess, 'saved_models/MNIST-DefenseGAN_f-CNN.hdf5')

        # PGD
        adv_x = my_PGD_BPDA(sess, defgan, cnn, test_x[:50], test_y[:50], loss_func='xent')
        # FGSM
        adv_x = my_PGD_BPDA(sess, defgan, cnn, test_x[:50], test_y[:50], loss_func='xent',
                            epsilon=0.3, max_steps=1, step_size=0.3, rand=False)
        # PGD with CW loss (i.e. CW L_inf)
        # adv_x = my_PGD_BPDA(sess, defgan, cnn, test_x[:50], test_y[:50], loss='cw')

        grid_show_image([test_x[:10], adv_x[:10]])

def eval_defgan(sess, mydefgan, cnn, x, y, adv_x, test_x, test_y, batch_size=50):
    """Evaluate the accuracy using batch_size=50.

    - test_x/y: 1000-sampled from original test set
    - adv_x: the adv example tensor built upon x
    """
    num_samples = test_x.shape[0]
    nbatch = num_samples // batch_size
    
    preds = cnn.predict(mydefgan(adv_x))
    acc = my_accuracy_wrapper(logits=preds, labels=cnn.y)

    res = 0.
    
    for i in range(nbatch):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_x = test_x[start:end]
        batch_y = test_y[start:end]

        t = time.time()
        print('Batch {} / {}'.format(i, nbatch))
        tmp = sess.run(acc, feed_dict={cnn.x: batch_x, cnn.y: batch_y})
        print('Attack done. Accuracy: {}, time: {:.3f}'.format(tmp, time.time()-t))
        res += tmp
    res /= nbatch
    print('Total accuracy: {:.3f}'.format(res))
    return res

def test_defgan(filename):
    """Test clean and three attacks on defgan models."""
    if os.path.exists(filename):
        print('Already exists {}'.format(filename))
        return
    sess = create_tf_session()
    _defgan = load_defgan(sess)
    mydefgan = MyDefGan(_defgan)
    (train_x, train_y), (test_x, test_y) = load_mnist_data()
    cnn = MNISTModel()
    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')

    # random sample test
    # FIXME use 1000 during production
    indices = random.sample(range(test_x.shape[0]), 1000)
    test_x = test_x[indices]
    test_y = test_y[indices]

    res = {}
    # clean
    print('Testing clean data ..')
    adv_x = cnn.x
    res['CNN clean'] = eval_defgan(sess, mydefgan, cnn, cnn.x, cnn.y, adv_x, test_x, test_y)
    
    # FGSM
    print('Testing FGSM ..')
    adv_x = my_PGD_BPDA(sess, mydefgan, cnn, cnn.x, cnn.y,
                        loss_func='xent', epsilon=0.3, max_steps=1, step_size=0.3, rand=False)
    res['FGSM'] = eval_defgan(sess, mydefgan, cnn, cnn.x, cnn.y, adv_x, test_x, test_y)
    print('Testing PGD ..')
    # PGD
    adv_x = my_PGD_BPDA(sess, mydefgan, cnn, cnn.x, cnn.y, loss_func='xent')
    res['PGD'] = eval_defgan(sess, mydefgan, cnn, cnn.x, cnn.y, adv_x, test_x, test_y)
    print('Testing CW l2 ..')
    # CW l2
    adv_x = my_CW_BPDA(sess, mydefgan, cnn, cnn.x, cnn.y, params={'max_iterations': 30})
    res['CW'] = eval_defgan(sess, mydefgan, cnn, cnn.x, cnn.y, adv_x, test_x, test_y)

    with open(filename, 'w') as fp:
        json.dump(res, fp, indent=4)
    

def __test():
    """Test CW L2 BPDA attack."""
    test_defgan('images/defgan.json')
    
    sess = create_tf_session()
    defgan = load_defgan(sess)
    mydefgan = MyDefGan(defgan)

    (train_x, train_y), (test_x, test_y) = load_mnist_data()

    cnn = MNISTModel()

    cnn.load_weights(sess, 'saved_models/MNIST-mnistcnn-CNN.hdf5')

    adv_x = my_CW_BPDA(sess, mydefgan, cnn, cnn.x, cnn.y, params={'max_iterations': 30})
    # tf_init_uninitialized(sess)
    with cleverhans.utils.TemporaryLogLevel(logging.INFO, "cleverhans.attacks.bpda"):
        adv_x_concrete = sess.run(adv_x, feed_dict={cnn.x: test_x[:50], cnn.y: test_y[:50]})

    adv_x_concrete = sess.run(adv_x, feed_dict={cnn.x: test_x[:50], cnn.y: test_y[:50]})

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
    test_defgan('images/defgan.json')
    pass

