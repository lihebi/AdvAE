import tensorflow as tf
from attacks import CLIP_MIN, CLIP_MAX

def my_add_noise(x, noise_factor=0.5):
    # noise_factor = 0.5
    noisy_x = x + noise_factor * tf.random.normal(shape=tf.shape(x), mean=0., stddev=1.)
    noisy_x = tf.clip_by_value(noisy_x, CLIP_MIN, CLIP_MAX)
    return noisy_x

def my_sigmoid_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))
def my_l2loss(x1, x2):
    return tf.reduce_mean(tf.square(x1-x2))

def my_softmax_xent(logits=None, labels=None):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))

def my_accuracy_wrapper(logits=None, labels=None):
    return tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1),
        tf.argmax(labels, 1)), dtype=tf.float32))

def my_compute_accuracy(sess, x, y, preds, x_test, y_test):
    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
        dtype=tf.float32))
    return sess.run(acc, feed_dict={x: x_test, y: y_test})

def tf_init_uninitialized(sess, choice=['global', 'local']):
    """
    global_vars = tf.global_variables()
    local_vars = tf.local_variables()

    choice: global, local
    """
    allvars=tf.global_variables()
    def init(allvars):
        # Find initialized status for all variables.
        is_var_init = [tf.is_variable_initialized(var) for var in allvars]
        is_initialized = sess.run(is_var_init)

        # List all variables that were not previously initialized.
        not_initialized_vars = [var for (var, init) in
                                zip(allvars, is_initialized) if not init]
        for v in not_initialized_vars:
            print('[!] not init: {}'.format(v.name))
        # Initialize all uninitialized variables found, if any.
        if not_initialized_vars:
            sess.run(tf.variables_initializer(not_initialized_vars))
            print('Initialized')
    if 'global' in choice:
        print('Initializing global variables ..')
        init(tf.global_variables())
    if 'local' in choice:
        print('Initializing local variables ..')
        init(tf.local_variables())
    
def create_tf_session():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

