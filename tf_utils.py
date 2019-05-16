import tensorflow as tf

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
