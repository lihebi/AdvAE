import numpy as np
from PIL import Image
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

__all__ = ['load_mnist', 'sample_and_view']

def load_mnist():
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    # convert data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    # train_x.shape
    # FIXME should I have the channel here?
    train_x = np.reshape(train_x, (train_x.shape[0], 28,28,1))
    test_x = np.reshape(test_x, (test_x.shape[0], 28,28,1))

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    return (train_x, train_y), (test_x, test_y)


def load_mnist_ds(batch_size):
    (train_x, train_y), (test_x, test_y) = load_mnist()
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
                        .shuffle(buffer_size=1024)\
                        .batch(batch_size, drop_remainder=True)\
                        .repeat()
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))\
                             .shuffle(buffer_size=1024)\
                             .batch(batch_size, drop_remainder=True)\
                             .repeat(1)
    steps_per_epoch = train_x.shape[0] // batch_size
    test_steps_per_epoch = test_x.shape[0] // batch_size
    return ds, test_ds, steps_per_epoch, test_steps_per_epoch


def load_cifar10():
    # 32, 32, 3, uint8
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    return (train_x, train_y), (test_x, test_y)


def download_cifar():
    # call once to download into ./cache/cifar-10-batches-bin
    DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join('cache', filename)
    cachedir = 'cache'
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                  filename, 100.0 * count * block_size / total_size))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(cachedir)

import sys
sys.path.append('./tf_models')
from official.vision.image_classification import cifar_preprocessing

def my_parse_record(raw_record, is_training, dtype):
    record_vector = tf.io.decode_raw(raw_record, tf.uint8)
    label = tf.cast(record_vector[0], tf.int32)
    depth_major = tf.reshape(record_vector[1:32*32*3+1],
                             [3, 32, 32])
    image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

    image = my_preprocess_image(image, is_training)
    image = tf.cast(image, dtype)
    label = tf.compat.v1.sparse_to_dense(label, (10,), 1)
    return image, label


def my_preprocess_image(image, is_training):
    if is_training:
        image = tf.image.resize_with_crop_or_pad(
            image, 32 + 8, 32 + 8)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    # DEBUG remove standardization, so that range is 0-255
    # image = tf.image.per_image_standardization(image)
    return image

def load_cifar10_ds(batch_size):
    data_dir = 'cache/cifar-10-batches-bin'
    ds = cifar_preprocessing.input_fn(
        is_training=True,
        data_dir=data_dir,
        batch_size=batch_size,
        # parse_record_fn=cifar_preprocessing.parse_record,
        parse_record_fn=my_parse_record,
        drop_remainder=True)

    # test ds
    test_ds = cifar_preprocessing.input_fn(
        is_training=False,
        data_dir=data_dir,
        batch_size=batch_size,
        # FIXME should this be 1?
        # num_epochs=1,
        parse_record_fn=my_parse_record)
    steps_per_epoch = cifar_preprocessing.NUM_IMAGES['train'] // batch_size
    test_steps_per_epoch = cifar_preprocessing.NUM_IMAGES['validation'] // batch_size

    next(iter(ds))[1].shape
    # TensorShape([128, 10])
    next(iter(ds))[0].shape
    # TensorShape([128, 32, 32, 3])
    # so this is a iterable dataset
    return ds, test_ds, steps_per_epoch, test_steps_per_epoch

def sample_and_view(x, num=10):
    stacked = np.concatenate(x[:num], 1)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    if x.shape[1] == 28:
        arr = (np.reshape(stacked, (28,28*num)) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        # rescale to 2 because it is a bit small
        img = img.resize((img.width*3, img.height*3), Image.ANTIALIAS)
        img.save(tmp)
    elif x.shape[1] == 32:
        # FIXME how to scale it?
        plt.imsave(tmp, stacked, dpi=500)
    print('#<Image: ' + tmp.name + '>')


def test():
    (train_x, train_y), (test_x, test_y) = load_mnist()
    (train_x, train_y), (test_x, test_y) = load_cifar10()
    sample_and_view(train_x)

    ds, test_ds, steps_per_epoch, test_steps_per_epoch = load_mnist_ds(50)
    for i, d in enumerate(test_ds):
        print(i)
    ds, test_ds, steps_per_epoch, test_steps_per_epoch = load_cifar10_ds(50)
    for i, d in enumerate(test_ds):
        print(i)

    x, y = next(iter(ds))
    # FIXME this is not [0,1]
    sample_and_view(x/255)
