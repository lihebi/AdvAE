import sys
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

sys.path.append('./tf_models')

# pip3 install --user git+https://www.github.com/keras-team/keras-contrib.git
# from keras_contrib.keras_contrib.applications.wide_resnet import WideResidualNetwork
# from keras_contrib.applications.wide_resnet import WideResidualNetwork
# from keras_contrib.applications.densenet import DenseNetFCN

# add tf_models to sys.path

from official.vision.image_classification import cifar_preprocessing
from official.vision.image_classification import common
from official.vision.image_classification import resnet_cifar_model

from data_utils import load_cifar10, sample_and_view

