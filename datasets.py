""" Define Datasets """
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from absl import flags

import utils


FLAGS = flags.FLAGS


def get_dataset():
    if FLAGS.dataset == 'mnist':
        return MNIST
    elif FLAGS.dataset == 'cifar10':
        return CIFAR10
    elif FLAGS.dataset == 'dir':
        return ImageDir
    else:
        raise NotImplementedError(
            "Dataset type {} not implemented".format(FLAGS.dataset))


def transform_data(x):
    x = tf.expand_dims(x, -1)
    x = tf.cast(x, tf.float32)
    x /= 255.0
    return x


class Dataset(object):
 
    def get_input_shape(self):
        raise NotImplementedError

    def create_train_dataset(self):
        raise NotImplementedError

    def create_val_dataset(self):
        raise NotImplementedError

    def create_test_dataset(self):
        raise NotImplementedError


class MNIST(Dataset):

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        self.x_test = x_test
        self.x_train, self.x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.05, random_state=123)
        self.shuffle_buffer = 600000

    def get_input_shape(self):
        return (28, 28, 1)

    def create_train_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        dataset = dataset.map(lambda x: transform_data(x))
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_val_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
        dataset = dataset.map(lambda x: transform_data(x))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_test_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        dataset = dataset.map(lambda x: transform_data(x))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset


class CIFAR10(Dataset):

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.cifar10.load_data()
        self.x_test = x_test
        self.x_train, self.x_val, y_train, y_val = train_test_split(
            x_train, y_train, train_size=10000, test_size=300, random_state=123)
        self.shuffle_buffer = 1000
        self.to_yuv = FLAGS.to_yuv

    def get_input_shape(self):
        return (32, 32, 3)

    def create_train_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        dataset = dataset.map(lambda x: self._transform_data(x, self.to_yuv))
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_val_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
        dataset = dataset.map(lambda x: self._transform_data(x, self.to_yuv))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_test_dataset(self):
        """ Dataset Iterator """
        dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        dataset = dataset.map(lambda x: self._transform_data(x, self.to_yuv))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def _transform_data(self, x, to_yuv):
        x = tf.cast(x, tf.float32)
        x /= 255.0
        if to_yuv:
            x = tf.image.rgb_to_yuv(x)
        return x


class ImageDir(Dataset):

    def __init__(self):
        self.test = utils.fix_path(FLAGS.test_dir)
        self.train = utils.fix_path(FLAGS.train_dir)
        self.val = utils.fix_path(FLAGS.val_dir)
        self.shuffle_buffer = 1000
        self.to_yuv = FLAGS.to_yuv
        self.input_shape = tuple(FLAGS.train_crop)

        assert os.path.exists(self.train), \
            "Path train_dir: {} does not exist".format(self.train)

        assert os.path.exists(self.val), \
            "Path val_dir: {} does not exist".format(self.val)

    def get_input_shape(self):
        return self.input_shape

    def _transform_data(self, x, to_yuv, is_training):
        x = tf.io.read_file(x)
        x = tf.io.decode_jpeg(
            x, channels=3, try_recover_truncated=True)
        x = tf.cast(x, tf.float32)
        x = tf.divide(x, 255.0)
        if to_yuv:
            x = tf.image.rgb_to_yuv(x)
        if is_training:
            x = tf.image.random_crop(x, self.input_shape)
            #x = tf.image.random_flip_left_right(x)
        else:
            x = tf.image.resize_with_crop_or_pad(
                x, self.input_shape[0], self.input_shape[1])
        return x

    def create_train_dataset(self):
        """ Dataset Iterator """
        pattern = self.train + os.path.sep + '*'
        dataset = tf.data.Dataset.list_files(pattern, shuffle=True)
        dataset = dataset.map(
            lambda x: self._transform_data(x, self.to_yuv, True))
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_val_dataset(self):
        """ Dataset Iterator """
        pattern = self.val + os.path.sep + '*'
        dataset = tf.data.Dataset.list_files(pattern, shuffle=False)
        dataset = dataset.map(
            lambda x: self._transform_data(x, self.to_yuv, False))
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)
        return dataset

    def create_test_dataset(self):
        """ Dataset Iterator """
        pattern = self.test + os.path.sep + '*'
        dataset = tf.data.Dataset.list_files(pattern, shuffle=False)
        dataset = dataset.map(
            lambda x: self._transform_data(x, self.to_yuv, False))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=False)
        dataset = dataset.repeat(1)
        return dataset
