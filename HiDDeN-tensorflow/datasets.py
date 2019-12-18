""" Define Datasets """
import tensorflow as tf
from sklearn.model_selection import train_test_split

from absl import flags


FLAGS = flags.FLAGS


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
