import os
import time

import tensorflow as tf

from absl import app
from absl import logging
from cfg import flags

import utils
import nets
import steps
import datasets

FLAGS = flags.FLAGS


FLAGS.batch_size = 16
FLAGS.epochs = 20
FLAGS.to_yuv = True

FLAGS.dataset = 'dir'
FLAGS.train_dir = 'D:\Kaggle\cats_and_dogs\hidden\\train\\'
FLAGS.val_dir = 'D:\Kaggle\cats_and_dogs\hidden\\val\\'
FLAGS.test_dir = 'D:\Kaggle\cats_and_dogs\hidden\\test\\'
FLAGS.train_crop = [128, 128, 3]


def main(argv):
    dataset = datasets.ImageDir()
    input_shape = dataset.get_input_shape()
    dataset_train = dataset.create_train_dataset()


if __name__ == '__main__':
    app.run(main)
