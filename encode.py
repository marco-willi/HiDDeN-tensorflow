""" Encode Images """
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

exp_id = 'exp1_cd_jpeg_mask'
FLAGS.noise_type = 'jpeg_mask'


FLAGS.tbdir = os.path.join('./tmp/tensorboard/', exp_id)
FLAGS.plotdir = os.path.join('./tmp/plots/', exp_id)
FLAGS.ckptdir = os.path.join('./tmp/ckpts/', exp_id)
FLAGS.logdir = os.path.join('./tmp/logs/', exp_id)

FLAGS.dataset = 'dir'
FLAGS.train_dir = 'D:\Kaggle\cats_and_dogs\hidden\\train\\'
FLAGS.val_dir = 'D:\Kaggle\cats_and_dogs\hidden\\val\\'
FLAGS.test_dir = 'D:\Kaggle\cats_and_dogs\hidden\\test\\'
FLAGS.train_crop = [64, 64, 3]
FLAGS.batch_size = 12
FLAGS.epochs = 50
FLAGS.to_yuv = True
FLAGS.loss_weight_distortion = 10


status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# prepare checkpointer
ckpt = tf.train.Checkpoint(
    step=step,
    epoch=epoch,
    optimizer_encoder_decoder=optimizer_encoder_decoder,
    optimizer_discriminator=optimizer_discriminator,
    encoder_decoder=encoder_decoder,
    discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, utils.fix_path(FLAGS.ckptdir), max_to_keep=FLAGS.keep_ckpts)

if ckpt_manager.latest_checkpoint is not None:
