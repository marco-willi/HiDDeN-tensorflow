import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_examples(n_samples, images):
    """ Plot n_samples from list of image batches """
    n_cols = len(images)
    index = 0
    for j in range(0, n_samples):
        for col_id, img in enumerate(images):
            img_to_plot = img[j:j+1, :, :, :]
            img_to_plot = np.squeeze(img_to_plot)
            plt.subplot(n_samples, n_cols, index + col_id + 1)
            plt.imshow(img_to_plot, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
        index += n_cols


def create_summary_writers(logdir):
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    eval_writer = tf.summary.create_file_writer(os.path.join(logdir, 'eval'))
    test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'eval'))
    return {'train': train_writer, 'eval': eval_writer, 'test': test_writer}


def create_messages(batch_size, msg_length):
    messages = tf.random.uniform(
        [batch_size, msg_length], minval=0, maxval=2, dtype=tf.int32)
    messages = tf.cast(messages, dtype=tf.float32)
    return messages
