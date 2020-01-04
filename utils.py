import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_examples(n_samples, images, start_image=0):
    """ Plot n_samples from list of image batches """
    n_cols = len(images)
    index = 0
    for j in range(0+start_image, n_samples+start_image):
        for col_id, img in enumerate(images):
            img_to_plot = img[j:j+1, :, :, :]
            img_to_plot = np.squeeze(img_to_plot)
            plt.subplot(n_samples, n_cols, index + col_id + 1)
            plt.imshow(img_to_plot, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
        index += n_cols


def create_summary_writers(path, writers=['train', 'val']):
    """ Create Different Summary File Writers """
    out = dict()
    for writer in writers:
        out[writer] = tf.summary.create_file_writer(
            os.path.join(path, writer))
    return out


def create_messages(batch_size, msg_length):
    messages = tf.random.uniform(
        [batch_size, msg_length], minval=0, maxval=2, dtype=tf.int32)
    messages = tf.cast(messages, dtype=tf.float32)
    return messages


def fix_path(path):
    """ Ensure path is resolved correctly """
    return os.path.abspath(os.path.expanduser(path))


def calc_difference_img(img_a, img_b):
    return tf.math.abs(tf.subtract(img_a, img_b)) * 10.0


def prep_imgs_to_plot(
        cover,
        encoded,
        transmitted_encoded,
        transmitted_cover,
        transform_fn=None):

    difference = calc_difference_img(cover, encoded)

    images = [
        cover,
        encoded,
        difference,
        transmitted_encoded,
        transmitted_cover]

    names = [
        'cover_images',
        'encoded_images',
        'difference_images',
        'transmitted_encoded_images',
        'transmitted_cover_images']

    descriptions = [
        'Cover Images',
        'Encoded Images',
        'Abslute Diff. Coded/Encoded Images (magnified)',
        'Transmitted Encoded Images',
        'Transmitted Cover Images']

    if transform_fn is not None:
        images = [transform_fn(x) for x in images]

    return {'images': images, 'names': names, 'descriptions': descriptions}


def summary_images(
        cover,
        encoded,
        transmitted_encoded,
        transmitted_cover,
        step,
        transform_fn=None):
    """ Create Summaries of different images for visualization """

    imgs = prep_imgs_to_plot(
        cover, encoded, transmitted_encoded, transmitted_cover, transform_fn)

    images_to_plot = imgs['images']

    names_to_plot = imgs['names']

    descriptions_to_plot = imgs['descriptions']

    for i, name in enumerate(names_to_plot):
        tf.summary.image(
            name=name,
            data=images_to_plot[i],
            step=step,
            max_outputs=6,
            description=descriptions_to_plot[i]
        )
