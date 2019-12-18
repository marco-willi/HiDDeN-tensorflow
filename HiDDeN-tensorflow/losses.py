""" Define Loss and Metric Functions"""
import tensorflow as tf


def distortion_loss(y, y_hat):
    """ Image Distortion Loss - L2 Norm """
    return tf.reduce_mean(tf.math.pow(y - y_hat, 2))


def recovery_loss(y, y_hat):
    """ Message Recovery Loss - L2 Norm """
    per_message = tf.reduce_mean(tf.math.pow(y - y_hat, 2), axis=0)
    return tf.reduce_mean(per_message)


def discriminator_loss(logits, y_expected):
    """ Discriminator Loss """
    y_true = tf.fill(logits.shape, y_expected)
    return tf.keras.losses.binary_crossentropy(
        y_true=y_true,
        y_pred=logits,
        from_logits=True)


def binarize(x):
    return tf.math.round(
        tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))


def decoding_error_rate(y, y_hat):
    """ Mean Error Rate across Batch """
    return tf.reduce_mean(tf.math.abs(y - binarize(y_hat)), axis=0)
