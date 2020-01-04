""" Track Metrics """
from collections import OrderedDict

import tensorflow as tf


_LOSSES_ = [
    'image_distortion',
    'message_distortion',
    'adversarial',
    'discriminator_cover',
    'discriminator_encoded',
    'discriminator_total',
    'encoder_decoder_total']


def binarize(x):
    """ Clip values to 0 and 1 and round to nearest integer (0, 1)"""
    return tf.math.round(tf.clip_by_value(
        x, clip_value_min=0, clip_value_max=1))


def decoding_error_rate(y, y_hat):
    """ Mean Binary Error Rate """
    return tf.reduce_mean(tf.math.abs(y - binarize(y_hat)), axis=[1])


class MetricsTracker(object):
    """ Create / Update / Reset Metrics """

    def __init__(self):

        self._losses = {'loss_{}'.format(x): tf.keras.metrics.Mean()
                        for x in _LOSSES_}

        self.metrics = {
            'metric_decoder_bit_error_rate':
                tf.keras.metrics.MeanAbsoluteError(),
            'metric_discriminator_precision':
                tf.keras.metrics.Precision(thresholds=0.5),
            'metric_discriminator_recall':
                tf.keras.metrics.Recall(thresholds=0.5)}

        self.metrics = {**self.metrics, **self._losses}

    def update(self,
               losses,
               messages,
               encoder_decoder_output,
               discriminator_on_cover,
               discriminator_on_encoded):

        y_true_discriminator = tf.concat([
            tf.ones_like(discriminator_on_encoded),
            tf.zeros_like(discriminator_on_cover)],
            axis=-1)

        y_pred_discriminator = tf.concat([
            tf.sigmoid(discriminator_on_encoded),
            1 - tf.sigmoid(discriminator_on_cover)],
            axis=-1)

        self.metrics['metric_decoder_bit_error_rate'].update_state(
            y_true=messages,
            y_pred=binarize(encoder_decoder_output['decoded_message'])
        )

        self.metrics['metric_discriminator_precision'].update_state(
            y_true_discriminator,
            y_pred_discriminator)

        self.metrics['metric_discriminator_recall'].update_state(
            y_true_discriminator,
            y_pred_discriminator)

        # update losses
        for name, value in losses.items():
            self.metrics['loss_{}'.format(name)].update_state(value)

    def results(self):
        res = {x: v.result().numpy() for x, v in self.metrics.items()}
        metrics_sorted = sorted(res.keys())
        return OrderedDict([(k, res[k]) for k in metrics_sorted])

    def reset(self):
        for metric in self.metrics.values():
            metric.reset_states()
