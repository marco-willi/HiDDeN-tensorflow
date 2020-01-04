""" Loss Calculations
    Output: mean per sample loss
"""
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS


def image_distortion_loss(y, y_hat):
    """ MSE (per image pair) """
    return tf.reduce_mean(tf.math.pow(y - y_hat, 2), axis=[1, 2, 3])


def message_distortion_loss(y, y_hat):
    """ MSE (per image/message)"""
    return tf.reduce_mean(tf.math.pow(y - y_hat, 2), axis=[1])


def classification_loss(logits, y_expected):
    """ Classification loss given static y_expected """
    y_true = tf.fill(logits.shape, y_expected)
    return tf.keras.losses.binary_crossentropy(
        y_true=y_true,
        y_pred=logits,
        from_logits=True)


@tf.function
def step_loss(
        cover_images,
        messages,
        encoder_decoder_output,
        discriminator_on_cover,
        discriminator_on_encoded):
    """ Calculate Loss for one Step """

    losses = dict()

    losses['image_distortion'] = image_distortion_loss(
        cover_images, encoder_decoder_output['encoded_image'])

    # Recovery loss: difference encoded/decoded messages
    losses['message_distortion'] = message_distortion_loss(
        messages, encoder_decoder_output['decoded_message'])

    losses['adversarial'] = classification_loss(
        logits=discriminator_on_encoded,
        y_expected=0.0)

    # Discriminator loss: how well to recognize cover images
    losses['discriminator_cover'] = classification_loss(
        logits=discriminator_on_cover,
        y_expected=0.0)

    # Discriminator Loss: how well to recognize encoded images
    losses['discriminator_encoded'] = classification_loss(
        logits=discriminator_on_encoded,
        y_expected=1.0)

    # total loss discriminator
    losses['discriminator_total'] = tf.add(
        losses['discriminator_cover'],
        losses['discriminator_encoded'])

    # total loss encoder decoder
    loss_encoder_decoder_total = \
        FLAGS.loss_weight_recovery * losses['message_distortion'] + \
        FLAGS.loss_weight_distortion * losses['image_distortion'] + \
        FLAGS.loss_weight_adversarial * losses['adversarial']
    losses['encoder_decoder_total'] = loss_encoder_decoder_total

    return losses
