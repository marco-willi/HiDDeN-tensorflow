""" Train / Eval Step """
import tensorflow as tf
from absl import flags

from losses import (
    distortion_loss, recovery_loss, discriminator_loss,
    binarize
)


FLAGS = flags.FLAGS


@tf.function
def calculate_step_losses(
        cover_images,
        messages,
        encoder_decoder_output,
        discriminator_on_cover,
        discriminator_on_encoded):

    loss_dict = dict()

    # Distortion loss: difference cover image and encoded image
    loss_dict['loss_encoder_distortion'] = distortion_loss(
        cover_images, encoder_decoder_output['encoded_image'])

    # Recovery loss: difference encoded/decoded messages
    loss_dict['loss_decoder_recovery'] = recovery_loss(
        messages, encoder_decoder_output['decoded_message'])

    # Adversarial Loss: how well to fool the adversary
    loss_dict['loss_adversarial'] = tf.reduce_mean(
        discriminator_loss(discriminator_on_encoded, 0.0))

    # Discriminator loss: how well to recognize cover images
    loss_dict['loss_discriminator_cover'] = tf.reduce_mean(
        discriminator_loss(discriminator_on_cover, 0.0))

    # Discriminator Loss: how well to recognize encoded images
    loss_dict['loss_discriminator_encoded'] = tf.reduce_mean(
        discriminator_loss(discriminator_on_encoded, 1.0))

    # total loss discriminator
    loss_dict['loss_discriminator_total'] = tf.add(
        loss_dict['loss_discriminator_cover'],
        loss_dict['loss_discriminator_encoded'])

    # total loss encoder decoder
    loss_encoder_decoder_total = \
        FLAGS.loss_weight_recovery * loss_dict['loss_decoder_recovery'] + \
        FLAGS.loss_weight_distortion * loss_dict['loss_encoder_distortion'] + \
        FLAGS.loss_weight_adversarial * loss_dict['loss_adversarial']
    loss_dict['loss_encoder_decoder_total'] = tf.squeeze(
        loss_encoder_decoder_total)

    return loss_dict


def calculate_step_metrics(
        messages,
        encoder_decoder_output,
        discriminator_on_cover,
        discriminator_on_encoded):

    # Other Metrics
    bit_error_rate = tf.reduce_mean(tf.math.abs(
        messages - binarize(encoder_decoder_output['decoded_message'])))

    y_true_discriminator = tf.concat([
        tf.ones_like(discriminator_on_encoded),
        tf.zeros_like(discriminator_on_cover)],
        axis=-1)
    y_pred_discriminator = tf.concat([
        tf.sigmoid(discriminator_on_encoded),
        1 - tf.sigmoid(discriminator_on_cover)],
        axis=-1)
    precision_metric = tf.keras.metrics.Precision(thresholds=0.5)
    precision_metric.update_state(
        y_true_discriminator,
        y_pred_discriminator)

    precision_discriminator = precision_metric.result()

    recall_metric = tf.keras.metrics.Recall(thresholds=0.5)
    recall_metric.update_state(
        y_true_discriminator,
        y_pred_discriminator)

    recall_discriminator = recall_metric.result()

    metrics_dict = {
        'metric_decoder_bit_error_rate': bit_error_rate,
        'metric_discriminator_precision': precision_discriminator,
        'metric_discriminator_recall': recall_discriminator}

    return metrics_dict


@tf.function
def train(
        cover_images,
        messages,
        encoder_decoder,
        discriminator,
        training,
        optimizer_encoder_decoder=None,
        optimizer_discriminator=None):

    with tf.GradientTape() as tape_ed, tf.GradientTape() as tape_adv:

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=training)

        discriminator_on_cover = discriminator(
            inputs={'image': cover_images},
            training=training)

        discriminator_on_encoded = discriminator(
            inputs={'image': encoder_decoder_output['encoded_image']},
            training=training)

        loss_dict = calculate_step_losses(
            cover_images,
            messages,
            encoder_decoder_output,
            discriminator_on_cover,
            discriminator_on_encoded)

    # gradient updates
    if training:
        grads_encoder_decoder = tape_ed.gradient(
            loss_dict['loss_encoder_decoder_total'],
            encoder_decoder.trainable_variables)
        optimizer_encoder_decoder.apply_gradients(
            zip(grads_encoder_decoder, encoder_decoder.trainable_variables))

        grads_discriminator = tape_adv.gradient(
            loss_dict['loss_discriminator_total'],
            discriminator.trainable_variables)
        optimizer_discriminator.apply_gradients(
            zip(grads_discriminator, discriminator.trainable_variables))

    outputs = {
        'encoder_decoder': encoder_decoder_output,
        'discriminator_on_cover': discriminator_on_cover,
        'discriminator_on_encoded': discriminator_on_encoded}

    return outputs



@tf.function
def train_old(
        cover_images,
        messages,
        encoder_decoder,
        discriminator,
        losses,
        training,
        optimizer_encoder_decoder=None,
        optimizer_discriminator=None
        ):

    with tf.GradientTape() as tape_ed, tf.GradientTape() as tape_adv:

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=training)

        discriminator_on_cover = discriminator(
            inputs={'image': cover_images},
            training=training)

        discriminator_on_encoded = discriminator(
            inputs={'image': encoder_decoder_output['encoded_image']},
            training=training)

        # Loss of Encoder Decoder
        loss_distortion = distortion_loss(
            cover_images, encoder_decoder_output['encoded_image'])

        loss_recover = recovery_loss(
            messages, encoder_decoder_output['decoded_message'])

        loss_adversarial = tf.reduce_mean(
            discriminator_loss(discriminator_on_encoded, 0.0))

        # loss of Discriminator
        loss_discriminator_cover = discriminator_loss(
            discriminator_on_cover, 0.0)

        loss_discriminator_encoded = discriminator_loss(
            discriminator_on_encoded, 1.0)

        # total loss discriminator
        loss_discriminator = tf.reduce_mean(
            loss_discriminator_cover + loss_discriminator_encoded)

        # total loss encoder decoder
        loss_encoder_decoder = \
            FLAGS.loss_weight_recover * loss_recover + \
            FLAGS.loss_weight_distortion * loss_distortion + \
            FLAGS.loss_weight_adversarial * loss_adversarial
        loss_encoder_decoder = tf.squeeze(loss_encoder_decoder)

    # gradient updates
    if training:
        grads_encoder_decoder = tape_ed.gradient(
            loss_encoder_decoder, encoder_decoder.trainable_variables)
        optimizer_encoder_decoder.apply_gradients(
            zip(grads_encoder_decoder, encoder_decoder.trainable_variables))

        grads_discriminator = tape_adv.gradient(
            loss_discriminator, discriminator.trainable_variables)
        optimizer_discriminator.apply_gradients(
            zip(grads_discriminator, discriminator.trainable_variables))

    # Other Metrics
    bit_error_rate = tf.reduce_mean(tf.math.abs(
        messages - binarize(encoder_decoder_output['decoded_message'])))
    
    true_negative_rate = tf.reduce_mean(
        binarize(1 - tf.sigmoid(discriminator_on_cover)))

    true_positive_rate = tf.reduce_mean(
        binarize(tf.sigmoid(discriminator_on_encoded)))
    
    losses_dict = {
        'loss_total_encoder_decoder': loss_encoder_decoder,
        'loss_total_discriminator': loss_discriminator,
        'loss_decoder_message_recovery': loss_recover,
        'loss_encoder_distortion': loss_distortion,
        'loss_encoder_fool_adversary': loss_adversarial,
        'loss_discriminator_recognize_cover': tf.reduce_mean(
            loss_discriminator_cover),
        'loss_discriminator_recognize_encoded': tf.reduce_mean(
            loss_discriminator_encoded)}

    metrics_dict = {
        'metric_decoder_bit_error_rate': bit_error_rate,
        'metric_discriminator_true_negative_rate': true_negative_rate,
        'metric_discriminator_true_positive_rate': true_positive_rate}

    return losses_dict, metrics_dict
