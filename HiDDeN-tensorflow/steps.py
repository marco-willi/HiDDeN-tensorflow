""" Train / Eval Step """
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


@tf.function
def train(
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
        loss_distortion = losses.distortion_loss(
            cover_images, encoder_decoder_output['encoded_image'])

        loss_recover = losses.recovery_loss(
            messages, encoder_decoder_output['decoded_message'])

        loss_adversarial = tf.reduce_mean(
            losses.discriminator_loss(discriminator_on_encoded, 0.0))

        # loss of Discriminator
        loss_discriminator_cover = losses.discriminator_loss(
            discriminator_on_cover, 0.0)

        loss_discriminator_encoded = losses.discriminator_loss(
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

        # Other Metrics
        bit_error_rate = tf.reduce_mean(tf.math.abs(
            messages - losses.binarize(encoder_decoder_output['decoded_message'])))

        true_negative_rate = tf.reduce_mean(
            losses.binarize(1 - tf.sigmoid(discriminator_on_cover)))

        true_positive_rate = tf.reduce_mean(
            losses.binarize(tf.sigmoid(discriminator_on_encoded)))

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

    losses_dict = {
        'loss_encoder_decoder': loss_encoder_decoder,
        'loss_discriminator': loss_discriminator,
        'loss_recover': loss_recover,
        'loss_distortion': loss_distortion,
        'loss_adversarial': loss_adversarial,
        'loss_discriminator_cover': tf.reduce_mean(
            loss_discriminator_cover),
        'loss_discriminator_encoded': tf.reduce_mean(
            loss_discriminator_encoded)}

    metrics_dict = {
        'metric_bit_error_rate': bit_error_rate,
        'metric_true_negative_rate': true_negative_rate,
        'metric_true_positive_rate': true_positive_rate}

    return losses_dict, metrics_dict
