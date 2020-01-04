""" Train / Eval Step """
import tensorflow as tf

from losses import step_loss


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

        loss_dict = step_loss(
            cover_images,
            messages,
            encoder_decoder_output,
            discriminator_on_cover,
            discriminator_on_encoded)

    # gradient updates
    if training:
        grads_encoder_decoder = tape_ed.gradient(
            loss_dict['encoder_decoder_total'],
            encoder_decoder.trainable_variables)
        optimizer_encoder_decoder.apply_gradients(
            zip(grads_encoder_decoder, encoder_decoder.trainable_variables))

        grads_discriminator = tape_adv.gradient(
            loss_dict['discriminator_total'],
            discriminator.trainable_variables)
        optimizer_discriminator.apply_gradients(
            zip(grads_discriminator, discriminator.trainable_variables))

    outputs = {
        'encoder_decoder': encoder_decoder_output,
        'discriminator_on_cover': discriminator_on_cover,
        'discriminator_on_encoded': discriminator_on_encoded}

    return outputs
