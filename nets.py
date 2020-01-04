"""
    Network Architecture
    - encoder
    - decoder
    - transmitter
    - discriminator
"""
import tensorflow as tf
from absl import flags

from noise import gaussian, dropout, cropout, crop, jpeg_mask

FLAGS = flags.FLAGS


def _kernel_initializer():
    """ Conv2D Kernel initializer """
    return FLAGS.cbr_initializer


def encoder(
        cover_image,
        message,
        input_shape,
        msg_length,
        n_convbnrelu_blocks):
    """ Create Encoder Net  """

    # Message Block
    m = tf.keras.layers.RepeatVector(
        input_shape[0] * input_shape[1])(message)
    m = tf.keras.layers.Reshape((input_shape[0:2]) + (msg_length, ))(m)

    # Image Processing Block
    x = cover_image

    for _ in range(0, n_convbnrelu_blocks):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=_kernel_initializer(),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    # Concatenate Message Block with Image Processing Block and Cover Image
    x = tf.keras.layers.Concatenate(axis=-1)([m, x, cover_image])

    # Encode Image
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=_kernel_initializer(),
        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    encoded_img = tf.keras.layers.Conv2D(
        filters=input_shape[-1],
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=_kernel_initializer(),
        padding='valid',
        activation='linear')(x)

    return encoded_img


def _apply_noise(x, cover_image, noise):
    """ Apply noise to image """

    if noise == 'identity':
        pass

    elif noise == 'gaussian':
        x = gaussian.GaussianBlurring2D(
            sigma=FLAGS.gaussian_sigma,
            kernel_size=(FLAGS.gaussian_kernel, FLAGS.gaussian_kernel),
            padding='same')(x)

    elif noise == 'dropout':
        x = dropout.Dropout()(
            (x, cover_image),
            keep_probability=FLAGS.dropout_p)

    elif noise == 'cropout':
        x = cropout.Cropout()(
            (x, cover_image),
            crop_proportion=FLAGS.cropout_p)

    elif noise == 'crop':
        x = crop.Crop()(x, crop_proportion=FLAGS.crop_p)

    elif noise == 'jpeg_mask':
        x = jpeg_mask.JPEG_Mask(n=8, quality=50)(x)

    else:
        raise NotImplementedError(
            "noise layer {} not implemented".format(noise))

    return x


def transmitter(encoded_image, cover_image, noise_layers):
    """ Transmitter: potentially noisy transmission of encoded_image """

    x = encoded_image

    for noise in noise_layers:
        x = _apply_noise(x, cover_image, noise)

    return x


def decoder(encoded_image, msg_length, n_convbnrelu_blocks):
    """ Decoder Net """

    x = encoded_image

    for _ in range(0, n_convbnrelu_blocks):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=_kernel_initializer(),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=msg_length,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=_kernel_initializer(),
        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    decoded_message = tf.keras.layers.Dense(msg_length)(x)
    return decoded_message


def encoder_decoder(
        input_shape,
        msg_length,
        noise_layers,
        n_convbnrelu_encoder,
        n_convbnrelu_decoder):
    """ EncoderDecoder Net """
    message = tf.keras.Input(shape=(msg_length, ), name='message')
    cover_image = tf.keras.Input(shape=input_shape, name='cover_image')

    encoded_image = encoder(
        cover_image, message, input_shape, msg_length, n_convbnrelu_encoder)
    # TODO: consider clipping image to valid range

    transmitted_encoded_image = transmitter(
        encoded_image, cover_image, noise_layers)

    transmitted_cover_image = transmitter(
        cover_image, cover_image, noise_layers)

    decoded_message = decoder(
        transmitted_encoded_image, msg_length, n_convbnrelu_decoder)

    model = tf.keras.Model(
        inputs={
            'cover_image': cover_image,
            'message': message},
        outputs={
            'encoded_image': encoded_image,
            'transmitted_encoded_image': transmitted_encoded_image,
            'transmitted_cover_image': transmitted_cover_image,
            'decoded_message': decoded_message})
    return model


def discriminator(input_shape, n_convbnrelu):
    """ Discriminator Net: Identify Encoded Images """

    image = tf.keras.Input(shape=input_shape)
    x = image

    for _ in range(0, n_convbnrelu):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=_kernel_initializer(),
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=image, outputs=logits)

    return model
