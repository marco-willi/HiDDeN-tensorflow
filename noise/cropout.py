import tensorflow as tf


class Cropout(tf.keras.layers.Layer):
    """ Crop a random Square of size crop_proportion from input
        and replaces remaining area with background.
        Applies same crop over a batch.
    """

    def __init__(self, **kwargs):
        super(Cropout, self).__init__(trainable=False, **kwargs)

    def call(self, inputs, crop_proportion=0.5):

        inputs, backgrounds = inputs

        crop_proportion = tf.cast(crop_proportion, tf.float32)

        if crop_proportion == 1.0:
            return inputs
        if crop_proportion == 0.0:
            return backgrounds

        x_dim = inputs.shape[1]
        y_dim = inputs.shape[2]
        c_dim = inputs.shape[3]

        crop_width = tf.cast(
            tf.math.sqrt((x_dim * y_dim * crop_proportion)), tf.int32)

        x_start = tf.random.uniform(
            (), minval=0,
            maxval=x_dim - crop_width,
            dtype=tf.int32)

        y_start = tf.random.uniform(
            (), minval=0,
            maxval=y_dim - crop_width,
            dtype=tf.int32)

        crop_mask = tf.ones((crop_width, crop_width, c_dim))

        inputs_mask = tf.image.pad_to_bounding_box(
            crop_mask,
            y_start,
            x_start,
            y_dim,
            x_dim
        )
        backgrounds_mask = tf.math.abs(inputs_mask - 1)

        inputs_masked = tf.multiply(inputs, inputs_mask)
        backgrounds_masked = tf.multiply(backgrounds, backgrounds_mask)

        combined = tf.add(inputs_masked, backgrounds_masked)

        return combined
