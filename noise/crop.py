import tensorflow as tf


class Crop(tf.keras.layers.Layer):
    """ Crop a random Square of size crop_proportion from inputs.
        Applies same crop over a batch of inputs.
    """

    def __init__(self, **kwargs):
        super(Crop, self).__init__(trainable=False, **kwargs)

    def call(self, inputs, crop_proportion=0.5):

        if crop_proportion == 1.0:
            return inputs
        if crop_proportion == 0.0:
            return tf.zeros_like(inputs)

        x_dim = inputs.shape[1]
        y_dim = inputs.shape[2]

        crop_width = tf.cast(
            tf.math.sqrt((x_dim * y_dim * crop_proportion)), tf.int32)

        max_x_coord = tf.cast(tf.subtract(x_dim, crop_width), tf.int32)
        max_y_coord = tf.cast(tf.subtract(y_dim, crop_width), tf.int32)

        x_start = tf.random.uniform(
            (), minval=0,
            maxval=max_x_coord,
            dtype=tf.int32)

        y_start = tf.random.uniform(
            (), minval=0,
            maxval=max_y_coord,
            dtype=tf.int32)

        inputs_cropped = tf.image.crop_to_bounding_box(
            inputs,
            y_start,
            x_start,
            crop_width,
            crop_width
        )

        return inputs_cropped
