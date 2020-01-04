import tensorflow as tf


class Dropout(tf.keras.layers.Layer):
    """ Randomly replace pixels from inputs with pixels from backgrounds
    """

    def __init__(self, **kwargs):
        super(Dropout, self).__init__(trainable=False, **kwargs)
        self.mult_layer = tf.keras.layers.Lambda(
            lambda x: tf.math.multiply(x[0], x[1]), trainable=False)

    def call(self, inputs, keep_probability=0.5):

        inputs, backgrounds = inputs

        assert keep_probability >= 0.0, \
            "keep_probability must not be negative, is: {}".format(
                keep_probability)

        assert keep_probability <= 1.0, \
            "keep_probability must be <= 1.0, is: {}".format(
                keep_probability)
        
        drop_probability = 1 - keep_probability

        mask2d = tf.random.uniform(
            inputs.shape[1:3], minval=0, maxval=1, dtype=tf.float32)

        mask3d = tf.broadcast_to(
            tf.expand_dims(mask2d, -1), shape=inputs.shape[1:])

        keep_mask = tf.math.ceil(
            tf.clip_by_value(mask3d - drop_probability, 0, 1))
        drop_mask = tf.math.abs(keep_mask - 1)

        masked_inputs = self.mult_layer([inputs, keep_mask])
        masked_backgrounds = self.mult_layer([backgrounds, drop_mask])

        combined = tf.math.add(masked_inputs, masked_backgrounds)

        return combined
