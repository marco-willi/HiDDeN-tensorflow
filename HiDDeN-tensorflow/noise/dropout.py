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

        if keep_probability != 0.5:
            raise NotImplementedError(
                "Only keep_prob == 0.5 implemented")

        mask2d = tf.random.uniform(
            inputs.shape[1:3], minval=0, maxval=2, dtype=tf.int32)

        mask3d = tf.broadcast_to(
            tf.expand_dims(mask2d, -1), shape=inputs.shape[1:])
        mask3d = tf.cast(mask3d, tf.dtypes.float32)

        negative_mask3d = tf.math.abs(mask3d - 1)

        masked_inputs = self.mult_layer([inputs, mask3d])
        masked_backgrounds = self.mult_layer([backgrounds, negative_mask3d])

        combined = tf.math.add(masked_inputs, masked_backgrounds)

        return combined
