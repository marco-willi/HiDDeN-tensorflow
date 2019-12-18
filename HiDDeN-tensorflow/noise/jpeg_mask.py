import tensorflow as tf
import numpy as np

from noise import dct


class JPEG_Mask(tf.keras.layers.Layer):
    """ Apply JPEG Mask  """

    def __init__(self, n, quality=50, **kwargs):
        super(JPEG_Mask, self).__init__(trainable=False, **kwargs)
        self.dct_transform = dct.DCT2D(n=n)
        self.dct_inverse = dct.InverseDCT2D(n=n)
        self.masks = self._create_masks(n, quality)

    def call(self, inputs):

        x = inputs

        n_channels = x.shape[-1]

        # DCT transformation
        x = self._center_img(inputs)
        x = self.dct_transform(x, self.masks[0:n_channels])
        x = tf.stack(tf.split(x, n_channels, axis=-1), -1)

        # Inverse DCT transformation
        x = tf.concat(tf.split(x, n_channels, axis=-1), -2)
        x = tf.squeeze(x, -1)
        x = self.dct_inverse(x)
        x = self._de_center_img(x)
        return x

    def _center_img(self, x):
        return tf.cast(x - 128, tf.float32)

    def _de_center_img(self, x):
        return tf.cast(x + 128, tf.float32)

    def _create_masks(self, n, quality=50):
        if quality != 50:
            raise NotImplementedError("Only quality=50 implemented")

        QY_MASK = np.zeros(shape=(n, n))
        QC_MASK = np.zeros(shape=(n, n))

        QY_MASK[0:5, 0:5] = 1
        QC_MASK[0:3, 0:3] = 1

        return [QY_MASK, QC_MASK, QC_MASK]
