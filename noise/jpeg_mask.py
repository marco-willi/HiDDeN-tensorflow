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
        self.n = n
    
    def build(self, input_shape):
        x_crop, y_crop = self._calc_crop_for_output(input_shape)
        self.crop = tf.keras.layers.Cropping2D(((0, y_crop), (0, x_crop)))
        self.padding = tf.keras.layers.ZeroPadding2D(
            ((0, y_crop), (0, x_crop)))

    def call(self, inputs):

        x = inputs

        n_channels = x.shape[-1]

        # pad if necessary
        x = self.padding(x)

        # DCT transformation
        x = self._center_img(x)
        x = self.dct_transform(x, self.masks[0:n_channels])
        x = tf.stack(tf.split(x, n_channels, axis=-1), -1)

        # Inverse DCT transformation
        x = tf.concat(tf.split(x, n_channels, axis=-1), -2)
        x = tf.squeeze(x, -1)
        x = self.dct_inverse(x)
        x = self._de_center_img(x)

        # crop if necessary
        x = self.crop(x)

        return x
    
    def _calc_crop_for_output(self, input_shape):
        """ Calculate how much to crop from output
            if during the DCT values had to be padded
        """
        x_short = input_shape[2] % self.n
        if x_short > 0:
            crop_x = self.n - x_short
        else:
            crop_x = 0
        y_short = input_shape[1] % self.n
        if y_short > 0:
            crop_y = self.n - y_short
        else:
            crop_y = 0
        return crop_x, crop_y

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
