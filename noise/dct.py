""" Discrete Cosine Transformation """
import math

import tensorflow as tf
import numpy as np


class DCTKernels(tf.keras.initializers.Initializer):
    """
        Kernel Initializer for DCT / IDCT 2D Convolutions
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, shape, dtype=None):

        assert len(shape) == 4, \
            "shape must be of rank 4, is: {}".format(shape)

        assert shape[0] == shape[1], \
            "shape[0] and shape[1] must be identical, shape is: {}".format(
                shape)

        assert shape[-1] == (shape[0] * shape[1]), \
            "shape[0] * shape[1] must equal shape[-1], shape is: {}".format(
                shape)

        block_size = shape[0]

        kernel = self._dct_kernel(block_size, self.normalize)

        return kernel

    def _dct(self, x, y, u, v, n):
        """ calculate discrete cosine transformation """
        a = tf.math.cos(((2 * x + 1) * u * math.pi) / (2 * n))
        b = tf.math.cos(((2 * y + 1) * v * math.pi) / (2 * n))
        return a * b

    def _dct_kernel(self, n, normalize):
        """ Build DCT 2D Convolutional Kernels """
        full_kernel = (n * n, n * n)
        G = np.zeros(shape=full_kernel)
        for x in range(0, n):
            for y in range(0, n):
                for u in range(0, n):
                    for v in range(0, n):
                        val = self._dct(x, y, u, v, n)
                        if normalize:
                            val *= self._normalize(u, v)
                        x_coord = n * u + v
                        y_coord = n * x + y
                        G[x_coord, y_coord] = val

        G = tf.cast(tf.Variable(G), tf.float32)

        G_filter = tf.reshape(G, shape=(n * n, n, n))

        G_filter_conv = tf.transpose(G_filter, perm=[1, 2, 0])

        G_filter_conv = tf.expand_dims(G_filter_conv, 2)

        return G_filter_conv

    def _norm_factor(self, a):
        if a == 0.0:
            return 1.0 / tf.sqrt(2.0)
        else:
            return 1.0

    def _normalize(self, u, v):
        return 0.25 * self._norm_factor(u) * self._norm_factor(v)


class DCT2D(tf.keras.layers.DepthwiseConv2D):
    """ DCT 2D Map """

    def __init__(self, n, normalize=True, **kwargs):
        initializer = DCTKernels(normalize=normalize)
        super(DCT2D, self).__init__(
            **kwargs,
            kernel_size=(n, n),
            strides=(n, n),
            depth_multiplier=n * n,
            depthwise_initializer=initializer,
            use_bias=False,
            trainable=False)

    def _mask_filters(self, res_channel, mask):
        """ Mask filters according to mask """
        mask = tf.reshape(
            mask, shape=(res_channel.shape[-1], ))
        mask = tf.cast(mask, tf.float32)
        return tf.multiply(res_channel, mask)

    def __call__(self, inputs, masks=None):
        """
            Args:
                inputs: tensor (batch, x, y, n x n, c)
                masks: list of c (n x n) binary masks
        """
        # process inputs channel-wise
        n_channels = inputs.shape[-1]

        if masks is not None:
            assert len(masks) == n_channels, \
                "length of masks ({}) must equal n_channels ({})".format(
                    len(masks), n_channels)

        res = list()
        splits = tf.split(inputs, n_channels, -1)

        for i, split in enumerate(splits):
            res_channel = super(DCT2D, self).__call__(split)
            if masks is not None:
                res_channel = self._mask_filters(res_channel, masks[i])
            res.append(res_channel)

        return tf.concat(res, -1)


class InverseDCT2D(tf.keras.layers.Conv2DTranspose):
    """ Inverse Discrete Cosine Transformation in 2D """

    def __init__(self, n, normalize=True):
        self.n = n
        self.filters_per_channel = n * n
        initializer = DCTKernels(normalize=normalize)
        super(InverseDCT2D, self).__init__(
            filters=1,
            kernel_size=(n, n),
            strides=(n, n),
            kernel_initializer=initializer,
            use_bias=False,
            trainable=False)

    def __call__(self, inputs):

        # process inputs channel-wise
        n_filters = inputs.shape[-1]
        n_channels = n_filters // self.filters_per_channel

        res = list()
        splits = tf.split(inputs, n_channels, -1)
        for split in splits:
            res.append(super(InverseDCT2D, self).__call__(split))

        return tf.concat(res, -1)
