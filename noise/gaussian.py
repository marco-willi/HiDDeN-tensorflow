import tensorflow as tf
import tensorflow_probability as tfp


class GaussianBlurr2D(tf.keras.initializers.Initializer):
    """ Initializer that generates a 2D Gaussian Kernel
        Args:
            sigma: variance of the Gaussian
    """

    def __init__(self, sigma):
        if not sigma > 0:
            raise ValueError("'sigma' must be positive")

        self.sigma = tf.cast(sigma, tf.float32)
        self.dist = tfp.distributions.Normal(loc=0, scale=sigma)

    def __call__(self, shape, dtype=None):
        """ shape: (kernel[0], kenel[1], channels, depth_multiplier) """

        if len(shape) != 4:
            raise ValueError("'shape' must be 4 dimensional")

        if shape[0] != shape[1]:
            raise ValueError("'shape' must be square")

        if shape[0] < 3:
            raise ValueError("'shape' must be at least 3")

        if (shape[0] % 2) != 1:
            raise ValueError("'shape' must be an odd number")

        kernel_radius = shape[0] // 2

        probs = self.dist.prob(
            tf.range(-kernel_radius, kernel_radius + 1, 1, dtype=tf.float32))

        kernel_vals = tf.tile(probs, tf.expand_dims(probs.shape[0], -1))

        kernel_vals_2D = tf.reshape(
            kernel_vals, shape=(probs.shape[0], probs.shape[0]))

        kernel = tf.multiply(kernel_vals_2D, tf.transpose(kernel_vals_2D))

        assert kernel.shape[0:2] == shape[0:2], \
            "kernel shape does not match requested shape"

        kernel = kernel / tf.reduce_sum(kernel)

        # replicate kernel across channels
        dim = 2
        while len(kernel.shape) < len(shape):
            kernel = tf.expand_dims(kernel, -1)
            if dim == 2:
                kernel = tf.keras.backend.repeat_elements(
                    kernel, rep=shape[dim], axis=-1)
            dim += 1

        return kernel


class GaussianBlurring2D(tf.keras.layers.DepthwiseConv2D):
    """ Gaussian Blurring of 2D Map """

    def __init__(self, sigma=0.84, **kwargs):
        self.gaussian_blurr_initializer = GaussianBlurr2D(sigma=sigma)
        super(GaussianBlurring2D, self).__init__(
            **kwargs,
            strides=(1, 1),
            depth_multiplier=1,
            depthwise_initializer=self.gaussian_blurr_initializer,
            use_bias=False,
            trainable=False)

    def __call__(self, inputs):
        return super(GaussianBlurring2D, self).__call__(inputs)
