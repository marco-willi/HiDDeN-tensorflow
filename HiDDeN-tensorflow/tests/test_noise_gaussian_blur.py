import tensorflow as tf
import numpy as np

from noise import gaussian


class GaussianKernelTests(tf.test.TestCase):

    def setUp(self):
        """ Compare with Values from
            https://en.wikipedia.org/wiki/Gaussian_blur
        """

        self.wiki_example = np.array([
            [0.00000067, 0.00002292, 0.00019117, 0.00038771,
             0.00019117, 0.00002292, 0.00000067],
            [0.00002292, 0.00078633, 0.00655965, 0.01330373,
             0.00655965, 0.00078633, 0.00002292],
            [0.00019117, 0.00655965, 0.05472157, 0.11098164,
             0.05472157, 0.00655965, 0.00019117],
            [0.00038771, 0.01330373, 0.11098164, 0.22508352,
             0.11098164, 0.01330373, 0.00038771],
            [0.00019117, 0.00655965, 0.05472157, 0.11098164,
             0.05472157, 0.00655965, 0.00019117],
            [0.00002292, 0.00078633, 0.00655965, 0.01330373,
             0.00655965, 0.00078633, 0.00002292],
            [0.00000067, 0.00002292, 0.00019117, 0.00038771,
             0.00019117, 0.00002292, 0.00000067]
        ])
        self.wiki_sigma = 0.84089642

    def testGaussianKernelWikiExample(self):

        blur = gaussian.GaussianBlurr2D(sigma=self.wiki_sigma)

        expected = np.expand_dims(self.wiki_example, -1)
        expected = np.expand_dims(expected, -1)

        actual = blur(expected.shape)
        actual.shape

        expected - actual

        ratio = expected / actual

        with self.cached_session(use_gpu=False):
            self.assertAllInRange(ratio, 0.99, 1.01)

    def testIdenticalAcrossChannels(self):
        blur = gaussian.GaussianBlurr2D(sigma=self.wiki_sigma)
        actual = blur((7, 7, 2, 1))

        with self.cached_session(use_gpu=False):
            self.assertAllEqual(actual[:, :, 0, 0], actual[:, :, 1, 0])

    def testGaussianKernel1D(self):
        bl = gaussian.GaussianBlurring2D(
            sigma=self.wiki_sigma, kernel_size=(7, 7),
            padding="valid")

        with self.cached_session(use_gpu=False):
            inputs = tf.ones((1, 7, 7, 1))
            outputs = bl(inputs)
            self.assertAlmostEqual(
                tf.reduce_sum(outputs).numpy(), 1, places=5)

    def testGaussianKernel2D(self):
        bl = gaussian.GaussianBlurring2D(
            sigma=self.wiki_sigma, kernel_size=(7, 7),
            padding="valid")

        with self.cached_session(use_gpu=False):
            inputs = tf.ones((1, 7, 7, 2))
            outputs = bl(inputs).numpy()
            self.assertAlmostEqual(tf.reduce_sum(
                outputs[0, 0, 0, 0]).numpy(), 1, places=5)
            self.assertAlmostEqual(tf.reduce_sum(
                outputs[0, 0, 0, 1]).numpy(), 1, places=5)
