import tensorflow as tf
import numpy as np

from noise import dropout


class DropOutTest(tf.test.TestCase):

    def setUp(self):
        self.layer = dropout.Dropout()

    def testEqualityAcrossChannels(self):
        """ Test that samples across channels are identical """

        # Test simple binary case
        input_image = tf.ones((1, 12, 12, 3))
        background_image = tf.zeros((1, 12, 12, 3))
        res = self.layer((input_image, background_image))
        channels = tf.split(res, res.shape[-1], axis=-1)

        with self.cached_session(use_gpu=False):
            self.assertAllEqual(channels[0], channels[1])
            self.assertAllEqual(channels[1], channels[2])

    def testSamplingProportion(self):
        input_image = tf.ones((1, 1000, 1000, 1))
        background_image = tf.zeros((1, 1000, 1000, 1))
        res = self.layer((input_image, background_image), 0.5)

        total_shape = np.prod(res.shape)

        with self.cached_session(use_gpu=False):
            actual = tf.reduce_sum(res) / (total_shape / 2)
            actual = actual.numpy()
            expected = 1.0
            self.assertAlmostEqual(actual, expected, places=2)

    def testMutabilityOnDifferentCalls(self):
        """ Confirm that different invocations of the layer
            lead to different samplings
        """
        input_image = tf.ones((1, 1000, 1000, 1))
        background_image = tf.zeros((1, 1000, 1000, 1))
        res1 = self.layer((input_image, background_image), 0.5)
        res2 = self.layer((input_image, background_image), 0.5)

        with self.cached_session(use_gpu=False):
            self.assertNotAllEqual(res1, res2)


if __name__ == '__main__':
    tf.test.main(argv=None)
