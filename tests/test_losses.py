import tensorflow as tf

import losses


class LossesTests(tf.test.TestCase):

    def testDistortionLoss(self):

        with self.cached_session(use_gpu=True):
            shape = (5, 32, 32, 3)
            deviation = 0.1

            cover_image = tf.ones(shape)
            encoded_image = tf.ones(shape)
            encoded_image = tf.add(encoded_image, deviation)

            actual = losses.distortion_loss(cover_image, encoded_image).numpy()
            expected = tf.math.pow(deviation, 2).numpy()

            self.assertAlmostEqual(actual, expected, places=5)
