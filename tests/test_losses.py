import math

import tensorflow as tf

import losses


class LossesTests(tf.test.TestCase):

    def testDistortionLossBatch(self):

        batch_size = 5

        with self.cached_session(use_gpu=True):
            shape = (batch_size, 32, 32, 3)
            deviation = 0.1

            cover_image = tf.ones(shape)
            encoded_image = tf.ones(shape)
            encoded_image = tf.add(encoded_image, deviation)

            actual = losses.image_distortion_loss(cover_image, encoded_image)
            expected = tf.constant(
                [math.pow(deviation, 2) for _ in range(0, batch_size)])

            self.assertAllClose(actual, expected, rtol=1e-06)

    def testDistortionLossSingleImage(self):

        with self.cached_session(use_gpu=True):
            y = tf.constant(
                [[1, 0, 1, 0],
                 [1, 0, 0, 1]], dtype=tf.float32)
            y_hat = tf.constant(
                [[1, 0, 1, 1],
                 [1, 0, 2, 1]], dtype=tf.float32)
            
            y = tf.expand_dims(y, 0)
            y_hat = tf.expand_dims(y_hat, 0)

            y = tf.expand_dims(y, -1)
            y_hat = tf.expand_dims(y_hat, -1)

            actual = losses.image_distortion_loss(y, y_hat)

            expected = tf.constant(
                [5 / 8], dtype=tf.float32)
   
            self.assertAllEqual(actual, expected)

