import tensorflow as tf

import metrics


class MetricsTests(tf.test.TestCase):

    def testBinaryErrorRate(self):

        with self.cached_session(use_gpu=True):
            y = tf.constant(
                    [[1, 0, 1, 0],
                     [1, 0, 1, 1]], dtype=tf.float32)
            y_hat = tf.constant(
                [[0.1, 0.9, 1, 0],
                 [1, 0, 0.8, 1]], dtype=tf.float32)

            actual = metrics.decoding_error_rate(y, y_hat)
            expected = tf.constant([0.50, 0])
            self.assertAllEqual(actual, expected)

    def testBinarize(self):

        with self.cached_session(use_gpu=True):

            x = tf.constant([1, 0, 0.5, 0.50001, 0.9, 0.4], dtype=tf.float32)
            actual = metrics.binarize(x)
            expected = tf.constant([1, 0, 0, 1, 1, 0], dtype=tf.float32)
            self.assertAllEqual(actual, expected)
