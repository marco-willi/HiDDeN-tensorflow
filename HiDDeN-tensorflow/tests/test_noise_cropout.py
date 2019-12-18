import tensorflow as tf

from noise import cropout


class CropoutTest(tf.test.TestCase):

    def setUp(self):
        self.layer = cropout.Cropout()

    def testCropProportions(self):
        shapes = [(1, 28, 28, 1), (2, 28, 28, 1),
                  (1, 28, 28, 3), (2, 28, 28, 3),
                  (2, 33, 33, 3)]

        props = [0.0, 0.25, 0.5, 0.75, 1.0]

        with self.cached_session(use_gpu=False):

            for shape in shapes:
                inputs = tf.ones(shape)
                backgrounds = tf.zeros(shape)

                for prop in props:
                    res = self.layer((inputs, backgrounds), prop)
                    crop_width = tf.cast(
                        tf.sqrt((shape[1] * shape[2] * prop)), tf.int32)
                    expected = tf.cast((
                        crop_width * crop_width * shape[3] * shape[0]),
                        tf.float32)
                    actual = tf.reduce_sum(res)
                    self.assertEqual(actual, expected)
