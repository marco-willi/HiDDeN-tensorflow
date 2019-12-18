import tensorflow as tf

from noise import crop


class CropTest(tf.test.TestCase):

    def setUp(self):
        self.layer = crop.Crop()

    def testCropProportions(self):
        shapes = [(1, 28, 28, 1), (2, 28, 28, 1),
                  (1, 28, 28, 3), (2, 28, 28, 3),
                  (2, 33, 33, 3)]

        props = [0.25, 0.5, 0.75, 1.0]

        with self.cached_session(use_gpu=True):

            for shape in shapes:
                inputs = tf.ones(shape)
                total_input_area = tf.cast(
                    shape[0] * shape[1] * shape[2] * shape[3], tf.float32)

                for prop in props:
                    res = self.layer(inputs, prop)
                    expected = tf.cast(total_input_area * prop, tf.float32)
                    actual = tf.reduce_sum(res)
                    self.assertAllInRange(
                        tf.divide(actual, expected), 0.9, 1.1)


# import matplotlib.pyplot as plt
# x = tf.ones(shape=(2, 28, 28, 1))
# y = crop.Crop()(x, keep_proportion=0.5)
# plt.imshow(y[0,:,:,0], cmap="gray")

# x = tf.ones(shape=(16, 28, 28, 3))
# y = crop.Crop()(x, keep_proportion=0.5)
# plt.imshow(y[0, :, :, :])
# plt.imshow(y[1, :, :, :])
# plt.imshow(y[2, :, :, :])
