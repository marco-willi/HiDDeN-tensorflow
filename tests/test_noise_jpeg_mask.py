import tensorflow as tf
import numpy as np

from noise import jpeg_mask


class JPEGMaskTests(tf.test.TestCase):

    def testShapeNotDivisibleBy8(self):

        shapes = [
            (1, 28, 28, 1),
            (1, 28, 32, 1),
            (1, 32, 28, 1),
            (1, 500, 500, 3)]

        with self.cached_session(use_gpu=True):

            for shape in shapes:
                jpeg = jpeg_mask.JPEG_Mask(n=8)
                test_img = tf.ones(shape=shape)
                output = jpeg(test_img)

                self.assertAllEqual(shape, output.shape)

if __name__ == '__main__':
    tf.test.main(argv=None)
