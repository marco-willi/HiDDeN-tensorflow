import tensorflow as tf
import numpy as np

from noise import dct


def create_single_batch(img_arr):
    x = tf.cast(img_arr, tf.float32)
    x = tf.expand_dims(x, 0)
    x_batch = tf.concat([x, x], 0)
    return x, x_batch


def center_img(x):
    return tf.cast(x - 128, tf.float32)


def de_center_img(x):
    return tf.cast(x + 128, tf.float32)


class DCTTests(tf.test.TestCase):

    def setUp(self):
        """ Create Test Data """

        # example from: https://en.wikipedia.org/wiki/JPEG
        test_wiki_gray = np.array([
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94]])
        
        self.test_wiki_expected = np.array([
            [-415.38, -30.19, -61.20, 27.24, 56.12, -20.1, -2.39, 0.46],
            [4.47, -21.86, -60.76, 10.25, 13.15, -7.09, -8.54, 4.88],
            [-46.83, 7.37, 77.13, -24.56, -28.91, 9.93, 5.42, -5.65],
            [-48.53, 12.07, 34.10, -14.76, -10.24, 6.3, 1.83, 1.95],
            [12.12, -6.55, -13.2, -3.95, -1.87, 1.75, -2.79, 3.14],
            [-7.73, 2.91, 2.38, -5.94, -2.38, 0.94, 4.3, 1.85],
            [-1.03, 0.18, 0.42, -2.42, -0.88, -3.02, 4.12, -0.66],
            [-0.17, 0.14, -1.07, -4.19, -1.17, -0.1, 0.5, 1.68]
        ])

        test_wiki_gray = tf.expand_dims(test_wiki_gray, -1)

        self.test_wiki_gray, self.test_wiki_gray_batch = \
            create_single_batch(test_wiki_gray)
        
        test_ones = tf.ones(shape=(32, 32, 3)) * 255.

        self.test_ones, self.test_ones_batch = \
            create_single_batch(test_ones)
    
    def testWikiExpected(self):

        test_img = self.test_wiki_gray
        expected, _ = create_single_batch(self.test_wiki_expected)

        with self.cached_session(use_gpu=True):
            dct_transform = dct.DCT2D(n=8)
            x = center_img(test_img)
            x = dct_transform(x)
            x = tf.reshape(x, expected.shape)
            self.assertAllClose(x, expected, atol=1e-2)

    def testEncodingDecodingGraySingle(self):

        test_img = self.test_wiki_gray

        with self.cached_session(use_gpu=True):

            dct_transform = dct.DCT2D(n=8)
            x = center_img(test_img)
            channels = x.shape[-1]
            x = dct_transform(x)
            x = tf.stack(tf.split(x, channels, axis=-1), -1)

            y = tf.concat(tf.split(x, channels, axis=-1), -2)
            y = tf.squeeze(y, -1)
            dct_inverse = dct.InverseDCT2D(n=8)
            y = dct_inverse(y)
            y = de_center_img(y)

            self.assertAllLessEqual(tf.subtract(y, test_img), 1e-3)

    def testEncodingDecodingGrayBatch(self):

        test_img = self.test_wiki_gray_batch

        with self.cached_session(use_gpu=True):

            dct_transform = dct.DCT2D(n=8)
            x = center_img(test_img)
            channels = x.shape[-1]
            x = dct_transform(x)
            x = tf.stack(tf.split(x, channels, axis=-1), -1)

            y = tf.concat(tf.split(x, channels, axis=-1), -2)
            y = tf.squeeze(y, -1)
            dct_inverse = dct.InverseDCT2D(n=8)
            y = dct_inverse(y)
            y = de_center_img(y)

            self.assertAllLessEqual(tf.subtract(y, test_img), 1e-3)

    def testEncodingDecodingOnes(self):

        test_img = self.test_ones

        with self.cached_session(use_gpu=True):

            dct_transform = dct.DCT2D(n=8)
            x = center_img(test_img)
            channels = x.shape[-1]
            x = dct_transform(x)
            x = tf.stack(tf.split(x, channels, axis=-1), -1)

            y = tf.concat(tf.split(x, channels, axis=-1), -2)
            y = tf.squeeze(y, -1)
            dct_inverse = dct.InverseDCT2D(n=8)
            y = dct_inverse(y)
            y = de_center_img(y)

            self.assertAllLessEqual(tf.subtract(y, test_img), 1e-3)

    def testEncodingDecodingOnesBatch(self):

        test_img = self.test_ones_batch

        with self.cached_session(use_gpu=True):

            dct_transform = dct.DCT2D(n=8)
            x = center_img(test_img)
            channels = x.shape[-1]
            x = dct_transform(x)
            x = tf.stack(tf.split(x, channels, axis=-1), -1)

            y = tf.concat(tf.split(x, channels, axis=-1), -2)
            y = tf.squeeze(y, -1)
            dct_inverse = dct.InverseDCT2D(n=8)
            y = dct_inverse(y)
            y = de_center_img(y)

            self.assertAllLessEqual(tf.subtract(y, test_img), 1e-3)


if __name__ == '__main__':
    tf.test.main(argv=None)
