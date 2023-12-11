import unittest
import tensorflow as tf


class ConvolutionDemo(unittest.TestCase):
    def setUp(self) -> None:
        input_shape = (1, 32, 32, 3)

        self.x = tf.ones(input_shape)

    def test_conv_1d_1(self):
        conv_1d_layer = tf.keras.layers.Conv1D(filters=5, kernel_size=3)
        z = conv_1d_layer(self.x)
        self.assertEqual(z.shape, (1, 32, 30, 5))

    def test_conv_1d_2(self):
        conv_1d_layer = tf.keras.layers.Conv1D(filters=7, kernel_size=10)
        z = conv_1d_layer(self.x)
        self.assertEqual(z.shape, (1, 32, 23, 7))

    def test_conv_2d_1(self):
        conv_1d_layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3))
        z = conv_1d_layer(self.x)
        self.assertEqual(z.shape, (1, 30, 30, 5))

    def test_conv_2d_2(self):
        conv_1d_layer = tf.keras.layers.Conv2D(filters=7, kernel_size=(10, 10))
        z = conv_1d_layer(self.x)
        self.assertEqual(z.shape, (1, 23, 23, 7))

    def test_conv_3d_1(self):
        input_shape = (1, 32, 32, 32, 3)
        x = tf.ones(input_shape)
        conv_1d_layer = tf.keras.layers.Conv3D(filters=5, kernel_size=3)
        z = conv_1d_layer(x)
        self.assertEqual(z.shape, (1, 30, 30, 30, 5))

    def test_conv_3d_2(self):
        input_shape = (1, 32, 32, 32, 3)
        x = tf.ones(input_shape)
        conv_1d_layer = tf.keras.layers.Conv3D(filters=7, kernel_size=10)
        z = conv_1d_layer(x)
        self.assertEqual(z.shape, (1, 23, 23, 23, 7))

    def test_conv_2d_ones(self):
        conv_1d_layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), kernel_initializer='ones')
        z = conv_1d_layer(self.x)
        self.assertEqual(z.shape, (1, 30, 30, 5))
