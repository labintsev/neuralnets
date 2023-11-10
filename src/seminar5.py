import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pytest

from test_utils import get_preprocessed_data


def build_conv_layer() -> tf.keras.layers.Conv2D:
    """
    Build Conv2D layer with some filters, kernel size and striding step
    Find necessary params, that match conditions
    Input image shape -> Output shape
    (10, 10, 3) -> (2, 2, 2)
    (20, 20, 3) -> (4, 4, 2)
    :param
    :return: keras convolutional layer
    """
    my_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=5)
    return my_layer


def build_padded_conv_layer(kernel_size) -> tf.keras.layers.Conv2D:
    """
    Build Conv2D layer with some filters and paddings
    Find necessary params, that match conditions
    Input image shape -> Output shape
    (10, 10, 3) -> (10, 10, 2)
    (20, 20, 3) -> (20, 20, 2)
    :param: kernel_size may vary
    :return: keras convolutional layer
    """
    my_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=kernel_size, padding='same')
    return my_layer


def build_depth_wise_conv_layer() -> tf.keras.layers.DepthwiseConv2D:
    """Build DepthWise Convolution layer """
    my_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=2)
    return my_layer


def build_pooling_layer() -> tf.keras.layers.MaxPooling2D:
    """Build MaxPooling layer with fixed pool and strides"""
    my_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    return my_layer



def build_up_conv_layer() -> tf.keras.layers.Conv2DTranspose:
    """Build Transpose Convolution layer"""
    my_layer = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2)
    return my_layer


def build_conv_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=(32, 32, 3)),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model


def train(net='../models/my_conv_net'):
    model = build_conv_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    model.fit(x_train, y_train, epochs=16, batch_size=128)
    model.evaluate(x_test, y_test)
    model.save(net)


def draw_weights(net_path):
    model = tf.keras.models.load_model(net_path)
    w = model.layers[0].kernel.numpy()
    w_min, w_max = np.min(w), np.max(w)
    for i in range(64):
        plt.subplot(8, 8, i + 1)

        # Rescale the weights to be between 0 and 255
        w_img = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(w_img.astype('uint8'))
        plt.axis('off')
    plt.savefig('../output/seminar5/weights.png')


if __name__ == '__main__':
    NET_PATH = '../models/my_conv_net'
    train(NET_PATH)
    draw_weights(NET_PATH)