"""Seminar 5. Convolutional Networks"""
import tensorflow as tf
##################
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
    # TODO Create layer with necessary filters, kernel size and striding step:
    # output = (input - kernel)/strides +1

    my_layer = tf.keras.layers.Conv2D(kernel_size=(5, 5), strides=5, filters=2)
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

    # TODO Create layer with necessary filters and padding. Kernel size is builder parameter:
    # output = input

    padding = 'same'  # to maintain the original image size
    my_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=kernel_size, strides=1, padding=padding)
    return my_layer


def build_depth_wise_conv_layer() -> tf.keras.layers.DepthwiseConv2D:
    """Build DepthWise Convolution layer """

    # TODO Create layer with necessary kernel size and depth multiplier:
    # output = (input - kernel) / depth + 1

    my_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=2)
    return my_layer


def build_pooling_layer() -> tf.keras.layers.MaxPooling2D:
    """Build MaxPooling layer with fixed pool and strides"""

    # TODO Create layer with necessary kernel size and strides:
    # output = input/strides + 1

    my_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    return my_layer


def build_up_conv_layer() -> tf.keras.layers.Conv2DTranspose:
    """Build Transpose Convolution layer"""

    # TODO Create layer with necessary filters, kernel size and strides:
    # output = input*strides + kernel - strides

    my_layer = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=(2, 2))
    return my_layer


def build_pretrained_model():
    base_model = tf.keras.applications.MobileNet(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs, outputs)


def build_conv_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float16),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7),
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
        # w_img = np.moveaxis(w_img, -1, 0)
        plt.imshow(w_img.astype('uint8'))
        plt.axis('off')
    plt.savefig('../output/seminar5/weights.png')


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    NET_PATH = '../models/my_conv_net'
    train(NET_PATH)
    draw_weights(NET_PATH)