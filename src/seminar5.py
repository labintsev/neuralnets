"""Seminar 5. Convolutional Networks"""
import tensorflow as tf

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
    # TODO Create layer with necessary filters, kernel size and striding step
    my_layer = None

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

    # TODO Create layer with necessary filters and padding. Kernel size is builder parameter.
    my_layer = None

    return my_layer


def build_depth_wise_conv_layer() -> tf.keras.layers.DepthwiseConv2D:
    """Build DepthWise Convolution layer """

    # TODO Create layer with necessary kernel size and depth multiplier
    my_layer = None
    return my_layer


def build_pooling_layer() -> tf.keras.layers.MaxPooling2D:
    """Build MaxPooling layer with fixed pool and strides"""

    # TODO Create layer with necessary kernel size and strides
    my_layer = None
    return my_layer


def build_up_conv_layer() -> tf.keras.layers.Conv2DTranspose:
    """Build Transpose Convolution layer"""

    # TODO Create layer with necessary filters, kernel size and strides
    my_layer = None
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
        tf.keras.layers.Conv2D(filters=16, kernel_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model


if __name__ == '__main__':
    model = build_conv_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
