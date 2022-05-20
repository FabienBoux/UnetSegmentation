import os

import keras.models
import tensorflow as tf
from tensorflow.keras import layers


# NEURAL NETWORK
def double_conv_block(x, n_filters):
    #  Use in both the encoder (or the contracting path) and the bottleneck of the U-Net.
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)  # TODO: 3D
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)  # TODO: 3D

    return x


def downsample_block(x, n_filters):
    # Function for downsampling or feature extraction to be used in the encoder.
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)  # TODO: 3D
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # Upsampling function upsample_block for the decoder (or expanding path) of the U-Net.
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)  # upsample #TODO: 3D
    x = layers.concatenate([x, conv_features])  # concatenate
    x = layers.Dropout(0.3)(x)  # dropout
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model(shape=(128, 128, 1)):
    # Inputs
    inputs = layers.Input(shape=shape)

    # Encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # Decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # Outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)

    # Unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# SAVE / LOAD MODELS
def save_unet_model(unet_model, path):
    unet_model.save(os.path.join(path, 'unet_model'))


def load_unet_model(path):
    return keras.models.load_model(os.path.join(path, 'unet_model'))
