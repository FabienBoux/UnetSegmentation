import glob
import os
import sys
from datetime import datetime
from random import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from pydicom import dcmread


# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# tf.config.set_visible_devices([], 'GPU')


# LOAD IMAGES / CREATE DATASET
def load_images(path=None, resolution=(128, 128)):
    data_images = np.zeros((0, resolution[0], resolution[1], 1))
    data_masks = np.zeros((0, resolution[0], resolution[1], 1))

    for dir in os.listdir(path):
        try:
            # Load inputs
            image = []
            location = []

            p = os.path.join(path, dir, "Image", os.listdir(os.path.join(path, dir, "Image"))[0])
            files = glob.glob(p + "/*.dcm")
            for file in files:
                dataset = dcmread(file)
                image.append(dataset.pixel_array)
                location.append(dataset.SliceLocation)
            image = np.array(image)
            image = image[np.argsort(location), :, :]
            image = image[::-1, :, :]

            flag_first = 0
            files = glob.glob(os.path.join(path, dir, "Labels/*"))
            for file in files:
                dataset = dcmread(glob.glob(file + '/*')[0])
                if flag_first == 1:
                    mask = mask + dataset.pixel_array
                else:
                    mask = dataset.pixel_array
                    flag_first = 1

            # Format inputs
            image = normalize(image)
            image, mask = resize(np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1), resolution=resolution)

            # Concatenate
            data_images = np.concatenate((data_images, np.expand_dims(np.moveaxis(image, 2, -3), axis=-1)))
            data_masks = np.concatenate((data_masks, np.expand_dims(np.moveaxis(mask, 2, -3), axis=-1)))

        except:
            print(dir)

    return data_images.astype(np.float32), data_masks.astype(int)


def create_dataset(path, p=.8, resolution=(128, 128)):
    data_images, data_masks = load_images(path, resolution=resolution)
    data_images, data_masks = shuffle(data_images, data_masks, random_state=0)

    train_data = tf.data.Dataset.from_tensor_slices(
        (data_images[:int(len(data_images) * p), :, :, :],
         data_masks[:int(len(data_images) * p), :, :, :]))

    test_data = tf.data.Dataset.from_tensor_slices(
        (data_images[int(len(data_images) * p):, :, :, :],
         data_masks[int(len(data_images) * p):, :, :, :]))

    return train_data, test_data


# PRE-PROCESS IMAGES (IMAGE & MASK)
def resize(input_image, input_mask, resolution=(128, 128)):
    input_image = tf.image.resize(input_image, resolution, method="nearest")
    input_mask = tf.image.resize(input_mask, resolution, method="nearest")

    return input_image, input_mask


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image):
    # TODO: exclude non brain voxels
    # Standardize histogram
    input_image = (input_image - np.mean(input_image)) / np.std(input_image)

    return input_image


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


# MAIN
if __name__ == '__main__':
    path = "C:\\Users\\Fabien Boux\\Desktop\\Dataset"

    resolution = (128, 128)

    datadir = os.path.join(path, "data")
    logdir = os.path.join(path, "logs\\fit_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    batch_size = 10
    buffer_size = 10
    train_test_ratio = .8
    validation_test_ratio = .5

    train_dataset, test_dataset = create_dataset(datadir, p=train_test_ratio, resolution=resolution)

    train_batches = train_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_size = int(len(test_dataset) * validation_test_ratio)
    validation_batches = test_dataset.take(validation_size).batch(batch_size)
    test_batches = test_dataset.skip(validation_size).take(len(test_dataset)).batch(batch_size)

    # Define the Keras TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # Define and train model
    unet_model = build_unet_model(shape=(resolution[0], resolution[1], 1))

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss="sparse_categorical_crossentropy",
                       metrics="accuracy")

    num_epochs = 10
    train_length = len(train_dataset)
    steps_per_epoch = train_length // batch_size
    val_subsplits = 5
    test_length = len(test_dataset)
    validation_steps = test_length // batch_size // val_subsplits

    model_history = unet_model.fit(train_batches,
                                   epochs=num_epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   validation_steps=validation_steps,
                                   validation_data=test_batches,
                                   callbacks=[tensorboard_callback])

    # x, y = load_images(path=datadir, resolution=(256, 256))
    # y_pred = unet_model.predict(x, batch_size=8)

    x = np.array([i.numpy() for i, j in test_dataset])
    y = np.array([j.numpy() for i, j in test_dataset])
    y_pred = unet_model.predict(x, batch_size=8)

    # TODO: this process could be improve
    np.place(y_pred, y_pred < .5, -1)
    np.place(y_pred, y_pred >= .5, 0)
    np.place(y_pred, y_pred < -.5, 1)

    nb = 5
    img = [50, 100, 600, 405, 270]
    img = [i for i in range(len(y)) if y[i].sum() > 10][:nb]
    fig, axs = plt.subplots(4, len(img))

    for i in range(len(img)):
        axs[0, i].imshow(x[img[i], :, :, 0], cmap=plt.cm.bone)
        axs[1, i].imshow(y[img[i], :, :, 0], cmap=plt.cm.bone)

        axs[2, i].imshow(y_pred[img[i], :, :, 0], cmap=plt.cm.bone)
        axs[3, i].imshow(y[img[i], :, :, 0] - y_pred[img[i], :, :, 0], cmap='jet', vmin=-1, vmax=1)

        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
        axs[2, i].set_axis_off()
        axs[3, i].set_axis_off()

    plt.savefig('test.png')
    plt.close()

    score = y[:, :, :, 0] - y_pred[:, :, :, 0]
    total = (y[:, :, :, 0] == 1).sum()
    correct = ((score == 0) & (y[:, :, :, 0] == 1)).sum()
    uncorrect = (score == -1).sum()
    missed = (score == 1).sum()

    print((correct, uncorrect, missed) / (0.01 * total))
