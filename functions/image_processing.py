import numpy as np
import tensorflow as tf


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
