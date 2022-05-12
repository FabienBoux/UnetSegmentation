import numpy as np
import tensorflow as tf
import cv2

from scipy.ndimage import gaussian_filter
from random import random

from deepbrain import Extractor


# PRE-PROCESS IMAGES (IMAGE & MASK)
def resize(input_image, input_mask, resolution=(128, 128)):
    tf.compat.v1.enable_eager_execution()

    input_image = tf.image.resize(input_image, resolution, method="bilinear").numpy()
    input_mask = tf.image.resize(input_mask, resolution, method="nearest").numpy()

    return input_image, input_mask


def normalize(input_image, mask=None):
    # Standardize histogram
    if mask is None:
        input_image = (input_image - np.mean(input_image)) / np.std(input_image)
    else:
        input_image = (input_image - np.mean(input_image[mask])) / np.std(input_image[mask])
        input_image[~mask] = 0

    return input_image


# BRAIN EXTRACTION
def extract_brain_mask(input_image, threshold=.25):
    # `prob` will be a 3d numpy image containing probability of being brain tissue for each of the voxels in `img`
    # mask can be obtained as: mask = prob > 0.5
    ext = Extractor()
    prob = ext.run(input_image)

    return prob >= threshold


# DATA AUGMENTATION
def augment(input_image, input_mask):
    # Flip
    new_image = cv2.flip(input_image, 1)
    new_mask = cv2.flip(input_mask, 1)

    # Gaussian filtering
    # new_image = gaussian_filter(new_image, sigma=random())

    return np.concatenate((input_image, new_image)), np.concatenate((input_mask, new_mask))

# POST-PROCESS IMAGES (MASK)
# TODO: morphological processing
