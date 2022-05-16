from random import random

import cv2 as cv
import numpy as np
import tensorflow as tf
from deepbrain import Extractor
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, erosion, dilation


# PRE-PROCESS IMAGES (IMAGE & MASK)


def binarize(input_mask):
    mask_bin = input_mask.copy()
    for s in range(input_mask.shape[-1]):
        _, mask_bin[:, :, s] = cv.threshold(input_mask[:, :, s], 0.5, 1, cv.THRESH_BINARY)

    return mask_bin


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
def extract_brain_mask(input_image, threshold=0):
    # `prob` will be a 3d numpy image containing probability of being brain tissue for each of the voxels in `img`
    # mask can be obtained as: mask = prob > 0.5
    ext = Extractor()
    prob = ext.run(input_image)

    return prob > threshold


# DATA AUGMENTATION
def augment(input_image, input_mask):
    # Flip
    new_image = cv.flip(input_image, 1)
    new_mask = cv.flip(input_mask, 1)

    # Gaussian filtering
    new_image = gaussian_filter(new_image, sigma=random())

    return new_image, new_mask


# POST-PROCESS IMAGES (MASK)
def dilate_erode(input_mask, disk_radius=3):
    new_mask = input_mask.copy()
    footprint = disk(disk_radius)
    for s in range(input_mask.shape[-1]):
        new_mask[:, :, s] = erosion(dilation(input_mask[:, :, s], footprint), footprint)

    return new_mask


def split_voi(input_mask):
    input_mask = input_mask.astype(np.uint8)

    # Get all contours
    count = 0
    new_mask = []
    for s in range(input_mask.shape[-1]):

        mask_slice = np.zeros(input_mask[:, :, s].shape, dtype=input_mask.dtype)
        if input_mask[:, :, s].sum() > 0:
            contours, hierarchy = cv.findContours(input_mask[:, :, s], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            for contour in contours:
                mask = np.zeros(input_mask[:, :, s].shape, dtype=input_mask.dtype)
                cv.fillPoly(mask, pts=[contour], color=1)

                if s > 0:
                    intersection = np.unique(cv.bitwise_and(new_mask[s - 1], new_mask[s - 1], mask=mask))[-1]
                else:
                    intersection = 0
                if (count > 0) & (intersection > 0):
                    cv.fillPoly(mask_slice, pts=[contour], color=int(intersection))
                else:
                    count += 1
                    cv.fillPoly(mask_slice, pts=[contour], color=count)

            new_mask.append(mask_slice)
        else:
            new_mask.append(mask_slice)

    new_mask = np.moveaxis(np.array(new_mask), 0, 2)

    # TODO: it seems that the total number of VOI is greater than the initial. Need verification
    # fig, axs = plt.subplots(2, 3)
    #
    # s = 68
    # mask_show(axs[0, 0], input_mask[:, :, s])
    # mask_show(axs[1, 0], new_mask[:, :, s])
    #
    # plt.savefig('test_split.png')
    # plt.close()

    return new_mask
