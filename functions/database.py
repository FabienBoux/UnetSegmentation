import glob
import os
from datetime import datetime
from random import random
from time import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pydicom import dcmread, dcmwrite
from sklearn.utils import shuffle

from functions.image_processing import binarize, resize, normalize, augment, extract_brain_mask, split_voi, \
    morphological_procedure

from functions.plot import image_show, mask_show


# LOAD IMAGES / CREATE DATASET
def load_image(path=None, resolution=(128, 128), axs=None, augmentation=False):
    # Load inputs
    image = []
    location = []

    files = glob.glob(os.path.join(path, "Labels/*"))
    for file in files:
        dataset = dcmread(glob.glob(file + '/*')[0])
        if file == files[0]:
            mask = dataset.pixel_array
        else:
            mask = mask + dataset.pixel_array
    mask = binarize(mask)

    p = os.path.join(path, "Image", os.listdir(os.path.join(path, "Image"))[0])
    files = glob.glob(p + "/*.dcm")
    if len(files) == 1:
        dataset = dcmread(files[0])
        image = dataset.pixel_array
    else:
        for i in range(len(files)):
            dataset = dcmread(files[i])
            image.append(dataset.pixel_array)
            location.append(dataset.ImagePositionPatient[-1])
        image = np.array(image)
        image = image[np.argsort(location), :, :]

    if axs is not None:
        slice = np.array([mask[s, :, :].sum() for s in range(mask.shape[0])]).argmax()

        image_show(axs[0, 0], image[slice, :, :])
        mask_show(axs[1, 0], mask[slice, :, :])
        axs[0, 0].set_title('Original')

    # Bias field correction
    # If necessary, see the Nipype library:
    # https://nipype.readthedocs.io/en/0.12.1/interfaces/generated/nipype.interfaces.brainsuite.brainsuite.html

    # Resize and move axis
    image, mask = resize(np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1), resolution=resolution)

    # Extract brain mask
    brain_mask = extract_brain_mask(image)
    image[~brain_mask] = 0

    # Reduce image to non-empty slices
    idx = [brain_mask[:, :, slice].sum() > 1 for slice in range(brain_mask.shape[-1])]
    image = image[:, :, idx]
    mask = mask[:, :, idx]
    brain_mask = brain_mask[:, :, idx]

    if axs is not None:
        slice = slice - idx[:80].count(False)

        image_show(axs[0, 1], image[:, :, slice])
        mask_show(axs[1, 1], mask[:, :, slice])
        axs[0, 1].set_title('Brain\nextracted')

    # Format inputs
    image = normalize(image, mask=brain_mask)  # image = normalize(image, mask=None)

    if axs is not None:
        image_show(axs[0, 2], image[:, :, slice])
        mask_show(axs[1, 2], mask[:, :, slice])
        axs[0, 2].set_title('Resized/\nnormalized')

    # Data augmentation
    if augmentation:
        image_aug, mask_aug = augment(image, mask)
        image = np.concatenate((image, image_aug), axis=-1)
        mask = np.concatenate((mask, mask_aug), axis=-1)

        if axs is not None:
            image_show(axs[0, 3], image_aug[:, :, slice])
            mask_show(axs[1, 3], mask_aug[:, :, slice])
            axs[0, 3].set_title('Image\naugmented')

    return image, mask


def load_images(path=None, resolution=(128, 128), display=None, augmentation=False):
    data_images = np.zeros((0, resolution[0], resolution[1], 1))
    data_masks = np.zeros((0, resolution[0], resolution[1], 1))

    if display is not None:
        if not os.path.isdir(display):
            display = None

    for directory in os.listdir(path):
        try:
            t0 = time()

            if display is not None:
                fig, axs = plt.subplots(2, 4)

                image, mask = load_image(path=os.path.join(path, directory),
                                         resolution=resolution, axs=axs, augmentation=augmentation)
            else:
                image, mask = load_image(path=os.path.join(path, directory),
                                         resolution=resolution, axs=None, augmentation=augmentation)

            # Concatenate
            if mask.shape == image.shape:
                data_images = np.concatenate((data_images, np.expand_dims(np.moveaxis(image, 2, -3), axis=-1)))
                data_masks = np.concatenate((data_masks, np.expand_dims(np.moveaxis(mask, 2, -3), axis=-1)))

                print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
                      " - {} loaded in {}s".format(directory, round(time() - t0)))

                if display is not None:
                    plt.savefig(os.path.join(display, 'Data_preparation_%s.png' % directory))

            else:
                # TODO: see how to include the images that fall in this case
                print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - error loaded {}".format(directory))

            if display is not None:
                plt.close()

        except:
            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - error loaded {}".format(directory))

    return data_images.astype(np.float32), data_masks.astype(int)


def create_dataset(path, p=.8, resolution=(128, 128), display=None, augmentation=True):
    t0 = time()

    data_images, data_masks = load_images(path, resolution=resolution, display=display, augmentation=augmentation)
    data_images, data_masks = shuffle(data_images, data_masks, random_state=0)

    train_data = tf.data.Dataset.from_tensor_slices(
        (data_images[:int(len(data_images) * p), :, :, :],
         data_masks[:int(len(data_images) * p), :, :, :]))

    test_data = tf.data.Dataset.from_tensor_slices(
        (data_images[int(len(data_images) * p):, :, :, :],
         data_masks[int(len(data_images) * p):, :, :, :]))

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
          " - database created in {}min".format(dir, round((time() - t0)) / 60))

    return train_data, test_data


# SAVE MASKS / COMPUTE ACCURACY METRICS
def score_classification(true_mask, pred_mask):
    diff = true_mask - pred_mask
    score = 100 * np.array([((diff == 0) & (true_mask == 1)).sum(), (diff == -1).sum()]) / (
            true_mask == 1).sum()

    return score


def save_mask(input_mask, true_mask=None, display=None, log_fname=None):
    if log_fname is None:
        log_fname = round(random() * 1e6)

    # Binarize
    new_mask = binarize(input_mask)

    # Plot
    if display is not None:
        if not os.path.isdir(display):
            display = None
        else:
            if true_mask is not None:
                slice = np.array([true_mask[:, :, s].sum() for s in range(true_mask.shape[-1])]).argmax()
                score = score_classification(true_mask, new_mask)
            else:
                slice = 50  # TODO: some random ?

            fig, axs = plt.subplots(1, 3)
            mask_show(axs[0], new_mask[:, :, slice])
            axs[0].set_title('Original\nprediction\n{} - {}%'.format(round(score[0]), round(score[1])))

    # Morphological processes
    new_mask = morphological_procedure(new_mask, disk_radius=3)

    if display is not None:
        if true_mask is not None:
            score = true_mask - new_mask
            score = 100 * np.array([((score == 0) & (true_mask == 1)).sum(), (score == -1).sum()]) / (
                    true_mask == 1).sum()
        mask_show(axs[1], new_mask[:, :, slice])
        axs[1].set_title('Morpho.\nprocessing\n{} - {}%'.format(round(score[0]), round(score[1])))

    # Split
    new_mask = split_voi(new_mask)

    if display is not None:
        mask_show(axs[2], new_mask[:, :, slice])
        axs[2].set_title('Split\nlabels')

    if display is not None:
        plt.savefig(os.path.join(display, 'Mask_generation_%s.png' % str(log_fname)))
        plt.close()

    return score
