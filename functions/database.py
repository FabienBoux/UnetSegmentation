import os
import glob
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from pydicom import dcmread
from sklearn.utils import shuffle
from time import time

from functions.image_processing import resize, normalize, augment, extract_brain_mask


# LOAD IMAGES / CREATE DATASET
def load_images(path=None, resolution=(128, 128)):
    data_images = np.zeros((0, resolution[0], resolution[1], 1))
    data_masks = np.zeros((0, resolution[0], resolution[1], 1))

    for dir in os.listdir(path):
        try:
            t0 = time()

            # Load inputs
            image = []
            location = []

            p = os.path.join(path, dir, "Image", os.listdir(os.path.join(path, dir, "Image"))[0])
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

            brain_mask = extract_brain_mask(image)
            image[~brain_mask] = 0

            flag = 0
            files = glob.glob(os.path.join(path, dir, "Labels/*"))
            for file in files:
                dataset = dcmread(glob.glob(file + '/*')[0])
                if flag == 0:
                    mask = dataset.pixel_array
                    flag = 1
                else:
                    mask = mask + dataset.pixel_array

            # Reduce image to non-empty slices
            idx = [brain_mask[slice, :, :].sum() > 1 for slice in range(brain_mask.shape[0])]
            image = image[idx, :, :]
            mask = mask[idx, :, :]
            brain_mask = brain_mask[idx, :, :]

            # Format inputs
            image = normalize(image, mask=brain_mask)
            image, mask = resize(np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1), resolution=resolution)

            # Data augmentation
            # image, mask = augment(np.moveaxis(image, 2, -3), np.moveaxis(mask, 2, -3))

            # Concatenate
            data_images = np.concatenate((data_images, np.expand_dims(np.moveaxis(image, 2, -3), axis=-1)))
            data_masks = np.concatenate((data_masks, np.expand_dims(np.moveaxis(mask, 2, -3), axis=-1)))

            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
                  " - {} loaded in {}s".format(dir, round(time() - t0)))

        except:
            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
                  " - error loaded: {}".format(dir))

    return data_images.astype(np.float32), data_masks.astype(int)


def create_dataset(path, p=.8, resolution=(128, 128)):
    t0 = time()

    data_images, data_masks = load_images(path, resolution=resolution)
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
