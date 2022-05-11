import os
import glob

import numpy as np
import tensorflow as tf

from pydicom import dcmread
from sklearn.utils import shuffle

from functions.image_processing import resize, normalize, augment


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
