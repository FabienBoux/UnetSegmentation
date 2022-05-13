import os
from datetime import datetime

import warnings

# warnings.simplefilter('module')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use("Qt5Agg")

from functions.database import create_dataset, save_mask
from functions.unet_architecture import build_unet_model

# Uncomment the following line to perform CPU execution instead of GPU execution
# tf.config.set_visible_devices([], 'GPU')
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# MAIN
if __name__ == '__main__':
    path = "C:\\Users\\Fabien Boux\\Desktop\\Dataset"
    resolution = (128, 128)

    # Define working folders
    datadir = os.path.join(path, "data")
    datadir2 = os.path.join(path, "data2")
    datadir2 = None

    logdir = os.path.join("logs\\fit_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir)
    outdir = os.path.join("outputs\\fit_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir)

    # Define neural network training hyperparameters
    batch_size = 10
    buffer_size = 10
    train_test_ratio = .8
    validation_test_ratio = .5
    num_epochs = 10
    val_subsplits = 5

    # Create database and split
    train_dataset, test_dataset = create_dataset(datadir, p=train_test_ratio, resolution=resolution, display=outdir)

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

    train_length = len(train_dataset)
    steps_per_epoch = train_length // batch_size
    test_length = len(test_dataset)
    validation_steps = test_length // batch_size // val_subsplits

    model_history = unet_model.fit(train_batches,
                                   epochs=num_epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   validation_steps=validation_steps,
                                   validation_data=test_batches,
                                   callbacks=[tensorboard_callback])

    # Test data
    if datadir2 is None:
        x = np.array([i.numpy() for i, j in test_dataset])
        y = np.array([j.numpy() for i, j in test_dataset])
    else:
        _, test_dataset2 = create_dataset(datadir2, p=0, resolution=resolution)
        x = np.array([i.numpy() for i, j in test_dataset2])
        y = np.array([j.numpy() for i, j in test_dataset2])

    # Predict
    y_pred = unet_model.predict(x, batch_size=8)

    save_mask(y_pred, y)