import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from functions.database import create_dataset, save_mask, score_classification, load_images, load_image
from functions.unet_architecture import build_unet_model
from functions.plot import plot_model_history

# Uncomment the following line to perform CPU execution instead of GPU execution
# tf.config.set_visible_devices([], 'GPU')
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# MAIN
if __name__ == '__main__':
    path = "C:\\Users\\Fabien Boux\\Desktop\\Dataset"
    resolution = (128, 128)

    # Define working folders
    datadir = os.path.join(path, "data_bl")
    datadir2 = os.path.join(path, "data_w6")

    outdir = os.path.join("outputs\\fit_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir)
    logdir = os.path.join(outdir, 'logs')
    os.makedirs(logdir)

    # Define neural network training hyperparameters
    batch_size = 20
    buffer_size = 10
    train_test_ratio = .8
    validation_test_ratio = .5
    num_epochs = 10  # TODO: see EarlyStopping to increase the number of epochs
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

    # Plot training history
    fig, axs = plt.subplots(1, 2)
    plot_model_history(axs, model_history)
    plt.savefig(os.path.join(logdir, 'Training.png'))
    plt.close()

    # Test
    correct_classif = []
    uncorrect_classif = []
    for dir in os.listdir(datadir2):
        try:
            # Load
            x, y = load_image(os.path.join(datadir2, dir), resolution=resolution, augmentation=False)
            x = np.expand_dims(np.moveaxis(x, -1, 0), axis=-1)
            y = np.expand_dims(np.moveaxis(y, -1, 0), axis=-1)
            y = np.moveaxis(y[:, :, :, 0], 0, -1)

            # Predict
            y_pred = unet_model.predict(x, batch_size=8)
            y_pred = np.moveaxis(y_pred[:, :, :, 0], 0, -1)
            if y_pred.sum() > (1 - y_pred).sum():
                y_pred = 1 - y_pred

            score = save_mask(y_pred, y, display=outdir, log_fname=dir)
            correct_classif.append(score[0])
            uncorrect_classif.append(score[1])

        except:
            pass

    fig, ax = plt.subplots(1, 1)
    ax.hist(correct_classif, bins=40, range=[0, 100], label='Correct')
    ax.hist(uncorrect_classif, bins=40, range=[0, 100], label='Uncorrect')
    ax.set_xlabel('% of total metastases volume')
    ax.set_xlim([0, 100])
    plt.savefig(os.path.join(logdir, 'Classification_accuracy.png'))
    plt.close()
