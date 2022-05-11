import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from functions.database import create_dataset
from functions.unet_architecture import build_unet_model

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.config.set_visible_devices([], 'GPU')

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
    num_epochs = 10
    val_subsplits = 5

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

    x = np.array([i.numpy() for i, j in test_dataset])
    y = np.array([j.numpy() for i, j in test_dataset])
    y_pred = unet_model.predict(x, batch_size=8)

    # TODO: this process could be improve
    np.place(y_pred, y_pred < .5, -1)
    np.place(y_pred, y_pred >= .5, 0)
    np.place(y_pred, y_pred < -.5, 1)

    nb = 5
    img = [i for i in range(len(y)) if y[i].sum() > 10][:nb]
    fig, axs = plt.subplots(4, len(img))

    for i in range(len(img)):
        axs[0, i].imshow(x[img[i], :, :, 0], cmap=plt.cm.bone, vmin=-3, vmax=3)
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
