import numpy as np
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap


# U-NET TRAINING
def plot_model_history(axs, model_history):
    for label in ['loss', 'val_loss']:
        axs[0].plot(model_history.history[label], label=label)
    for label in ['accuracy', 'val_accuracy']:
        axs[1].plot(model_history.history[label], label=label)

    axs[0].set_xlabel('Epoch number')
    axs[1].set_xlabel('Epoch number')
    axs[0].legend()
    axs[1].legend()


# IMAGES
def image_show(ax, input_image):
    ax.imshow(input_image, cmap=plt.cm.bone, interpolation='none')
    ax.axis('off')


def mask_show(ax, input_mask):
    if len(np.unique(input_mask)) < 3:
        cmap = ListedColormap([[0, 0, 0], [1, 1, 0]])
    else:
        cmap = ListedColormap(np.random.rand(256, 3))
        cmap.colors[0] = [0, 0, 0]

    ax.imshow(input_mask, cmap=cmap, interpolation='none')
    ax.axis('off')
