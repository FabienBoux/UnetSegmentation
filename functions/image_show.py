import numpy as np
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap


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
