from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as PltFigureCanvas
import skimage

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(targets, predictions, normalize: bool = False, title: Optional[str] = None, as_array: bool = False) -> np.ndarray:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

    cm = confusion_matrix(targets, predictions)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    if as_array:
        canvas = PltFigureCanvas(fig)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if as_array:
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        array = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        reshaped_image = array.reshape(int(height), int(width), 3)

        reshaped_float_image = skimage.img_as_float32(reshaped_image)

        return reshaped_float_image

    else:
        return ax
