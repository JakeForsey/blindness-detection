import cv2
import numpy as np


def eight_bit_normalization(image: np.array):
    return image / 255


def resize(image: np.array, width: int, height: int):
    return cv2.resize(image, (width, height))


def crop_dark_borders(image: np.array, tol: int):
    mask = image > tol
    mask = mask.any(2)
    return image[np.ix_(mask.any(1), mask.any(0))]

