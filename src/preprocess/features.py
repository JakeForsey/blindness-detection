import cv2
import numpy as np


def bens(image: np.array, image_weight, blur_window, blur_sigma_x, blur_weight, bias):
    """
    Preprocessing step recommended here:
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

    :param image: image to preprocess
    :param image_weight: relative weight of the un-blurred image
    :param blur_window: not sure
    :param blur_sigma_x: not sure
    :param blur_weight: relative weight of the blurred image
    :param bias: not sure
    :return:
    """

    return cv2.addWeighted(
        image, image_weight,
        cv2.GaussianBlur(image, blur_window, blur_sigma_x), blur_weight,
        bias
    )
