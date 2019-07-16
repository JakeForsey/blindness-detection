from typing import Tuple

import cv2
import numpy as np


def eight_bit_normalization(image: np.ndarray, border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return image / 255, border_mask


def resize(image: np.ndarray, width: int, height: int, border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    resized_image = cv2.resize(image, (width, height))
    resized_mask = cv2.resize(border_mask, (width, height))

    return resized_image, resized_mask


def resize_and_pad(
        image: np.ndarray,
        width: int, height: int,
        border_colour: int,
        border_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize an image respecting aspect ratio and inserting new image at the centre
    of add a border.

    Based on:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    :param border_mask: mask of the border areas
    :param image: image to resize
    :param width: width of the new image (must equal height)
    :param height: height of the new image (must equal width)
    :param border_colour: colour of the border
    :return: new image
    """
    assert width == height, "resize_and_pad assumes width and height are the same"

    def _resize_and_pad(array, new_shape, border_colour, interpolation=cv2.INTER_LINEAR):
        array = cv2.resize(array, (new_shape[1], new_shape[0]), interpolation=interpolation)

        # Calculate the size of the borders
        delta_w = width - new_shape[1]
        delta_h = width - new_shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        return cv2.copyMakeBorder(
            array,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(border_colour, border_colour, border_colour)
        )

    # old_shape is in (height, width) format
    old_shape = image.shape[:2]

    # Calculate the shape that the image should be after resizing
    ratio = float(width) / max(old_shape)
    new_shape = tuple([int(axis * ratio) for axis in old_shape])

    resized_image = _resize_and_pad(image, new_shape, border_colour)
    resized_mask = _resize_and_pad(border_mask, new_shape, interpolation=cv2.INTER_NEAREST, border_colour=False)

    return resized_image, resized_mask


def fill_dark_borders(image: np.ndarray, border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image[border_mask == 0] = 0
    return image, border_mask


def crop_dark_borders(image: np.ndarray, tol: int, border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = image > tol
    mask = mask.any(2)
    return image[np.ix_(mask.any(1), mask.any(0))], border_mask[np.ix_(mask.any(1), mask.any(0))]


def normalize_left_right(image: np.ndarray, border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def close_image(image: np.ndarray, radius: int):
        return cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius,radius),
            ),
        )

    def open_image(image: np.ndarray, radius: int):
        return cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius,radius),
            ),
        )

    def find_optic_disk(
        image: np.ndarray,
        radius: int,
        sigma: float,
        structuring_element_radius: int,
    ):
        image = open_image(image, structuring_element_radius)
        image = close_image(image, structuring_element_radius)
        red = image[:,:,0]
        red = cv2.GaussianBlur(red, (radius, radius), sigma)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(red)
        return red, maxLoc

    image_copy = image.copy()
    # Rescale to a size tested for the magic numbers below.
    image_copy = cv2.resize(image_copy,(512, 512))
    _, od = find_optic_disk(image_copy, 99, 10, 25)
    if od[0] > image_copy.shape[1] / 2:
        image = cv2.flip(image, 1)
        border_mask = cv2.flip(border_mask, 1)

    return image, border_mask
