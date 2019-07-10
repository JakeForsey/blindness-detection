import cv2
import numpy as np


def eight_bit_normalization(image: np.ndarray) -> np.ndarray:
    return image / 255


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height))


def crop_dark_borders(image: np.ndarray, tol: int) -> np.ndarray:
    mask = image > tol
    mask = mask.any(2)
    return image[np.ix_(mask.any(1), mask.any(0))]

def normalize_left_right(image: np.ndarray) -> np.ndarray:
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

    # Rescale to a size tested for the magic numbers below.
    image = cv2.resize(image,(512, 512))
    _, od = find_optic_disk(image, 99, 10, 25)
    if od[0] > image.shape[1] / 2:
        image = cv2.flip(image, 1)
    return image
