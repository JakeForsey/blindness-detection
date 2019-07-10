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


def enhance_fovea(image: np.array, radius, border_tol, blur_sigma, fovea_aoi_size, width: int, height: int):
    tmp_image = image.copy()

    tmp_image[(tmp_image < border_tol).all(2)] = 255
    # gray = cv2.GaussianBlur(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY), (radius, radius), blur_sigma)
    blur = cv2.GaussianBlur(tmp_image, (radius, radius), blur_sigma)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    (_, _, fovea_loc, _) = cv2.minMaxLoc(gray)

    cv2.circle(image, fovea_loc, radius, (255, 0, 0), 1)

    half_size = fovea_aoi_size // 2
    fovea_aoi = image[
                max(0, fovea_loc[1] - half_size): min(image.shape[0], fovea_loc[1] + half_size),
                max(0, fovea_loc[0] - half_size): min(image.shape[0], fovea_loc[0] + half_size),
                # 1st band most clearly highlights exudates
                0
                ]

    fovea_aoi = cv2.resize(fovea_aoi, (width, height))
    fovea_aoi = np.expand_dims(fovea_aoi, 2)

    image = np.concatenate([image, fovea_aoi], axis=2)

    return image
