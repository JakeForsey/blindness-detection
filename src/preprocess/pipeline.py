from typing import Tuple
from typing import Callable
from typing import Union
from typing import List

import cv2
import numpy as np

from src.optimization.experiment import PIPELINE_STAGES

import matplotlib.pyplot as plt


class Pipeline:
    def __init__(
            self,
            stages: Union[List[Tuple[Callable, dict]], List[Tuple[str, dict]]],
            debug: bool = False
    ):

        # If the stages arg has not been initialised
        if isinstance(stages[0][0], str):
            stages = self.initialise_stages(stages)

        self._stages = stages
        self._debug = debug

    def __call__(self, image):
        border_mask = self._create_border_mask(image)
        image_stages = [("input", image), ("mask", border_mask)]

        for f, kwargs in self._stages:
            image, border_mask = f(image, border_mask=border_mask, **kwargs)
            assert image.shape == border_mask.shape, f"{f.__name__} did not maintain the shape of the mask"

            if self._debug:
                image_stages.append((f.__name__, image))

        if self._debug:
            most_bands = max([stage[1].shape[2] for stage in image_stages])

            fig, axes = plt.subplots(len(image_stages), most_bands + 1)
            fig.set_figheight(30)
            fig.set_figwidth(30)

            for ax, (stage, image) in zip(axes, image_stages):
                ax[0].imshow(image[:, :, :3])
                ax[0].axis('off')

                for channel_idx in range(0, image.shape[2]):
                    ax[channel_idx + 1].imshow(image[:, :, channel_idx])
                    ax[channel_idx + 1].axis('off')

                ax[0].set_title(stage)

        return image

    def _create_border_mask(self, image, resize_factor: int = 3):
        """
        Inspired(!) by:
        https://medium.com/@sonu008/image-enhancement-contrast-stretching-using-opencv-python-6ad61f6f171c

        :param image:
        :param resize_factor:
        :return:
        """
        width, height = round(image.shape[1] / resize_factor), round(image.shape[0] / resize_factor)

        resized_image = cv2.resize(image, (width, height))

        grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Log transform
        grey = 0.2 * (np.log(1 + np.float32(grey)))
        grey = cv2.convertScaleAbs(grey)

        _, mask = cv2.threshold(
            cv2.GaussianBlur(grey, (21, 21), 10),
            0, 255,
            cv2.THRESH_OTSU
        )

        # Back up to full size
        resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

        # Convert to 3 dimensions
        border_mask = np.stack((resized_mask, ) * 3, axis=-1)

        return border_mask

    @staticmethod
    def initialise_stages(stages: List[Tuple[str, dict]]) -> List[Tuple[Callable, dict]]:
        return [(PIPELINE_STAGES[stage], kwargs) for stage, kwargs in stages]
