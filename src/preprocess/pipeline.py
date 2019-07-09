from typing import Tuple
from typing import Callable

import matplotlib.pyplot as plt


class Pipeline:
    def __init__(self, stages: Tuple[Callable, dict], debug: bool = False):
        self._stages = stages
        self._debug = debug

    def __call__(self, image):
        image_stages = [("input", image)]
        for f, kwargs in self._stages:
            image = f(image, **kwargs)
            if self._debug:
                image_stages.append((f.__name__, image))

        if self._debug:
            most_bands = max([stage[1].shape[2] for stage in image_stages])

            fig, axes = plt.subplots(len(image_stages), most_bands + 1)
            fig.set_figheight(15)
            fig.set_figwidth(30)

            for ax, (stage, image) in zip(axes, image_stages):
                ax[0].imshow(image[:, :, :3])

                for channel_idx in range(0, image.shape[2]):
                    ax[channel_idx + 1].imshow(image[:, :, channel_idx])

                ax[0].set_title(stage)

        return image
