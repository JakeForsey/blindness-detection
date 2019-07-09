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
            fig, axes = plt.subplots(1, len(image_stages))
            fig.set_figheight(10)
            fig.set_figwidth(30)

            for ax, (stage, image_stage) in zip(axes, image_stages):
                ax.imshow(image_stage)
                ax.set_title(stage)

        return image
