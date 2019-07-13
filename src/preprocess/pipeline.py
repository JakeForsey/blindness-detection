from typing import Tuple
from typing import Callable
from typing import Union
from typing import List

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
        image_stages = [("input", image)]
        for f, kwargs in self._stages:
            image = f(image, **kwargs)
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

    @staticmethod
    def initialise_stages(stages: List[Tuple[str, dict]]) -> List[Tuple[Callable, dict]]:
        return [(PIPELINE_STAGES[stage], kwargs) for stage, kwargs in stages]
