from src.constants import TEST_IMAGE_WIDTH
from src.constants import TEST_IMAGE_HEIGHT
from src.optimization.base import ExperimentGenerator
from src.optimization.experiment import Experiment
from src.preprocess.features import bens
from src.preprocess.normalize import crop_dark_borders
from src.preprocess.normalize import resize
from src.preprocess.normalize import eight_bit_normalization


EXPERIMENTS = [
    Experiment(
        pipeline_stages=[
            (
                crop_dark_borders,
                {
                    "tol": 10
                }
            ),
            (
                resize,
                {
                    "width": TEST_IMAGE_WIDTH,
                    "height": TEST_IMAGE_HEIGHT
                }
            ),
            (
                bens,
                {
                    "image_weight": 4, "blur_window": (0, 0), "blur_sigma_x": 10, "blur_weight": -4, "bias": 128
                }
            ),
            (
                eight_bit_normalization,
                {}
            )
        ],
        description="Development",
        train_test_data_frame="../input/train.csv",
        train_test_directory="../input/train_images",
        model=("MnistExampleV01", {}),
        batch_size=100,
        optimzier=("SGD", {"lr": 0.0001, "momentum": 0.9}),
        test_size=0.2
    ),
]


class HandTunedExperiements(ExperimentGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experiments = EXPERIMENTS

    def __next__(self):
        return self._experiments[0]
