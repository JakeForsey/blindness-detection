import itertools

from src.constants import TEST_IMAGE_WIDTH
from src.constants import TEST_IMAGE_HEIGHT
from src.optimization.base import ExperimentGenerator
from src.optimization.experiment import Experiment

EXPERIMENTS = [
    Experiment(
        description="Large images",
        pipeline_stages=[
            (
                "crop_radius",
                {
                    "width_proportion": 0.8,
                    "height_proportion": 0.8
                }
            ),
            (
                "resize",
                {
                    "width": 224,
                    "height": 224,
                },
            ),
            #(
            #    "crop_dark_borders",
            #    {
            #        "tol": 10,
            #    },
            #),
            #(
            #    "resize_and_pad",
            #    {
            #        "width": 256,
            #        "height": 256,
            #        # Matching bias in bens
            #        "border_colour": 0
            #    },
            #),
            (
                "bens",
                {
                    "image_weight": 4,
                    "blur_window": (0, 0),
                    "blur_sigma_x": 22,
                    "blur_weight": -4,
                    "bias": 128,
                },
            ),
            (
                "fill_dark_borders",
                {},
            ),
            (
                "eight_bit_normalization",
                {}
            )
        ],
        # Introduce slight variations into training cycle in an attempt to increase
        # data set size
        augmentation_stages=[
            # ("none", {})
            ("rotate", {"limit": (-10, 10), "p": 0.2}),
            ("horizontal_flip", {}),
            ("grid_distort", {"p": 0.1, "distort_limit": 0.1}),
            ("brightness_contrast", {"p": 0.3, "contrast_limit": 0.3, "brightness_limit": 0.3}),
            ("crop", {"p": 0.4, "min_max_height": (200, 224), "height": 224, "width": 224, "w2h_ratio": 1.0})
        ],
        train_test_data_frames=["data/aptos2019-blindness-detection/train.csv"],
        train_test_directories=["data/aptos2019-blindness-detection/train_images"],
        model=("resnet18", {"num_classes": 5, "pretrained": True}),
        batch_size=64,
        optimizer=("Adam", {"lr": 1e-5 , "weight_decay": 0.00001}),
        test_size=0.4,
        max_epochs=120,
        sampler=("RandomSampler", {}),
        lr_scheduler=("ExponentialLR", {"gamma": 0.99})
    ),
]


class HandTunedExperiments(ExperimentGenerator):

    def __init__(self, experiment_ranges, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if experiment_ranges is None:
            experiment_ranges = [range(len(EXPERIMENTS))]

        self._experiments = [
            EXPERIMENTS[i]
            for i in range(len(EXPERIMENTS))
            if i in set(itertools.chain(*experiment_ranges))
        ]
        self._index = 0

    def __next__(self):
        self._index += 1
        if self._index <= len(self._experiments):
            return self._experiments[self._index - 1]
        else:
            raise StopIteration

    def __iter__(self):
        self._index = 0
        return self
