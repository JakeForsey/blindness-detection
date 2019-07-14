import itertools

from src.constants import TEST_IMAGE_WIDTH
from src.constants import TEST_IMAGE_HEIGHT
from src.optimization.base import ExperimentGenerator
from src.optimization.experiment import Experiment


EXPERIMENTS = [
    Experiment(
        description="Resnet, Bens normalization, over sampling, 256x256, 2015 & 2019 training dataset",
        # Standardize differences and hand engineer features to reduce the required complexity
        # to model the problem
        pipeline_stages=[
            (
                "crop_dark_borders",
                {
                    "tol": 10,
                },
            ),
            (
                "resize",
                {
                    "width": TEST_IMAGE_WIDTH,
                    "height": TEST_IMAGE_HEIGHT,
                },
            ),
            (
                "bens",
                {
                    "image_weight": 4,
                    "blur_window": (0, 0),
                    "blur_sigma_x": 20,
                    "blur_weight": -4,
                    "bias": 128,
                },
            ),
            (
                "resize",
                {
                    "width": 256,
                    "height": 256,
                },
            ),
            (
                "eight_bit_normalization",
                {}
            )
        ],
        train_test_data_frames=[
            "data/aptos2019-blindness-detection/train.csv",
            # To use this dataset, the column names need to be converted to "id_code" and "diagnosis"
            "/media/jake/ssd/aptos2015-blindness-detection/trainLabels.csv"
        ],
        train_test_directories=[
            "data/aptos2019-blindness-detection/train_images",
            # To use this dataset, the column names need to be converted to "id_code" and "diagnosis"
            "/media/jake/ssd/aptos2015-blindness-detection/train"
        ],
        model=("resnet18", {"num_classes": 5, "pretrained": False}),
        batch_size=100,
        optimizer=("Adam", {"lr": 1e-3}),
        test_size=0.2,
        max_epochs=100,
        sampler=("ImbalancedAPTOSDatasetSampler", {})
    ),
    Experiment(
        description="Resnet, Bens normalization, over sampling, 128x128, 2015 & 2019 training dataset",
        pipeline_stages=[
            (
                "crop_dark_borders",
                {
                    "tol": 10,
                },
            ),
            (
                "resize",
                {
                    "width": TEST_IMAGE_WIDTH,
                    "height": TEST_IMAGE_HEIGHT,
                },
            ),
            (
                "bens",
                {
                    "image_weight": 4,
                    "blur_window": (0, 0),
                    "blur_sigma_x": 20,
                    "blur_weight": -4,
                    "bias": 128,
                },
            ),
            (
                "eight_bit_normalization",
                {}
            )
        ],
        # Introduce slight variations into training cycle in an attempt to increase
        # data set size
        augmentation_stages=[
            ("rotate", {"limit": 10, "p": 0.2}),
            ("grid_distort", {"p": 0.2}),
            ("brightness_contrast", {"p": 0.2, "contrast_limit": 0.3, "brightness_limit": 0.3}),
        ],
        train_test_data_frames=["data/aptos2019-blindness-detection/train.csv"],
        train_test_directories=["data/aptos2019-blindness-detection/train_images"],
        model=("resnet18", {"num_classes": 5, "pretrained": False}),
        batch_size=100,
        optimizer=("SGD", {"lr": 0.001, "momentum": 0.9}),
        test_size=0.2,
        max_epochs=35,
        sampler=("ImbalancedAPTOSDatasetSampler", {})
    ),
    Experiment(
        description="Pretrained Resnet18 with 2015 and 2019 training datasets with Bens normalization and over sampling",
        pipeline_stages=[
            (
                "crop_dark_borders",
                {
                    "tol": 10,
                },
            ),
            (
                "resize",
                {
                    "width": 128,
                    "height": 128,
                },
            ),
            (
                "eight_bit_normalization",
                {}
            )
        ],
        train_test_data_frames=[
            "data/aptos2019-blindness-detection/train.csv",
            # To use this dataset, the column names need to be converted to "id_code" and "diagnosis"
            "/media/jake/ssd/aptos2015-blindness-detection/trainLabels.csv"
        ],
        train_test_directories=[
            "data/aptos2019-blindness-detection/train_images",
            # To use this dataset, the column names need to be converted to "id_code" and "diagnosis"
            "/media/jake/ssd/aptos2015-blindness-detection/train"
        ],
        model=("resnet18", {"num_classes": 5, "pretrained": False}),
        batch_size=100,
        optimizer=("Adam", {"lr": 1e-3}),
        test_size=0.2,
        max_epochs=100,
        sampler=("ImbalancedAPTOSDatasetSampler", {})
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
