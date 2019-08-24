import json
import time
from typing import Tuple
from typing import List
from typing import Optional
from uuid import uuid4

import albumentations
import pandas as pd
from torch import optim
from torch.utils.data.sampler import RandomSampler


from src.models.mnist import MnistExampleV01
from src.models.resnet import resnet18
from src.models.efficient_net import efficientnet_b0, efficientnet_b1
from src.models.resnet import resnext101_32x8d
from src.preprocess.normalize import crop_dark_borders
from src.preprocess.normalize import crop_radius
from src.preprocess.normalize import fill_dark_borders
from src.preprocess.normalize import eight_bit_normalization
from src.preprocess.normalize import resize
from src.preprocess.normalize import resize_and_pad
from src.preprocess.normalize import normalize_left_right
from src.preprocess.features import bens
from src.preprocess.features import enhance_fovea
from src.data.sampling import ImbalancedAPTOSDatasetSampler

SAMPLERS = {
    "ImbalancedAPTOSDatasetSampler": ImbalancedAPTOSDatasetSampler,
    "RandomSampler": RandomSampler,
}

OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

LR_SCHEDULERS = {
    "ExponentialLR": optim.lr_scheduler.ExponentialLR
}

MODELS = {
    "MnistExampleV01": MnistExampleV01,
    "resnet18": resnet18,
    "resnext101_32x8d": resnext101_32x8d,
    "efficientnet-b0":efficientnet_b0,
    "efficientnet-b1": efficientnet_b1

}

PIPELINE_STAGES = {
    "crop_dark_borders": crop_dark_borders,
    "crop_radius": crop_radius,
    "fill_dark_borders": fill_dark_borders,
    "resize": resize,
    "resize_and_pad": resize_and_pad,
    "enhance_fovea": enhance_fovea,
    "bens": bens,
    "eight_bit_normalization": eight_bit_normalization,
    "normalize_left_right": normalize_left_right
}

AUGMENTATION_STAGES = {
    # Not all albumentations augmentation transforms work for some reason so only add
    # after checking they work
    "none": albumentations.NoOp,
    "rotate": albumentations.Rotate,
    "grid_distort": albumentations.GridDistortion,
    "brightness_contrast": albumentations.RandomBrightnessContrast,
    "crop": albumentations.RandomSizedCrop,
    "horizontal_flip": albumentations.HorizontalFlip
}


class Experiment:

    def __init__(
            self,
            pipeline_stages: List[Tuple[str, dict]],
            augmentation_stages: List[Tuple[str, dict]],
            train_test_directories: List[str],
            train_test_data_frames: List[str],
            model: Tuple[str, dict],
            batch_size: int,
            optimizer: Tuple[str, dict],
            test_size: float,
            max_epochs: int,
            sampler: Tuple[str, dict],
            lr_scheduler: Tuple[str, dict],
            description: Optional[str] = None,

    ):
        self._id = str(uuid4())
        self._pipeline_stages = pipeline_stages
        # TODO look at albumentations.core.serialization.to_dict and albumentations.core.serialization.from_dict
        self._augmentation_stages = augmentation_stages
        self._train_test_directories = train_test_directories
        self._train_test_data_frames = train_test_data_frames
        self._model_string, self._model_kwargs = model
        self._batch_size = batch_size
        self._optimizer_string, self._optimizer_kwargs = optimizer
        self._max_epochs = max_epochs
        self._test_size = test_size
        self._sampler_string, self._sampler_kwargs = sampler
        self._lr_scheduler_string, self._lr_scheduler_kwargs = lr_scheduler

        if description is None:
            description = time.time()
        self._description = description

    def __str__(self):
        return str(self.state_dict())

    def id(self):
        return self._id

    def pipeline_stages(self):
        # Can't use Pipleine.initialise_stages() as it causes a circular import...
        return [(PIPELINE_STAGES[stage], kwargs) for stage, kwargs in self._pipeline_stages]

    def augmentation_stages(self):
        return [(AUGMENTATION_STAGES[aug_stage], kwargs) for aug_stage, kwargs in self._augmentation_stages]

    def train_test_directories(self) -> List[str]:
        return self._train_test_directories

    def train_test_data_frames(self):
        return [pd.read_csv(file_path) for file_path in self._train_test_data_frames]

    def model(self, input_shape):
        return MODELS[self._model_string](shape=input_shape, **self._model_kwargs)

    def batch_size(self):
        return self._batch_size

    def optimizer(self):
        return OPTIMIZERS[self._optimizer_string], self._optimizer_kwargs

    def lr_scheduler(self):
        return LR_SCHEDULERS[self._lr_scheduler_string], self._lr_scheduler_kwargs

    def test_size(self):
        return self._test_size

    def max_epochs(self):
        return self._max_epochs

    def description(self):
        return self._description

    def sampler(self):
        return SAMPLERS[self._sampler_string], self._sampler_kwargs

    @staticmethod
    def from_dict(state_dict):
        return Experiment(
            pipeline_stages=state_dict["pipeline_stages"],
            train_test_directories=state_dict["train_test_directories"],
            train_test_data_frames=state_dict["train_test_data_frames"],
            model=(state_dict["model"], state_dict["model_kwargs"]),
            batch_size=state_dict["batch_size"],
            optimizer=(state_dict["optimizer"], state_dict["optimizer_kwargs"]),
            test_size=state_dict["test_size"],
            max_epochs=state_dict["pipeline_stages"],
            sampler=(state_dict["sampler"], state_dict["sampler_kwargs"]),
            description=state_dict["description"],
            augmentation_stages=state_dict["augmentation_stages"],
            lr_scheduler=(state_dict["lr_scheduler"], state_dict["lr_scheduler_kwargs"])
        )

    def to_json(self):
        return json.dumps(self.state_dict())

    def state_dict(self):
        return {
            "id": self._id,
            "description": self._description,
            "pipeline_stages": self._pipeline_stages,
            "train_test_directories": str(self._train_test_directories),
            "train_test_data_frames": str(self._train_test_data_frames),
            "model": self._model_string,
            "model_kwargs": self._model_kwargs,
            "batch_size": self._batch_size,
            "optimizer": self._optimizer_string,
            "optimizer_kwargs": self._optimizer_kwargs,
            "max_epochs": self._max_epochs,
            "test_size": self._test_size,
            "sampler": self._sampler_string,
            "sampler_kwargs": self._sampler_kwargs,
            "augmentation_stages": self._augmentation_stages,
            "lr_scheduler": self._lr_scheduler_string,
            "lr_scheduler_kwargs": self._lr_scheduler_kwargs
        }