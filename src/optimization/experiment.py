import json
import time
from typing import Tuple
from typing import List
from typing import Optional
from uuid import uuid4

import pandas as pd
from torch import optim

from src.models.mnist import MnistExampleV01
from src.models.resnet import resnet18
from src.preprocess.normalize import crop_dark_borders
from src.preprocess.normalize import eight_bit_normalization
from src.preprocess.normalize import resize
from src.preprocess.normalize import normalize_left_right
from src.preprocess.features import bens
from src.preprocess.features import enhance_fovea


OPTIMIZERS = {
    "SGD": optim.SGD,
}

MODELS = {
    "MnistExampleV01": MnistExampleV01,
    "resnet18": resnet18,
}

STAGES = {
    "crop_dark_borders": crop_dark_borders,
    "resize": resize,
    "enhance_fovea": enhance_fovea,
    "bens": bens,
    "eight_bit_normalization": eight_bit_normalization,
    "normalize_left_right": normalize_left_right
}


class Experiment:

    def __init__(
            self,
            pipeline_stages: List[Tuple[str, dict]],
            train_test_directories: List[str],
            train_test_data_frames: List[str],
            model: Tuple[str, dict],
            batch_size: int,
            optimzier: Tuple[str, dict],
            test_size: float,
            max_epochs: int,
            description: Optional[str] = None,

    ):
        self._id = str(uuid4())
        self._pipeline_stages = pipeline_stages
        self._train_test_directories = train_test_directories
        self._train_test_data_frames = train_test_data_frames
        self._model_string, self._model_kwargs = model
        self._batch_size = batch_size
        self._optimzier_string, self._optimizer_kwargs = optimzier
        self._max_epochs = max_epochs
        self._test_size = test_size

        if description is None:
            description = time.time()
        self._description = description

    def id(self):
        return self._id

    def pipeline_stages(self):
        # Can't use Pipleine.initialise_stages() as it causes a circular import...
        return [(STAGES[stage], kwargs) for stage, kwargs in self._pipeline_stages]

    def train_test_directories(self) -> List[str]:
        return self._train_test_directories

    def train_test_data_frames(self):
        return [pd.read_csv(file_path) for file_path in self._train_test_data_frames]

    def model(self, input_shape):
        return MODELS[self._model_string](shape=input_shape, **self._model_kwargs)

    def batch_size(self):
        return self._batch_size

    def optimizer(self):
        return OPTIMIZERS[self._optimzier_string], self._optimizer_kwargs

    def test_size(self):
        return self._test_size

    def max_epochs(self):
        return self._max_epochs

    def description(self):
        return self._description

    def from_json(self, json_string):
        raise NotImplemented()

    def to_json(self):
        return json.dumps({
            "id": self._id,
            "description": self._description,
            "pipeline_stages": self._pipeline_stages,
            "train_test_directories": str(self._train_test_directories),
            "train_test_data_frames": str(self._train_test_data_frames),
            "model": self._model_string,
            "model_kwargs": self._model_kwargs,
            "batch_size": self._batch_size,
            "optimizer": self._optimzier_string,
            "optimizer_kwargs": self._optimizer_kwargs,
            "max_epochs": self._max_epochs,
            "test_size": self._test_size,
        })
