import time
from typing import Tuple
from typing import List
from typing import Callable
from typing import Optional
from uuid import uuid4

import pandas as pd
from torch import optim

from src.models.mnist import MnistExampleV01


OPTIMIZERS = {
    "SGD": optim.SGD,
}

MODELS = {
    "MnistExampleV01": MnistExampleV01
}


class Experiment:

    def __init__(
            self,
            pipeline_stages: List[Tuple[Callable, dict]],
            train_test_directory: str,
            train_test_data_frame: str,
            model: Tuple[str, dict],
            batch_size: int,
            optimzier: Tuple[str, dict],
            test_size: float,
            description: Optional[str] = None,

    ):
        self._id = uuid4()
        self._pipeline_stages = pipeline_stages
        self._train_test_directory = train_test_directory
        self._train_test_data_frame = train_test_data_frame
        self._model_string, self._model_kwargs = model
        self._batch_size = batch_size
        self._optimzier_string, self._optimizer_kwargs = optimzier
        self._test_size = test_size

        if description is None:
            description = time.time()
        self._description = description

    def id(self):
        return self._pipeline_stages

    def pipeline_stages(self):
        return self._pipeline_stages

    def train_test_directory(self) -> str:
        return self._train_test_directory

    def train_test_data_frame(self):
        return pd.read_csv(self._train_test_data_frame)

    def model(self):
        return MODELS[self._model_string](**self._model_kwargs)

    def batch_size(self):
        return self._batch_size

    def optimizer(self):
        return OPTIMIZERS[self._optimzier_string], self._optimizer_kwargs

    def test_size(self):
        return self._test_size

    def description(self):
        return self._description

    def from_json(self, json):
        raise NotImplemented()

    def to_serializable(self):
        raise NotImplemented()
