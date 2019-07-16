from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Subset as TorchSubset

from src.data.dataset import APTOSDataset


class BaseImbalancedDatasetSampler(TorchSampler, ABC):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Modified from:
    https://github.com/ufoym/imbalanced-dataset-sampler

    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset: TorchDataset, indices=None, num_samples=None):
        # No call to super()

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    @abstractmethod
    def _get_label(self, dataset, idx):
        pass

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedAPTOSDatasetSampler(BaseImbalancedDatasetSampler):
    """
    Implementation of ImabalancedDatasetSampler that can be used on a torch.utils.data.Subset
    of a torch.utils.data.ConcatDataset of one or more APTOSDataset.

    WARNING: Dog shit code ahead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_read_image(self, dataset: TorchDataset, value: bool):
        if isinstance(dataset, APTOSDataset):
            dataset.set_read_image(value)

        elif isinstance(dataset, TorchConcatDataset):
            for ds in dataset.datasets:
                self._set_read_image(ds, value)

        elif isinstance(dataset, TorchSubset):
            self._set_read_image(dataset.dataset, value)

    def _set_read_id(self, dataset: TorchDataset, value: bool):
        if isinstance(dataset, APTOSDataset):
            dataset.set_read_id(value)

        elif isinstance(dataset, TorchConcatDataset):
            for ds in dataset.datasets:
                self._set_read_id(ds, value)

        elif isinstance(dataset, TorchSubset):
            self._set_read_id(dataset.dataset, value)

    def _get_label(self, dataset: TorchDataset, idx: int):

        self._set_read_image(dataset, False)
        self._set_read_id(dataset, False)
        diagnosis = dataset[idx]
        self._set_read_image(dataset, True)
        self._set_read_id(dataset, True)

        return int(diagnosis)
