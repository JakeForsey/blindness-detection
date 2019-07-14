from abc import ABC, abstractmethod

import bisect
import torch
import torch.utils.data
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Subset as TorchSubset

from src.data.dataset import APTOSDataset


class ImbalancedDatasetSampler(TorchSampler, ABC):
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


class ImbalancedAPTOSDatasetSampler(ImbalancedDatasetSampler):
    """
    Implementation of ImabalancedDatasetSampler that can be used on a torch.utils.data.Subset
    of a torch.utils.data.ConcatDataset of one or more APTOSDataset.

    WARNING: Dog shit code ahead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label(self, dataset, idx):
        # Well, here we are... Custom implementation to fetch a diagnosis from a
        # torch.utils.data.Subset of a torch.utils.data.ConcatDataset of one or more APTOSDataset
        # without reading the image.
        if isinstance(dataset, TorchSubset):
            idx = dataset.indices[idx]
            if isinstance(dataset.dataset, TorchConcatDataset):

                def concat_dataset_index(concat_dataset: TorchConcatDataset, idx: int):
                    dataset_idx = bisect.bisect_right(concat_dataset.cumulative_sizes, idx)
                    if dataset_idx == 0:
                        sample_idx = idx
                    else:
                        sample_idx = idx - concat_dataset.cumulative_sizes[dataset_idx - 1]
                    return sample_idx, dataset_idx

                sample_idx, dataset_idx = concat_dataset_index(dataset.dataset, idx)

                if isinstance(dataset.dataset.datasets[0], APTOSDataset):
                    return dataset.dataset.datasets[dataset_idx].__getitem__(sample_idx, diagnosis_only=True)

        else:
            raise NotImplementedError("Only implemented for a Subset of Concatenated APTOSDatasets")
