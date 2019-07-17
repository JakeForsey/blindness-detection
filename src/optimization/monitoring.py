import os
from typing import List

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset as TorchDataset

from src.optimization.experiment import Experiment
from src.visualisations import plot_confusion_matrix
from src.preprocess.augmentation import AugmentedCollate


class APTOSMonitor:

    def __init__(self, experiment: Experiment, cv_iteration: int, base_directory="results/logdir"):
        if not os.path.isdir(base_directory):
            os.mkdir(base_directory)

        self._epoch = 0
        self._experiment = experiment
        self._summary_writer = SummaryWriter(
            os.path.join(base_directory, f"{experiment.id()} - {cv_iteration}")
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._summary_writer.close()

    def on_train_batch_end(self, batch_idx, data, train_loader, loss):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            self._epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    def on_train_end(self, losses: torch.Tensor):
        self._summary_writer.add_scalar(
            tag="train/loss",
            scalar_value=losses.mean(),
            global_step=self._epoch
        )

    def on_test_batch_end(self):
        pass

    def on_test_end(
            self,
            predictions_proba: torch.Tensor,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            ids: List[str],
            losses: torch.Tensor
    ):
        self._epoch += 1

        predictions_proba = predictions_proba.cpu()
        predictions = predictions.cpu()
        targets = targets.cpu()
        losses = losses.cpu()

        self._summary_writer.add_scalar(
            tag="test/loss",
            scalar_value=losses.mean(),
            global_step=self._epoch
        )

        self._summary_writer.add_scalar(
            tag="test/accuracy_score",
            scalar_value=accuracy_score(targets, predictions),
            global_step=self._epoch
        )

        self._summary_writer.add_scalar(
            tag="test/cohen_kappa_score",
            scalar_value=cohen_kappa_score(
                targets,
                predictions,
                weights='quadratic',
            ),
            global_step=self._epoch
        )

        self._summary_writer.add_scalar(
            tag="test/f1_score_micro",
            scalar_value=f1_score(targets, predictions, average="micro"),
            global_step=self._epoch
        )

        self._summary_writer.add_scalar(
            tag="test/f1_score_macro",
            scalar_value=f1_score(targets, predictions, average="macro"),
            global_step=self._epoch
        )

        confusion_matrix_array = plot_confusion_matrix(targets, predictions, as_array=True)
        self._summary_writer.add_image(
            tag="test/confusion_matrix",
            img_tensor=torch.from_numpy(confusion_matrix_array),
            global_step=self._epoch,
            dataformats="HWC"
        )

        confusion_matrix_array = plot_confusion_matrix(targets, predictions, as_array=True, normalize=True)
        self._summary_writer.add_image(
            tag="test/confusion_matrix_normalized",
            img_tensor=torch.from_numpy(confusion_matrix_array),
            global_step=self._epoch,
            dataformats="HWC"
        )

    def on_cv_start(self, dataset: TorchDataset, augmentations: AugmentedCollate):
        self._epoch = 1

        self._summary_writer.add_text(
            tag="description",
            text_string=self._experiment.description()
        )

        self._summary_writer.add_text(
            tag="pipeline_stages",
            text_string=" -> ".join([f"{f}({kwargs})" for f, kwargs in self._experiment._pipeline_stages])
        )

        self._summary_writer.add_text(
            tag="model",
            text_string=f"{self._experiment._model_string}({self._experiment._model_kwargs})"
        )

        self._summary_writer.add_text(
            tag="datasets",
            text_string=f"{self._experiment.train_test_directories()}"
        )

        normalized_images = [torch.from_numpy(image) for image, _, _ in [dataset[i] for i in range(10)]]
        self._summary_writer.add_image(
            tag=f"normalized_images",
            img_tensor=torch.cat(normalized_images, 2),
            dataformats="CHW"
        )

        # Augmentations are applied to lists of numpy arrays and are returned as batches of tensors
        augmented_images = [
            image.squeeze() for image, _, _ in [augmentations([dataset[i]]) for i in range(10)]
        ]
        self._summary_writer.add_image(
            tag=f"augmented_images",
            img_tensor=torch.cat(augmented_images, 2),
            dataformats="CHW"
        )

    def on_cv_end(self):
        pass
