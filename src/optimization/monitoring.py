import os

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


from src.optimization.experiment import Experiment


class APTOSTensorboard:

    def __init__(self, experiment: Experiment, cv_iteration: int, base_directory="results/logdir"):
        if not os.path.isdir(base_directory):
            os.mkdir(base_directory)

        self._experiment = experiment
        self._summary_writer = SummaryWriter(
            os.path.join(base_directory, f"{experiment.id()} - {cv_iteration}")
        )

    def process(self, epoch, predictions_proba: torch.Tensor, predictions,  targets, ids):

        self._summary_writer.add_scalar(
            tag="accuracy_score",
            scalar_value=accuracy_score(targets, predictions),
            global_step=epoch
        )

        self._summary_writer.add_scalar(
            tag="cohen_kappa_score",
            scalar_value=cohen_kappa_score(targets, predictions),
            global_step=epoch
        )

        self._summary_writer.add_scalar(
            tag="f1_score_micro",
            scalar_value=f1_score(targets, predictions, average="micro"),
            global_step=epoch
        )

        self._summary_writer.add_scalar(
            tag="f1_score_macro",
            scalar_value=f1_score(targets, predictions, average="macro"),
            global_step=epoch
        )

        self._summary_writer.add_text(
            tag="description",
            text_string=self._experiment.description()
        )
