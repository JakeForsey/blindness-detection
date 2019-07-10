#!/usr/bin/env python
"""
Searches models and hyper parameters.
"""
import logging
import os
import sqlite3
from typing import Tuple
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import random_split

from src.preprocess.pipeline import Pipeline
from src.data.dataset import APTOSDataset
from src.optimization.hand_tuned import HandTunedExperiments
from src.optimization.experiment import Experiment
from src.optimization.result import Result


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DATA_LOADER_WORKERS = 6

DEVELOP_MODE = False
DEVELOP_MODE_SAMPLES = 10

if DEVELOP_MODE:
    LOGGER.warn("Running in develop mode, only %s samples will be used.", DEVELOP_MODE_SAMPLES)

CROSS_VALIDATION_ITERATIONS = 3

RESULTS_DIRECTORY = "./results"
if not os.path.isdir(RESULTS_DIRECTORY):
    os.mkdir(RESULTS_DIRECTORY)

DEVICE = "cuda:0"
if DEVICE == "cuda:0":
    torch.cuda.set_device(torch.device("cuda:0"))


def train(log_interval, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    predictions = []
    predictions_proba = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            preds_proba = model(data)
            predictions_proba.extend(preds_proba)

            preds = preds_proba.argmax(dim=1)
            predictions.extend(preds)

            targets.extend(target)

    return torch.stack(predictions_proba), torch.stack(predictions),  torch.stack(targets)


def run_experiment(experiment: Experiment, debug_pipeline: bool = False) -> List[Result]:

    pipeline = Pipeline(experiment.pipeline_stages(), debug=debug_pipeline)

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    dataset = TorchConcatDataset(
        [APTOSDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )
    if DEVELOP_MODE:
        dataset, _ = random_split(dataset, [DEVELOP_MODE_SAMPLES, len(dataset) - DEVELOP_MODE_SAMPLES])

    results = []
    for cv_iteration in range(1,  CROSS_VALIDATION_ITERATIONS + 1):
        LOGGER.info("Cross validation iteration: %s", cv_iteration)

        test_size = experiment.test_size()
        train_ds, test_ds = random_split(
            dataset,
            [round((1 - test_size) * len(dataset)), round(test_size * len(dataset))]
        )

        train_loader = TorchDataLoader(
            train_ds,
            batch_size=experiment.batch_size(),
            num_workers=DATA_LOADER_WORKERS,
        )

        test_loader = TorchDataLoader(
            test_ds,
            batch_size=experiment.batch_size(),
            num_workers=DATA_LOADER_WORKERS,
        )

        model = experiment.model(input_shape=train_ds[0][0].shape)

        optimizer_class, optim_kwargs = experiment.optimizer()
        optimizer = optimizer_class(model.parameters(), **optim_kwargs)

        metric_df = pd.DataFrame(columns=["experiment_id", "epoch", "test_loss", "test_accuracy"])
        for epoch in range(1, experiment.max_epochs() + 1):
            LOGGER.info("Epoch: %s", epoch)

            train(1, model, train_loader, optimizer, epoch)
            predictions_proba, predictions,  targets = test(model, test_loader)

        predictions = predictions.tolist()
        targets = targets.tolist()

        results_df = pd.DataFrame({
            "experiment_id": [experiment.id() for _ in range(len(targets))],
            "cross_validation_iteration": [cv_iteration for _ in range(len(targets))],
            "targets": targets,
            "predictions": predictions,
        })

        results.append(Result(experiment, metric_df, results_df))

    return results


def main():
    experiment_generator = HandTunedExperiments()

    for experiment in experiment_generator:
        # Every time an experiment is completed, persist the cross validation results
        results = run_experiment(experiment)
        for result in results:

            if not DEVELOP_MODE:
                with sqlite3.connect(os.path.join(RESULTS_DIRECTORY, "db.db")) as conn:
                    result.persist(RESULTS_DIRECTORY, conn)


if __name__ == "__main__":
    main()
