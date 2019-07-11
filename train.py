#!/usr/bin/env python
"""
Searches models and hyper parameters.
"""
import logging
import os
import sqlite3
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import random_split

from src.argument_parser import parse_training_arguments
from src.preprocess.pipeline import Pipeline
from src.data.dataset import APTOSDataset
from src.optimization.hand_tuned import HandTunedExperiments
from src.optimization.experiment import Experiment
from src.optimization.result import Result
from src.optimization.monitoring import APTOSMonitor
from src.ml import train
from src.ml import test

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def run_experiment(
        experiment: Experiment,
        debug_pipeline: bool = False,
        develop_mode: bool = False,
        data_loader_workers: int = 1,
        cross_validation_iterations: int = 3,
        device: str = "cpu",
        develop_mode_sampls: int = 10
) -> List[Result]:
    LOGGER.info("Beginning experiment: %s, %s", experiment.id(), experiment.description())

    pipeline = Pipeline(experiment.pipeline_stages(), debug=debug_pipeline)

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    dataset = TorchConcatDataset(
        [APTOSDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )

    if develop_mode:
        dataset, _ = random_split(dataset, [develop_mode_sampls, len(dataset) - develop_mode_sampls])

    results = []
    for cv_iteration in range(1,  cross_validation_iterations + 1):
        LOGGER.info("Cross validation iteration: %s", cv_iteration)

        with APTOSMonitor(experiment, cv_iteration) as monitor:
            LOGGER.info(f'tensorboard --logdir "{monitor._summary_writer.log_dir}"')

            monitor.process_cv_start()

            test_size = experiment.test_size()
            train_ds, test_ds = random_split(
                dataset,
                [round((1 - test_size) * len(dataset)), round(test_size * len(dataset))]
            )

            train_loader = TorchDataLoader(
                train_ds,
                batch_size=experiment.batch_size(),
                num_workers=data_loader_workers,
            )

            test_loader = TorchDataLoader(
                test_ds,
                batch_size=experiment.batch_size(),
                num_workers=data_loader_workers,
            )

            model = experiment.model(input_shape=train_ds[0][0].shape)

            optimizer_class, optim_kwargs = experiment.optimizer()
            optimizer = optimizer_class(model.parameters(), **optim_kwargs)

            metric_df = pd.DataFrame(columns=["experiment_id", "epoch", "test_loss", "test_accuracy"])
            for epoch in range(1, experiment.max_epochs() + 1):
                LOGGER.info("Epoch: %s", epoch)

                train(1, model, train_loader, optimizer, epoch)
                predictions_proba, predictions,  targets, ids = test(model, test_loader)

                monitor.process_epoch(epoch, predictions_proba, predictions, targets, ids)

            monitor.process_cv_end(predictions_proba, predictions,  targets, ids)

        predictions = predictions.tolist()
        targets = targets.tolist()

        results_df = pd.DataFrame({
            "experiment_id": [experiment.id() for _ in range(len(targets))],
            "cross_validation_iteration": [cv_iteration for _ in range(len(targets))],
            "targets": targets,
            "predictions": predictions,
            "id_code": ids
        })

        results.append(Result(experiment, metric_df, results_df))

    return results


def main(
        develop_mode,
        data_loader_workers,
        cross_validation_iterations,
        results_directory,
        device,
        experiments,
):

    # TODO Make this experiment generator dynamic e.g. select ExperimentGenerator
    #  type based on cmd line arguments
    experiment_generator = HandTunedExperiments(experiments)

    for experiment in experiment_generator:
        # Every time an experiment is completed, persist the cross validation results
        results = run_experiment(
            experiment=experiment,
            develop_mode=develop_mode,
            data_loader_workers=data_loader_workers,
            cross_validation_iterations=cross_validation_iterations,
            device=device
        )
        for result in results:

            with sqlite3.connect(os.path.join(results_directory, "db.db")) as conn:
                result.persist(results_directory, conn)


if __name__ == "__main__":
    args = parse_training_arguments()

    if args.develop_mode:
        LOGGER.warn("Running in develop mode, only %s samples will be used.", args.develop_mode)

    if not os.path.isdir(args.results_directory):
        os.mkdir(args.results_directory)

    if args.device == "cuda:0":
        torch.cuda.set_device(torch.device("cuda:0"))

    main(
        args.develop_mode,
        args.data_loader_workers,
        args.cross_validation_iterations,
        args.results_directory,
        args.device,
        args.experiments,
    )
