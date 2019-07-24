#!/usr/bin/env python
"""
Searches models and hyper parameters.
"""
import numpy as np
import logging
import os
import sqlite3
from typing import List

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import random_split as torch_random_split
from torchsummary import summary as torch_summary
from torch.utils.data import Subset


from src.argument_parser import parse_training_arguments
from src.preprocess.pipeline import Pipeline
from src.preprocess.augmentation import AugmentedCollate
from src.data.dataset import APTOSDataset
from src.optimization.hand_tuned import HandTunedExperiments
from src.optimization.experiment import Experiment
from src.optimization.result import Result
from src.optimization.monitoring import APTOSMonitor
from src.ml import train
from src.ml import test
from sklearn.model_selection import StratifiedShuffleSplit
from src.optimization.loss import Kgbloss

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

    #preprocessing
    pipeline = Pipeline(experiment.pipeline_stages(), debug=debug_pipeline)
    #augmentations
    augmentations = AugmentedCollate(experiment.augmentation_stages())

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    # File system cache with a 1 to 1 mapping to an experiment, used to cache data for multiple workers,
    # can safely be used in each cross validation run
    cache = joblib.Memory(f'./cachedir/{experiment.id()}', verbose=0)
    LOGGER.info("Initialised cache: %s", cache)

    LOGGER.info("Creating APTOSDataset for the following directories: %s", directories)


    dataset = TorchConcatDataset(
        [APTOSDataset(df, directory, pipeline, cache) for df, directory in zip(dfs, directories)]
    )

    # To facilitate software development this makes running end to end tests feasible
    if develop_mode:
        LOGGER.warn("Running in develop mode, using a fraction of the whole dataset")
        dataset, _ = torch_random_split(dataset, [develop_mode_sampls, len(dataset) - develop_mode_sampls])

    results = []

    # Stratified ShuffleSplit cross-validator, Provides train/test indices to split data in train/test sets.
    sss = StratifiedShuffleSplit(n_splits=cross_validation_iterations,
                                 test_size=experiment.test_size(),
                                 train_size=1-experiment.test_size(),
                                 random_state=0)
    #TODO: will probably need debugging when more than one datasets are added
    labels = np.asarray([x for x in dfs[0]["diagnosis"]])
    split_generator=sss.split(np.zeros(labels.shape), labels)

    for cv_iteration, (train_index, test_index) in zip(range(1,  cross_validation_iterations + 1), split_generator):
        LOGGER.info("Cross validation iteration: %s", cv_iteration)

        with APTOSMonitor(experiment, cv_iteration) as monitor:
            LOGGER.info(f'tensorboard --logdir "{monitor._summary_writer.log_dir}"')

            if experiment.dataset_split()=="stratified":
                test_ds = Subset(dataset, test_index)
                train_ds = Subset(dataset, train_index)
            else:
                test_size = experiment.test_size()
                train_ds, test_ds = torch_random_split(
                    dataset,
                    [round((1 - test_size) * len(dataset)), round(test_size * len(dataset))]
                )

            print("train data size: {}". format(train_ds.__len__()))
            print(np.histogram(labels[train_index], 5))

            print("test data size: {}". format(test_ds.__len__()))
            print(np.histogram(labels[test_index], 5))

            sampler, sampler_kwargs = experiment.sampler()
            sampler = sampler(train_ds, **sampler_kwargs)

            train_loader = TorchDataLoader(
                train_ds,
                batch_size=experiment.batch_size(),
                num_workers=data_loader_workers,
                # Potentially an unconventional use of collate_fn, but it does make the
                # train data loader responsible for augmentations which is nice.
                collate_fn=augmentations,
                sampler=sampler
            )

            test_loader = TorchDataLoader(
                test_ds,
                batch_size=experiment.batch_size(),
                num_workers=data_loader_workers,
            )

            model = experiment.model(input_shape=train_ds[0][0].shape)
            print(torch_summary(model.cuda(), train_ds[0][0].shape))

            optimizer_class, optim_kwargs = experiment.optimizer()
            optimizer = optimizer_class(model.parameters(), **optim_kwargs)

            lr_scheduler, scheduler_kwargs = experiment.lr_scheduler()
            lr_scheduler = lr_scheduler(optimizer, **scheduler_kwargs)

            monitor.on_cv_start(train_ds, augmentations)

            for epoch in range(1, experiment.max_epochs() + 1):

                LOGGER.info("Epoch: %s", epoch)

                train(model, train_loader, optimizer, device, monitor)
                lr_scheduler.step()

                predictions_proba, predictions,  targets, ids, losses = test(model, test_loader, device, monitor)

                if epoch % 2 == 0:
                    checkpoint = {
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'experiment': experiment.state_dict()
                    }

                    checkpoint_directory = f'results/{experiment.id()}'
                    if not os.path.isdir(checkpoint_directory):
                        os.mkdir(checkpoint_directory)

                    torch.save(checkpoint, os.path.join(checkpoint_directory, f'{cv_iteration}-{epoch}-checkpoint.pth'))

            monitor.on_cv_end()

        predictions = predictions.tolist()
        targets = targets.tolist()

        results_df = pd.DataFrame({
            "experiment_id": [experiment.id() for _ in range(len(targets))],
            "cross_validation_iteration": [cv_iteration for _ in range(len(targets))],
            "targets": targets,
            "predictions": predictions,
            "id_code": ids
        })

        results.append(Result(experiment, results_df))

    # Deletes content on disk... (until experiments have a unique hash this make sense)
    cache.clear()

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

    main(
        args.develop_mode,
        args.data_loader_workers,
        args.cross_validation_iterations,
        args.results_directory,
        args.device,
        args.experiments,
    )
