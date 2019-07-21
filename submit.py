"""
Submits predictions to Kaggle via a kernel.
"""
import logging

import pandas as pd
import torch
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import DataLoader as TorchDataLoader

from src.argument_parser import parse_submission_arguments
from src.optimization.experiment import Experiment
from src.data.dataset import APTOSSubmissionDataset
from src.preprocess.pipeline import Pipeline
from src.ml import inference


def main(
        checkpoint_file_path: str,
        data_directory: str = "../input/aptos2019-blindness-detection/test_images",
        data_frame: str = "../input/aptos2019-blindness-detection/test.csv",
        sample_submission: str = "../input/aptos2019-blindess-detection/sample_submission.csv"
):
    print("Beginning submission")
    # Always load the first iteration from cross validation? Should really re-train on the whole dataset
    checkpoint = torch.load(checkpoint_file_path)
    print("Loaded checkpoint")

    experiment_state_dict = checkpoint["experiment"]
    experiment_state_dict.update(
        train_test_directories=[data_directory],
        train_test_data_frames=[data_frame]
    )
    experiment = Experiment.from_dict(experiment_state_dict)
    print("Initialised experiment: %s", experiment)

    pipeline = Pipeline(experiment.pipeline_stages())
    print("Initialised pipeline")

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    dataset = TorchConcatDataset(
        [APTOSSubmissionDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )
    print("Initialised dataset")

    loader = TorchDataLoader(
        dataset,
        batch_size=experiment.batch_size(),
    )
    print("Initialised loader")

    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    print("Initialised model")

    print("Beginning inference")
    predictions_proba, predictions, ids = inference(model, loader, "cpu")

    sample = pd.read_csv(sample_submission)
    sample.diagnosis = predictions

    sample.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    args = parse_submission_arguments()
    print("Running submission with following args: %s", args)

    main(
        args.checkpoint_file_path,
        args.data_directory,
        args.data_frame
    )
