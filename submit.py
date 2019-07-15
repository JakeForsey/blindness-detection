"""
Submits predictions to Kaggle via a kernel.
"""
import os

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
        experiment_id: str,
        results_directory: str = "results",
        input_directory: str = "../input/aptos2019-blindness-detection"
):
    # Always load the first iteration from cross validation? Should really re-train on the whole dataset
    checkpoint = torch.load(os.path.join(results_directory, experiment_id, "1-checkpoint.pth"))

    experiment_state_dict = checkpoint["experiment"]
    experiment_state_dict.update(
        train_test_directories=[os.path.join(input_directory, "test_images")],
        train_test_data_frames=[os.path.join(input_directory, "test.csv")]
    )
    experiment = Experiment.from_dict(experiment_state_dict)

    pipeline = Pipeline(experiment.pipeline_stages())

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    dataset = TorchConcatDataset(
        [APTOSSubmissionDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )

    loader = TorchDataLoader(
        dataset,
        batch_size=experiment.batch_size(),
    )

    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])

    predictions_proba, predictions, ids = inference(model, loader, "cpu")

    sample = pd.read_csv(os.path.join(input_directory, "sample_submission.csv"))

    sample.diagnosis = predictions
    sample.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    args = parse_submission_arguments()

    main(
        args.experiment_id,
        args.results_directory,
        args.input_directory
    )
