"""
Submits predictions to Kaggle via a kernel.
"""
import logging
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


LOGGER = logging.getLogger(__name__)


def main(
        experiment_id: str,
        results_directory: str = "results",
        input_directory: str = "../input/aptos2019-blindness-detection"
):
    LOGGER.info("Beginning submission")
    # Always load the first iteration from cross validation? Should really re-train on the whole dataset
    checkpoint = torch.load(os.path.join(results_directory, experiment_id, "1-checkpoint.pth"))
    LOGGER.info("Loaded checkpoint")

    experiment_state_dict = checkpoint["experiment"]
    experiment_state_dict.update(
        train_test_directories=[os.path.join(input_directory, "test_images")],
        train_test_data_frames=[os.path.join(input_directory, "test.csv")]
    )
    experiment = Experiment.from_dict(experiment_state_dict)
    LOGGER.info("Initialised experiement")

    pipeline = Pipeline(experiment.pipeline_stages())
    LOGGER.info("Initialised pipeline")

    dfs = experiment.train_test_data_frames()
    directories = experiment.train_test_directories()

    dataset = TorchConcatDataset(
        [APTOSSubmissionDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )
    LOGGER.info("Initialised dataset")

    loader = TorchDataLoader(
        dataset,
        batch_size=experiment.batch_size(),
    )
    LOGGER.info("Initialised loader")

    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    LOGGER.info("Initialised model")

    LOGGER.info("Beggining inference")
    predictions_proba, predictions, ids = inference(model, loader, "cpu")

    sample = pd.DataFrame({
        "id_code": ids,
        "diagnosis": predictions,
    })

    sample.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    args = parse_submission_arguments()
    LOGGER.info("Running submission with following args: %s", args)

    main(
        args.experiment_id,
        args.results_directory,
        args.input_directory
    )
