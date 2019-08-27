"""
Submits predictions to Kaggle via a kernel.
"""
import logging
import os
import random

import cv2
import matplotlib.pyplot as plt

import pandas as pd
import torch
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import DataLoader as TorchDataLoader

from src.argument_parser import parse_submission_arguments
from src.constants import CLASSES
from src.optimization.experiment import Experiment
from src.data.dataset import APTOSSubmissionDataset
from src.preprocess.pipeline import Pipeline
from src.preprocess.normalize import eight_bit_normalization
from src.preprocess.augmentation import AugmentedCollate
from src.ml import inference


def main(
        checkpoint_file_path: str,
        data_directory: str = "../input/aptos2019-blindness-detection/test_images",
        data_frame: str = "../input/aptos2019-blindness-detection/test.csv",
        sample_submission: str = "../input/aptos2019-blindness-detection/sample_submission.csv",
        device: str = "cuda:0",
        samples_to_visualise: int = 30
):
    print("Beginning submission")
    print(f"Using {device} for submissions")

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
    test_augmentations = AugmentedCollate(experiment.test_augmentation_stages())

    dataset = TorchConcatDataset(
        [APTOSSubmissionDataset(df, directory, pipeline) for df, directory in zip(dfs, directories)]
    )
    print("Initialised dataset")

    loader = TorchDataLoader(
        dataset,
        batch_size=experiment.batch_size(),
        collate_fn=test_augmentations
    )
    print("Initialised loader")

    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])

    print("Initialised model")

    print("Beginning inference")
    predictions_proba, predictions, ids = inference(model, loader, device)

    sample = pd.read_csv(sample_submission)
    sample.diagnosis = predictions

    sample.to_csv("submission.csv", index=False)

    visualisations_directory = os.path.join("samples", experiment.id())
    if not os.path.isdir(visualisations_directory):
        os.makedirs(visualisations_directory, exist_ok=True)

    sample_indexes = random.sample(population=list(range(len(ids))), k=samples_to_visualise)
    for sample_index in sample_indexes:
        id_ = ids[sample_index]
        prediction = predictions[sample_index]
        proba = predictions_proba[sample_index].cpu().numpy()

        # Second argument is mask
        image, _ = eight_bit_normalization(
            cv2.cvtColor(cv2.imread(os.path.join(data_directory, f"{id_}.png")), cv2.COLOR_BGR2RGB),
            # No mask
            None
        )

        plt.figure(figsize=(30, 15))
        fig, axes = plt.subplots(1, 2)
        fig.set_dpi(150)
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].title.set_text(f"Raw image")

        bar_list = axes[1].bar(x=CLASSES, height=proba - proba.min())
        bar_list[prediction].set_color('r')
        axes[1].title.set_text(f"Classes")
        plt.setp(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(visualisations_directory, f"{id_}.jpeg"))
        plt.clf()


if __name__ == "__main__":
    args = parse_submission_arguments()
    print("Running submission with following args: %s", args)

    main(
        args.checkpoint_file_path,
        data_directory=args.data_directory,
        data_frame=args.data_frame,
        sample_submission=args.sample_submission,
        device=args.device,
        samples_to_visualise=args.samples_to_visualise
    )
