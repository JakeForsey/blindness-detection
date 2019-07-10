"""
Searches models and hyper parameters.
"""
import logging
from typing import Tuple
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader

from src.preprocess.pipeline import Pipeline
from src.data.dataset import APTOSDataset
from src.optimization.hand_tuned import HandTunedExperiements
from src.optimization.experiment import Experiment
from src.optimization.result import Result


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DEVELOP_MODE = False
DEVELOP_MODE_PERCENT = 5

if DEVELOP_MODE:
    LOGGER.warn("Running in develop mode, only %s percent of data will be used.", DEVELOP_MODE_PERCENT)

CROSS_VALIDATION_ITERATIONS = 3

DEVICE = "cuda:0"
if DEVICE == "cuda:0":
    torch.cuda.set_device(torch.device("cuda:0"))


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy)
    )

    return test_loss, accuracy


def split_data_frame(df: pd.DataFrame, iterations: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

    for iteration in range(iterations):
        test_df = df.sample(frac=test_size, random_state=iteration)
        train_df = df.drop(test_df.index)

        test_df.reset_index(inplace=True)
        train_df.reset_index(inplace=True)

        yield train_df, test_df


def run_experiment(experiment: Experiment, debug_pipeline: bool = False) -> List[Result]:
    df = experiment.train_test_data_frame()

    results = []
    for cv_iteration, (train_df, test_df) in enumerate(split_data_frame(
            df, CROSS_VALIDATION_ITERATIONS, experiment.test_size()
    )):
        LOGGER.info("Cross validation iteration: %s", cv_iteration)

        if DEVELOP_MODE:
            train_df = train_df.sample(frac=DEVELOP_MODE_PERCENT / 100).reset_index()
            test_df = test_df.sample(frac=DEVELOP_MODE_PERCENT / 100).reset_index()

        pipeline = Pipeline(experiment.pipeline_stages(), debug=debug_pipeline)

        train_ds = APTOSDataset(
            train_df,
            experiment.train_test_directory(),
            pipeline
        )
        train_loader = TorchDataLoader(
            train_ds,
            batch_size=experiment.batch_size()
        )

        test_ds = APTOSDataset(
            test_df,
            experiment.train_test_directory(),
            pipeline
        )
        test_loader = TorchDataLoader(
            test_ds,
            batch_size=experiment.batch_size()
        )

        model = experiment.model(input_shape=train_ds[0][0].shape)
        if DEVICE == "cuda:0":
            model.cuda()

        optimizer_class, optim_kwargs = experiment.optimizer()
        optimizer = optimizer_class(model.parameters(), **optim_kwargs)

        metric_df = pd.DataFrame(columns=["experiment_id", "epoch", "test_loss", "test_accuracy"])
        for epoch in range(1, experiment.max_epochs() + 1):
            LOGGER.info("Epoch: %s", epoch)

            train(1, model, DEVICE, train_loader, optimizer, epoch)
            loss, accuracy = test(model, DEVICE, test_loader)

            metric_df = metric_df.append({
                "experiment_id": experiment.id(),
                "cross_validation_iteration": cv_iteration,
                "epoch": epoch,
                "test_loss": loss,
                "test_accuracy": accuracy
            }, ignore_index=True)

        results.append(Result(experiment, metric_df))

    return results


def main():
    experiment_generator = HandTunedExperiements()

    results = []
    for experiment in experiment_generator:

        results = run_experiment(experiment)

    # TODO Send results to db
    import pickle
    with open("test.p", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
