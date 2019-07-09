import pandas as pd

from src.optimization.experiment import Experiment


class Result:
    """Represents a completed experiment."""
    def __init__(self, experiment: Experiment, metric_df: pd.DataFrame):
        self._experiement = experiment
        self._metric_df = metric_df
