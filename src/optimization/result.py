import json
import os
import sqlite3

import pandas as pd

from src.optimization.experiment import Experiment


class Result:
    """Represents a completed experiment."""

    def __init__(self, experiment: Experiment, results_df: pd.DataFrame):
        self._experiment = experiment
        self._results_df = results_df

    def json_file_path(self, directory):
        return os.path.join(directory, self._experiment.id() + ".json")

    def persist(self, directory: str, connection: sqlite3.Connection):

        # Persist the performance metrics
        self._results_df.to_sql(
            "results",
            connection,
            if_exists="append",
            index=False
        )

        # Persist the experiment definition
        json_file_path = self.json_file_path(directory)
        if not os.path.isfile(json_file_path):
            with open(json_file_path, 'w') as f:
                json.dump(self._experiment.to_json(), f)

