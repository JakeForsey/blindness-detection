from abc import ABC


class ExperimentGenerator(ABC):
    # TODO interface that abstracts hyper parameter optimization e.g. baysian optimization, grid search
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass
