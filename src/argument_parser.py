import argparse


def parse_training_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--develop-mode",
        action="store_true",
        help="Whether to run in develop mode or not."
    )

    arg_parser.add_argument(
        "--develop-mode-samples",
        type=int,
        help="How many samples to use when running in develop mode.",
        default=10
    )

    arg_parser.add_argument(
        "--data-loader-workers",
        type=int,
        help="How many data loaders to run in parallel.",
        default=1
    )

    arg_parser.add_argument(
        "--cross-validation-iterations",
        type=int,
        help="How many train-test cross validation splits to make per experiment.",
        default=3
    )

    arg_parser.add_argument(
        "--results-directory",
        type=str,
        help="Base directory in which to store results.",
        default="./results"
    )

    arg_parser.add_argument(
        "--device",
        type=str,
        help="Base directory in which to store results.",
        default="cuda:0",
        choices=["cpu", "cuda:0", "cuda:1"]
    )

    return arg_parser.parse_args()
