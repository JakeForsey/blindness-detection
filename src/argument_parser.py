import argparse
import re
from typing import Iterable

def range_argument(range_str: str) -> Iterable[int]:
    """
    Parse a range string in the form x-y into a python range.
    """
    pattern = re.compile(r"^(?P<from>\d+)(-(?P<to>\d+))?$")

    match = pattern.match(range_str)

    if not match:
        raise argparse.ArgumentTypeError(
            'Required range format <from>-<to>, inclusive.',
        )

    frm = int(match['from'])

    if match['to'] is None:
        to = frm
    else:
        to = int(match['to'])

    if frm > to:
        raise argparse.ArgumentTypeError(
            'Cannot parse range with <from> greater than <to>.',
        )

    return range(frm, to+1)

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

    arg_parser.add_argument(
        "--experiments",
        type=range_argument,
        help="A range of experiment indices to run in the form <from>-<to>, inclusive.",
        action='append',
    )

    return arg_parser.parse_args()
