"""Constructs a dataset from generated arithmetic data."""

from __future__ import annotations
from typing import Sequence

import os

import pandas as pd

_DEFAULT_DATA_DIRECTORY = 'data/arithmetics'

ARITHMETIC_DATASET_TAGS: Sequence = tuple([
    f'{n_digits}D{mode}'
    for n_digits in range(2, 6)
    for mode in {'+', '-'}
] + ['2Dx', '1DC'])

QUESTION_KEY = 'question'
ANSWER_KEY = 'answer'


class ArithmeticDataset:
    def __init__(self, tag, data_directory: str = _DEFAULT_DATA_DIRECTORY) -> None:
        self.data_directory = data_directory
        self.tag = tag
        self.path = os.path.join(data_directory, tag) + '.csv'

        self.data = pd.read_csv(self.path)
