"""Constructs a dataset from GradeSchoolMath data."""

from __future__ import annotations

import dataclasses
import json

from typing import Optional


_DEFAULT_TRAIN_PATH = 'data/gsm8k/train.jsonl'
_DEFAULT_TEST_PATH = 'data/gsm8k/test.jsonl'


class Dataset:
    def __init__(self, path: str = _DEFAULT_TRAIN_PATH) -> None:
        self.path = path
        self.data = []

        with open(self.path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line.rstrip('\n|\r')))


@dataclasses.dataclass
class TrainTestSets:
    train: Dataset
    test: Dataset

    def __init__(
        self,
        train: Optional[TrainTestSets] = None,
        test: Optional[TrainTestSets] = None) -> None:
        self.train = Dataset(_DEFAULT_TRAIN_PATH) if train is None else train
        self.test = Dataset(_DEFAULT_TEST_PATH) if test is None else test
