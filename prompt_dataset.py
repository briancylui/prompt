"""Constructs a dataset from generated prompts for arithmetic data."""

from __future__ import annotations

import json
import os

import pandas as pd

_DEFAULT_DATA_DIRECTORY = 'results/arithmetics/prompt'

QUESTION_KEY = 'question'
ANSWER_KEY = 'answer'
PROMPT_KEY = 'prompt'
RESPONSE_KEY = 'response'


class PromptDataset:
    def __init__(self, filename: str, data_directory: str = _DEFAULT_DATA_DIRECTORY) -> None:
        self.data_directory = data_directory
        self.filename = filename
        self.path = os.path.join(data_directory, filename)

        records = []
        with open(self.path, 'r') as file:
            for json_line in file:
                records.append(json.loads(json_line))

        self.data = pd.DataFrame.from_records(records)
