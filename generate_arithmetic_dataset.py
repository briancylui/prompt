"""Saves question-answer tuples for arithmetic datasets for GPT-3 models.

Reference: Page 21-22 of https://arxiv.org/pdf/2005.14165.pdf.
"""

import os

import pandas as pd

from arithmetic_dataset import ARITHMETIC_DATASET_TAGS
from sample_arithmetic_dataset import ArithmeticDatasetGenerator


def get_dataframe(tag: str, size: int) -> pd.DataFrame:
    if tag not in set(ARITHMETIC_DATASET_TAGS):
        raise NotImplementedError

    dataset = ArithmeticDatasetGenerator(tag)
    questions = []
    answers = []
    for sample in dataset.sample(size):
        questions.append(sample.question)
        answers.append(sample.answer)

    return pd.DataFrame.from_dict({
        'question': questions,
        'answer': answers,
    })
    

def save_data(tag: str, size: int, path: str) -> None:
    data = get_dataframe(tag, size)
    data.to_csv(path, index=False)


if __name__ == '__main__':
    sample_size = 2000
    save_directory = 'data/arithmetics'
    for tag in ARITHMETIC_DATASET_TAGS:
        save_data(tag, sample_size, os.path.join(save_directory, tag) + '.csv')
