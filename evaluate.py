"""Evaluates GPT-3 responses against ground-truth answers."""

import glob
import json
import os
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

_QUESTION_KEY = 'question'
_ANSWER_KEY = 'answer'
_PROMPT_KEY = 'prompt'
_RESPONSE_KEY = 'response'

_RESULT_DIRECTORY = 'results/arithmetics'


def get_last_integer(response: str) -> Optional[int]:
    for end in range(len(response) - 1, -1, -1):
        if not response[end].isnumeric():
            continue

        start = end
        while start >= 0 and response[start].isnumeric():
            start -= 1  # Must run for at least one iteration.
        return int(response[(start + 1):(end + 1)])
    return None


class Evaluator:
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = paths
    
    def evaluate(self) -> Mapping[str, Mapping[str, Union[int, float]]]:
        self.result = {}
        
        for path in self.paths:
            records = []
            with open(path, 'r') as file:
                for json_line in file:
                    records.append(json.loads(json_line))
            
            total_n = 0
            n_matches = 0

            for record in records:
                total_n += 1
                answer = int(record[_ANSWER_KEY])
                response = record[_RESPONSE_KEY]
                model_answer = get_last_integer(response)
                if model_answer is not None and answer == model_answer:
                    n_matches += 1
            self.result[path] = {
                'accuracy': n_matches / total_n,
                'n_correct': n_matches,
                'total': total_n}
        return self.result


if __name__ == '__main__':
    save_path = os.path.join(_RESULT_DIRECTORY, 'summary.csv')
    evaluator = Evaluator(
        paths=sorted(glob.glob(_RESULT_DIRECTORY + '/*.jsonl'))
    )
    results = evaluator.evaluate()
    pd.DataFrame.from_dict(
        results, orient='index'
        ).reset_index().to_csv(save_path, index=False)