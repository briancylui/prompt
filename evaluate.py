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
_NEW_PROMPT_KEY = 'new_prompt'
_NEW_RESPONSE_KEY = 'new_response'

_RESULT_DIRECTORY = 'results/arithmetics/answer_to_prompt_across_tags'


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
            filename = path.split('/')[-1].rstrip('.jsonl')
            model = filename.split('_')[0]
            filename = filename.lstrip(model + '_')
            tag = filename[:3]

            unwanted_suffix = '_prompt_answer'
            if filename.endswith(unwanted_suffix):
                filename = filename.rstrip(unwanted_suffix)
            helper_tag = filename[-3:] if not filename[-3:].isnumeric() else tag
            n_shots = int(filename[4])

            records = []
            with open(path, 'r') as file:
                for json_line in file:
                    records.append(json.loads(json_line))
            
            total_n = 0
            n_matches = 0

            for record in records:
                total_n += 1
                answer = int(record[_ANSWER_KEY])
                response = record[_NEW_RESPONSE_KEY] if _NEW_RESPONSE_KEY in record else record[_RESPONSE_KEY]
                model_answer = get_last_integer(response)
                if model_answer is not None and answer == model_answer:
                    n_matches += 1

            self.result[path] = {
                'model': model,
                'tag': tag,
                'helper_tag': helper_tag,
                'n_shots': n_shots,
                'accuracy': n_matches / total_n,
                'n_correct': n_matches,
                'total': total_n}

        return self.result


if __name__ == '__main__':
    result_directory = _RESULT_DIRECTORY

    save_path = os.path.join(result_directory, 'summary.csv')
    evaluator = Evaluator(
        paths=sorted(glob.glob(result_directory + '/*.jsonl'))
    )
    results = evaluator.evaluate()
    df = pd.DataFrame.from_dict(
        results, orient='index')
    df.index.name = 'source'
    df = df.reset_index()
    df.to_csv(save_path, index=False)
