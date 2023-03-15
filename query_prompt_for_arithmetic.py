"""Queries helpful prompts from the GPT-3 models for arithmetic datasets."""

import functools
import json
import os
import random

from tqdm import tqdm
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import pandas as pd

from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService

from arithmetic_dataset import ArithmeticDataset, ARITHMETIC_DATASET_TAGS


_DEFAULT_API_KEY = 'Y3kBDun6qevSxiAn6Mjm33amYvvMw9sm'
_API_WEBSITE = 'https://crfm-models.stanford.edu'

_MODEL_NAMES = ['ada', 'babbage', 'curie', 'davinci']
_OPENAI_MODEL_NAMES = [f'openai/text-{name}-001' for name in _MODEL_NAMES]

_QUESTION_KEY = 'question'
_ANSWER_KEY = 'answer'
_PROMPT_KEY = 'prompt'
_RESPONSE_KEY = 'response'

_DEFAULT_MODEL_NAME = _OPENAI_MODEL_NAMES[0]

_RESULT_DIRECTORY = 'results/arithmetics'

# Number of testing and validation question-answer pairs.
_N_TEST = 100
_N_VAL = 100

_RequestFunc = Callable[[Request], RequestResult]


def setup(
    api_key: str = _DEFAULT_API_KEY,
    website: str = _API_WEBSITE,
    verbose: bool = True) -> _RequestFunc:
    # An example of how to use the request API.
    auth = Authentication(api_key=api_key)
    service = RemoteService(website)

    # Access account and show my current quotas and usages
    account: Account = service.get_account(auth)
    if verbose:
        print(account.usages)

    make_request: _RequestFunc = functools.partial(
        service.make_request,
        auth=auth)
    return make_request


# Builds the arithmetic dataset.
datasets = {
    tag: ArithmeticDataset(tag)
    for tag in ARITHMETIC_DATASET_TAGS
}


def sample_helper_tag(
    question: str,
    helper_tag: str,
    n_shots: int = 0,
) -> Iterator[Tuple[str, str]]:
    del question  # Unused.
    val_data = datasets[helper_tag].data.iloc[_N_TEST:]
    indices = random.sample(range(len(val_data)), n_shots)
    for _, row in val_data.iloc[indices].iterrows():
        yield row[_QUESTION_KEY], row[_ANSWER_KEY]


def process_example(example: Sequence[Any]) -> str:
    question, helpful_question, helpful_answer = example
    return (f'Q: {question}\n' +
            f'Prompt: Q: {helpful_question}\n' + 
            f'A: {helpful_answer}\n\n')


def process_question_for_prompt(question: str) -> str:
    return f'Q: {question}\nPrompt: '


def get_prompt(
    curr_question: str,
    helper_tag: str,
    n_shots: int = 0,
    sample_examples: Callable[[str, str, int], Iterator[Sequence[Any]]] = sample_helper_tag,
    process_example: Callable[[Sequence[Any]], str] = process_example,
    process_question: Callable[[str], str] = process_question_for_prompt,
) -> str:
    prompt = 'Write a helpful prompt for the question.\n\n'
    for example in sample_examples(curr_question, helper_tag, n_shots):
        prompt += process_example(example)
    return prompt + process_question(curr_question)


def get_response(
    model: str,
    question: str,
    helper_tag: str,
    n_shots: int,
    request_func: _RequestFunc,
) -> Tuple[str, str]:
    prompt = get_prompt(question, helper_tag, n_shots)
    request = Request(model=model, prompt=prompt, max_tokens=10, stop_sequences=['.'])
    request_result: RequestResult = request_func(request=request)
    return prompt, request_result.completions[0].text


def get_data_slice(
    tag: str,
    indices: Optional[Sequence[int]] = None,
    first_n: Optional[int] = None,
) -> pd.DataFrame:
    data = (datasets[tag].data.iloc[:_N_TEST]
            if indices is None else
            datasets[tag].data.iloc[indices])
    if first_n is None:
        first_n = len(data)
    return data.iloc[:first_n]


def query_model(
    model: str,
    tag: str,
    helper_tag: str,
    n_shots: int,
    request_func: _RequestFunc,
    indices: Optional[Sequence[int]] = None,
    first_n: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    data = get_data_slice(tag, indices, first_n)
    for _, row in tqdm(data.iterrows()):
        yield get_response(
            model, row[_QUESTION_KEY], helper_tag, n_shots, request_func)


def save_model_results(
    path: str,
    model: str,
    tag: str,
    helper_tag: str,
    n_shots: int,
    request_func: _RequestFunc,
    indices: Optional[Sequence[int]] = None,
    first_n: Optional[int] = None
):
    prompt_response_pairs = query_model(
        model, tag, helper_tag, n_shots, request_func, indices, first_n)
    data = get_data_slice(tag, indices, first_n)
    question_answer_pairs = [
        (row[_QUESTION_KEY], row[_ANSWER_KEY]) for _, row in data.iterrows()]

    with open(path, 'a') as file:
        for (question, answer), (prompt, response) in zip(
            question_answer_pairs, prompt_response_pairs):
            record = {
                _QUESTION_KEY: question,
                _ANSWER_KEY: answer,
                _PROMPT_KEY: prompt,
                _RESPONSE_KEY: response
            }
            file.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    make_request = setup()

    model_index = 3
    model_name = _MODEL_NAMES[model_index]
    openai_model_name = _OPENAI_MODEL_NAMES[model_index]
    model = openai_model_name
    for tag in ARITHMETIC_DATASET_TAGS:
        for helper_tag in ARITHMETIC_DATASET_TAGS:
            if helper_tag == tag:
                continue

            n_shots = 1
            first_n = 100
            path = f'{_RESULT_DIRECTORY}/{model_name}_{tag}_{n_shots}_{first_n}_{helper_tag}.jsonl'
            if os.path.exists(path):
                continue

            save_model_results(
                path=path,
                model=model,
                tag=tag,
                helper_tag=helper_tag,
                n_shots=n_shots,
                request_func=make_request,
                first_n=first_n)
