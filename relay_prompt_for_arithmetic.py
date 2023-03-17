"""Relays generated helpful prompts to the GPT-3 models for arithmetic datasets."""

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
from prompt_dataset import PromptDataset


_DEFAULT_API_KEY = 'Y3kBDun6qevSxiAn6Mjm33amYvvMw9sm'
_API_WEBSITE = 'https://crfm-models.stanford.edu'

_MODEL_NAMES = ['ada', 'babbage', 'curie', 'davinci']
_OPENAI_MODEL_NAMES = [f'openai/text-{name}-001' for name in _MODEL_NAMES]

_QUESTION_KEY = 'question'
_ANSWER_KEY = 'answer'
_PROMPT_KEY = 'prompt'
_RESPONSE_KEY = 'response'
_NEW_PROMPT_KEY = 'new_prompt'
_NEW_RESPONSE_KEY = 'new_response'

_DEFAULT_MODEL_NAME = _OPENAI_MODEL_NAMES[0]

_SOURCE_DIRECTORY = 'results/arithmetics/prompt_across_tags'
_RESULT_DIRECTORY = 'results/arithmetics/answer_to_prompt_across_tags'

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


def process_example(example: Sequence[Any]) -> str:
    question, helpful_question, helpful_answer = example
    return (f'Q: {question}\n' +
            f'Prompt: Q: {helpful_question}\n' + 
            f'A: {helpful_answer}\n\n')


def process_question_vanilla(question: str) -> str:
    return f'Q: {question}\nA: '


def get_prompt(
    curr_question: str,
    helper_text: str,
    process_question: Callable[[str], str] = process_question_vanilla,
) -> str:
    return helper_text.strip('\n') + '\n\n' + process_question(curr_question)


def get_response(
    model: str,
    question: str,
    helper_text: str,
    request_func: _RequestFunc,
) -> Tuple[str, str]:
    prompt = get_prompt(question, helper_text)
    request = Request(model=model, prompt=prompt, max_tokens=20, stop_sequences=['.'])
    request_result: RequestResult = request_func(request=request)
    return prompt, request_result.completions[0].text


def get_data_slice(
    prompt_dataset: PromptDataset,
    first_n: Optional[int] = None,
) -> pd.DataFrame:
    if first_n is None:
        first_n = len(prompt_dataset.data)
    return prompt_dataset.data.iloc[:first_n]


def query_model(
    model: str,
    prompt_dataset: PromptDataset,
    request_func: _RequestFunc,
    first_n: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    data = get_data_slice(prompt_dataset, first_n)
    for _, row in tqdm(data.iterrows()):
        yield get_response(
            model, row[_QUESTION_KEY], row[_RESPONSE_KEY], request_func)


def save_model_results(
    path: str,
    model: str,
    prompt_dataset: PromptDataset,
    request_func: _RequestFunc,
    first_n: Optional[int] = None
):
    prompt_response_pairs = query_model(
        model, prompt_dataset, request_func, first_n)
    data = get_data_slice(prompt_dataset, first_n)
    original_query_tuples = [
        (row[_QUESTION_KEY], row[_ANSWER_KEY], row[_PROMPT_KEY], row[_RESPONSE_KEY])
        for _, row in data.iterrows()]

    with open(path, 'a') as file:
        for (question, answer, prompt, response), (new_prompt, new_response) in zip(
            original_query_tuples, prompt_response_pairs):
            record = {
                _QUESTION_KEY: question,
                _ANSWER_KEY: answer,
                _PROMPT_KEY: prompt,
                _RESPONSE_KEY: response,
                _NEW_PROMPT_KEY: new_prompt,
                _NEW_RESPONSE_KEY: new_response,
            }
            file.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    make_request = setup()

    model_index = 3
    model_name = _MODEL_NAMES[model_index]
    openai_model_name = _OPENAI_MODEL_NAMES[model_index]
    model = openai_model_name

    first_n = 100
    source_directory = _SOURCE_DIRECTORY
    result_directory = _RESULT_DIRECTORY
    extension = '.jsonl'

    # Builds the prompt datasets.
    datasets = {
        filename: PromptDataset(filename, source_directory)
        for filename in sorted(os.listdir(source_directory))
        if filename.endswith('.jsonl')
    }

    for filename, prompt_dataset in datasets.items():
        path = os.path.join(
            result_directory,
            filename.rstrip(extension) + '_answer' + extension)
        if os.path.exists(path):
            continue

        print(f'*** Processing: {path}')
        save_model_results(
            path=path,
            model=model,
            prompt_dataset=prompt_dataset,
            request_func=make_request,
            first_n=first_n)
