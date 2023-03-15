"""Queries from the GPT-3 models."""

import json

from tqdm import tqdm

from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService

from dataset import TrainTestSets


_DEFAULT_API_KEY = 'Y3kBDun6qevSxiAn6Mjm33amYvvMw9sm'
_API_WEBSITE = 'https://crfm-models.stanford.edu'

_MODEL_NAMES = ['ada', 'babbage', 'curie', 'davinci']
_OPENAI_MODEL_NAMES = [f'openai/{name}' for name in _MODEL_NAMES]

_QUESTION_KEY = 'question'
_ANSWER_KEY = 'answer'
_PROMPT_KEY = 'prompt'
_RESPONSE_KEY = 'response'

_DEFAULT_MODEL_NAME = _OPENAI_MODEL_NAMES[0]

_RESULT_DIRECTORY = 'results/gsm8k'


# An example of how to use the request API.
auth = Authentication(api_key=_DEFAULT_API_KEY)
service = RemoteService(_API_WEBSITE)

# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# Builds the GradeSchoolMath dataset.
train_test_data = TrainTestSets()

# Specifies a model.
model = _DEFAULT_MODEL_NAME

samples = [(sample[_QUESTION_KEY], sample[_ANSWER_KEY]) for sample in train_test_data.train.data[195:196]]


def get_prompt(curr_question, samples):
    prefix = 'Answer the last question.\n\n'
    prompt = prefix
    for question, answer in samples:
        curr_string = (f'Question: {question}\n' +
                       f'Answer: {answer}\n\n')
        prompt += curr_string
    prompt += (f'Question: {curr_question}\nAnswer: ')
    return prompt


def get_response(model, question, auth):
    prompt = get_prompt(question, samples)
    request = Request(model=model, prompt=prompt)
    request_result: RequestResult = service.make_request(auth, request)
    return prompt, request_result.completions[0].text


def get_responses(model, questions, auth, first_n=None):
    if first_n is None:
        first_n = len(questions)
    return [get_response(model, question, auth)
            for question in tqdm(questions[:first_n])]


def query_model(model, dataset, auth, first_n=None):
    questions = [interaction[_QUESTION_KEY] for interaction in dataset.data]
    return get_responses(model, questions, auth, first_n)


def save_model_results(model, dataset, auth, path, first_n=None):
    prompt_response_pairs = query_model(model, dataset, auth, first_n)
    questions = [interaction[_QUESTION_KEY] for interaction in dataset.data]
    answers = [interaction[_ANSWER_KEY] for interaction in dataset.data]

    with open(path, 'w') as file:
        for question, answer, (prompt, response) in zip(questions, answers, prompt_response_pairs):
            record = {
                _QUESTION_KEY: question,
                _ANSWER_KEY: answer,
                _PROMPT_KEY: prompt,
                _RESPONSE_KEY: response
            }
            file.write(json.dumps(record) + '\n')


save_model_results('openai/ada', train_test_data.train, auth,
                   path=f'{_RESULT_DIRECTORY}/train_ada_3_shot.jsonl',
                   first_n=3)
