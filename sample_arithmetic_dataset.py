"""Generates datasets for arithmetics for GPT-3 models.

Reference: Page 21-22 of https://arxiv.org/pdf/2005.14165.pdf.
"""

from dataclasses import dataclass
import enum
import frozendict

import random
from typing import Iterator


@enum.unique
class ComputationMode(enum.Enum):
    ADD = enum.auto()
    SUBTRACT = enum.auto()
    MULTIPLY = enum.auto()


_SIGN_TO_COMPUTATION_MODE = frozendict.frozendict({
    '+': ComputationMode.ADD,
    '-': ComputationMode.SUBTRACT,
    'x': ComputationMode.MULTIPLY,
})

_SIGNS = sorted(_SIGN_TO_COMPUTATION_MODE.keys())


def compute(first: int, second: int, mode: ComputationMode) -> int:
    if mode == ComputationMode.ADD:
        return first + second
    elif mode == ComputationMode.SUBTRACT:
        return first - second
    elif mode == ComputationMode.MULTIPLY:
        return first * second
    else:
        raise NotImplementedError


@dataclass
class ArithmeticDatasetSample:
    question: str
    answer: str


class ArithmeticDatasetGenerator:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def sample(self, n: int, seed: int = 0) -> Iterator[ArithmeticDatasetSample]:
        mode = self.tag[-1]
        if mode in {'+', '-', 'x'}:
            n_digits = int(self.tag[0])
            upper_bound = 10**n_digits

            mode_text = 'plus' if mode == '+' else (
                'minus' if mode == '-' else 'times')
            computation_mode = _SIGN_TO_COMPUTATION_MODE[mode]

            random.seed(seed)
                
            for _ in range(n):
                first = random.randrange(upper_bound)
                second = random.randrange(upper_bound)
                yield ArithmeticDatasetSample(
                    question=f'What is {first} {mode_text} {second}?',
                    answer=str(compute(first, second, computation_mode)))
        else:
            # self.tag == '1DC'
            n_digits = 1
            upper_bound = 10**n_digits

            random.seed(seed)
                
            for _ in range(n):
                first = random.randrange(upper_bound)
                second = random.randrange(upper_bound)
                third = random.randrange(upper_bound)
                first_mode = random.choice(_SIGNS)
                second_mode = random.choice(_SIGNS)
                first_sign = first_mode if first_mode != 'x' else '*'
                second_sign = second_mode if second_mode != 'x' else '*'
                answer = str(compute(
                    first, compute(second, third, _SIGN_TO_COMPUTATION_MODE[second_mode]),
                    _SIGN_TO_COMPUTATION_MODE[first_mode]))

                yield ArithmeticDatasetSample(
                    question=f'What is {first} {first_sign} ({second} {second_sign} {third})?',
                    answer=answer)
