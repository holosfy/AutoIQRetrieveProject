#!/usr/bin/env python
# -*-coding=utf-8-*-
"""
@Author : liumiangang
@Date : 2023/10/25
"""
from typing import Dict, List

import numpy as np

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, root_validator


class LCSLengthExampleSelector(BaseExampleSelector, BaseModel):

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    threshold: float = -1.0
    """Threshold at which algorithm stops. Set to -1.0 by default.
    
    For negative threshold:
    select_examples sorts examples by ngram_overlap_score, but excludes none.
    For threshold greater than 1.0:
    select_examples excludes all examples, and returns an empty list.
    For threshold equal to 0.0:
    select_examples sorts examples by ngram_overlap_score,
    and excludes examples with no ngram overlap with input.
    """

    top_k: int = 1

    @root_validator(pre=True)
    def check_dependencies(cls, values: Dict) -> Dict:
        """Check that valid dependencies exist."""
        try:
            from nltk.translate.bleu_score import (  # noqa: F401
                SmoothingFunction,
                sentence_bleu,
            )
        except ImportError as e:
            raise ImportError(
                "Not all the correct dependencies for this ExampleSelect exist."
                "Please install nltk with `pip install nltk`."
            ) from e

        return values

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def lcs(self, str1, str2):
        len1 = len(str1)
        len2 = len(str2)
        ans = 0

        array = [[0 for _ in range(len2 + 1)], [0 for _ in range(len2 + 1)]]
        current = 0
        last = 1

        for i in range(len1):
            for j in range(len2):
                if str1[i] == str2[j]:
                    array[current][j + 1] = array[last][j] + 1
                    ans = max(ans, array[current][j + 1])
                else:
                    array[current][j + 1] = max(array[current][j + 1], array[last][j])
            current, last = last, current
        return ans


    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Return list of examples sorted by ngram_overlap_score with input.

        Descending order.
        Excludes any examples with ngram_overlap_score less than or equal to threshold.
        """
        inputs = list(input_variables.values())
        examples = []
        k = len(self.examples)
        score = [0.0] * k
        first_prompt_template_key = self.example_prompt.input_variables[0]
        second_prompt_template_key = self.example_prompt.input_variables[1]

        for i in range(k):
            if inputs[0] == self.examples[i][first_prompt_template_key]:
                score[i] = -100
            else:
                score1 = self.lcs(inputs[0], self.examples[i][first_prompt_template_key])
                score2 = self.lcs(inputs[0], self.examples[i][second_prompt_template_key])
                score[i] = 1.0 * score1 + 0.1 * score2


        while True:
            arg_max = np.argmax(score)
            if (score[arg_max] < self.threshold) or abs(
                score[arg_max] - self.threshold
            ) < 1e-9 or len(examples) >= self.top_k:
                break

            examples.append(self.examples[arg_max])
            score[arg_max] = self.threshold - 1.0

        return examples