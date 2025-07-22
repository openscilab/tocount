# -*- coding: utf-8 -*-
"""tocount functions."""
from enum import Enum
from .rule_based import universal_tokens_estimator
from .rule_based import openai_tokens_estimator_gpt_3_5
from .rule_based import openai_tokens_estimator_gpt_4
from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE

class TextEstimator(Enum):
    """Text token estimator enum."""

    RULE_BASED_UNIVERSAL = "RULE BASED UNIVERSAL"
    RULE_BASED_GPT_3_5 = "RULE BASED GPT 3.5"
    RULE_BASED_GPT_4 = "RULE BASED GPT 4"
    DEFAULT = RULE_BASED_UNIVERSAL

text_estimator_map = {TextEstimator.RULE_BASED_UNIVERSAL: universal_tokens_estimator,
                      TextEstimator.RULE_BASED_GPT_3_5: openai_tokens_estimator_gpt_3_5,
                      TextEstimator.RULE_BASED_GPT_4: openai_tokens_estimator_gpt_4}


def estimate_text_tokens(text: str, estimator: TextEstimator = TextEstimator.DEFAULT) -> int:
    """
    Estimate text tokens number.

    :param text: input text
    :param estimator: estimator type
    :return: tokens number
    """
    if not isinstance(text, str):
        raise ValueError(INVALID_TEXT_MESSAGE)
    if not isinstance(estimator, TextEstimator):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text=text)

