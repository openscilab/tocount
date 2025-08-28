# -*- coding: utf-8 -*-
"""tocount functions."""
from enum import Enum
from .rule_based import universal_tokens_estimator
from .rule_based import openai_tokens_estimator_gpt_3_5
from .rule_based import openai_tokens_estimator_gpt_4
from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE


class _TextEstimatorRuleBased(Enum):
    """Rule based text token estimator enum."""

    UNIVERSAL = "RULE BASED UNIVERSAL"
    GPT_3_5 = "RULE BASED GPT 3.5"
    GPT_4 = "RULE BASED GPT 4"
    DEFAULT = UNIVERSAL


class TextEstimator:
    """Text token estimator class."""

    RULE_BASED = _TextEstimatorRuleBased
    DEFAULT = RULE_BASED.DEFAULT


text_estimator_map = {TextEstimator.RULE_BASED.UNIVERSAL: universal_tokens_estimator,
                      TextEstimator.RULE_BASED.GPT_3_5: openai_tokens_estimator_gpt_3_5,
                      TextEstimator.RULE_BASED.GPT_4: openai_tokens_estimator_gpt_4}


def estimate_text_tokens(text: str, estimator: TextEstimator = TextEstimator.DEFAULT) -> int:
    """
    Estimate text tokens number.

    :param text: input text
    :param estimator: estimator type
    :return: tokens number
    """
    if not isinstance(text, str):
        raise ValueError(INVALID_TEXT_MESSAGE)
    if not isinstance(estimator, (TextEstimator, _TextEstimatorRuleBased)):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text)
