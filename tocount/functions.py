# -*- coding: utf-8 -*-
"""Tocount functions."""
from enum import Enum

from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE
from .heuristic.rule_based import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4
from .linear.estimator import linear_tokens_estimator


class _TextEstimatorRuleBased(Enum):
    """Rule based text token estimator enum."""

    UNIVERSAL = "RULE BASED UNIVERSAL"
    GPT_3_5 = "RULE BASED GPT 3.5"
    GPT_4 = "RULE BASED GPT 4"
    DEFAULT = UNIVERSAL


class _TextEstimatorLinear(Enum):
    """Linear text token estimator enum."""

    TIKTOKEN_R50K_LINEAR_ALL = "TIKTOKEN R50K LINEAR ALL"
    TIKTOKEN_R50K_LINEAR_ENGLISH = "TIKTOKEN R50K LINEAR ENGLISH"
    DEFAULT = TIKTOKEN_R50K_LINEAR_ENGLISH


class TextEstimator:
    """Text token estimator class."""

    RULE_BASED = _TextEstimatorRuleBased
    LINEAR = _TextEstimatorLinear
    DEFAULT = RULE_BASED.DEFAULT


text_estimator_map = {
    TextEstimator.RULE_BASED.UNIVERSAL: universal_tokens_estimator,
    TextEstimator.RULE_BASED.GPT_3_5: openai_tokens_estimator_gpt_3_5,
    TextEstimator.RULE_BASED.GPT_4: openai_tokens_estimator_gpt_4,
    TextEstimator.LINEAR.TIKTOKEN_R50K_LINEAR_ALL: linear_tokens_estimator,
    TextEstimator.LINEAR.TIKTOKEN_R50K_LINEAR_ENGLISH: linear_tokens_estimator,
}


def estimate_text_tokens(text: str, estimator: TextEstimator = TextEstimator.DEFAULT) -> int:
    """
    Estimate text tokens number.

    :param text: input text
    :param estimator: estimator type
    :return: tokens number
    """
    if not isinstance(text, str):
        raise ValueError(INVALID_TEXT_MESSAGE)
    if not isinstance(estimator, (TextEstimator, _TextEstimatorRuleBased, _TextEstimatorLinear)):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text)