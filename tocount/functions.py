# -*- coding: utf-8 -*-
"""Tocount functions."""
from enum import Enum

from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE
from .rule_based.functions import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4
from .tiktoken_r50.functions import linear_tokens_estimator_english as tiktoken_r50_linear_tokens_estimator_english
from .tiktoken_r50.functions import linear_tokens_estimator_all as tiktoken_r50_linear_tokens_estimator_all


class _TextEstimatorRuleBased(Enum):
    """Rule based text token estimator enum."""

    UNIVERSAL = "RULE BASED UNIVERSAL"
    GPT_3_5 = "RULE BASED GPT 3.5"
    GPT_4 = "RULE BASED GPT 4"
    DEFAULT = UNIVERSAL


class _TextEstimatorTikTokenR50K(Enum):
    """TikToken R50K text token estimator enum."""

    LINEAR_ALL = "TIKTOKEN_R50K_LINEAR_ALL"
    LINEAR_ENGLISH = "TIKTOKEN_R50K_LINEAR_ENGLISH"
    DEFAULT = LINEAR_ENGLISH


class TextEstimator:
    """Text token estimator class."""

    RULE_BASED = _TextEstimatorRuleBased
    TIKTOKEN_R50K = _TextEstimatorTikTokenR50K
    DEFAULT = RULE_BASED.DEFAULT


text_estimator_map = {
    TextEstimator.RULE_BASED.UNIVERSAL: universal_tokens_estimator,
    TextEstimator.RULE_BASED.GPT_3_5: openai_tokens_estimator_gpt_3_5,
    TextEstimator.RULE_BASED.GPT_4: openai_tokens_estimator_gpt_4,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ALL: tiktoken_r50_linear_tokens_estimator_all,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ENGLISH: tiktoken_r50_linear_tokens_estimator_english,
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
    if not isinstance(estimator, (TextEstimator, _TextEstimatorRuleBased, _TextEstimatorTikTokenR50K)):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text)
