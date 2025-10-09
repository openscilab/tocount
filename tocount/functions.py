# -*- coding: utf-8 -*-
"""Tocount functions."""
from enum import Enum

from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE
from .rule_based.functions import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4
from .tiktoken_r50k.functions import linear_tokens_estimator_all as r50k_linear_all
from .tiktoken_r50k.functions import linear_tokens_estimator_english as r50k_linear_english
from .tiktoken_cl100k.functions import linear_tokens_estimator_all as cl100k_linear_all
from .tiktoken_cl100k.functions import linear_tokens_estimator_english as cl100k_linear_english
from .tiktoken_o200k.functions import linear_tokens_estimator_all as o200k_linear_all
from .tiktoken_o200k.functions import linear_tokens_estimator_english as o200k_linear_english
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


class _TextEstimatorTikTokenCL100K(Enum):
    """TikToken cl100k text token estimator enum."""

    LINEAR_ALL = "TIKTOKEN_CL100K_LINEAR_ALL"
    LINEAR_ENGLISH = "TIKTOKEN_CL100K_LINEAR_ENGLISH"
    DEFAULT = LINEAR_ENGLISH


class _TextEstimatorTikTokenO200K(Enum):
    """TikToken o200k text token estimator enum."""

    LINEAR_ALL = "TIKTOKEN_O200K_LINEAR_ALL"
    LINEAR_ENGLISH = "TIKTOKEN_O200K_LINEAR_ENGLISH"
    DEFAULT = LINEAR_ENGLISH


class TextEstimator:
    """Text token estimator class."""

    RULE_BASED = _TextEstimatorRuleBased
    TIKTOKEN_R50K = _TextEstimatorTikTokenR50K
    TIKTOKEN_CL100K = _TextEstimatorTikTokenCL100K
    TIKTOKEN_O200K = _TextEstimatorTikTokenO200K
    DEFAULT = RULE_BASED.DEFAULT


text_estimator_map = {
    TextEstimator.RULE_BASED.UNIVERSAL: universal_tokens_estimator,
    TextEstimator.RULE_BASED.GPT_3_5: openai_tokens_estimator_gpt_3_5,
    TextEstimator.RULE_BASED.GPT_4: openai_tokens_estimator_gpt_4,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ALL: r50k_linear_all,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ENGLISH: r50k_linear_english,
    TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL: cl100k_linear_all,
    TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH: cl100k_linear_english,
    TextEstimator.TIKTOKEN_O200K.LINEAR_ALL: o200k_linear_all,
    TextEstimator.TIKTOKEN_O200K.LINEAR_ENGLISH: o200k_linear_english,
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
    if not isinstance(estimator, (
        TextEstimator, _TextEstimatorRuleBased, _TextEstimatorTikTokenR50K, _TextEstimatorTikTokenCL100K, _TextEstimatorTikTokenO200K)):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text)
