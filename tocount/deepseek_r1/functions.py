# -*- coding: utf-8 -*-
"""Deepseek R1 functions."""
from ..params import DEEPSEEK_R1_LINEAR_MODELS


def _linear_estimator(text: str, model: str = "english") -> int:
    """
    Perform linear estimation.

    :param text: input text
    :param model: model name
    """
    params = DEEPSEEK_R1_LINEAR_MODELS[model]["coefficient"]
    a = params["a"]
    b = params["b"]
    char_count = len(text)
    estimate = a * char_count + b
    return int(round(estimate))


def linear_tokens_estimator_english(text: str) -> int:
    """
    Linear token estimator for the English text.

    :param text: input text
    """
    return _linear_estimator(text, "english")


def linear_tokens_estimator_all(text: str) -> int:
    """
    Linear token estimator for all languages.

    :param text: input text
    """
    return _linear_estimator(text, "all")