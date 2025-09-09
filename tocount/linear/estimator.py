# -*- coding: utf-8 -*-
"""Tocount functions."""
from ..params import LINEAR_MODELS, INVALID_LINEAR_MODEL_MESSAGE

def linear_tokens_estimator(text: str, model: str = "English") -> int:
    """
    Linear token estimator.

    :param text: Input text.
    :param model: Model name (case-insensitive), must exist in params.LINEAR_MODELS.
    :return: Token estimate as an integer.
    """
    key = (model).strip().lower()

    p = LINEAR_MODELS[key]
    a = float(p["a"])
    b = float(p["b"])

    char_count = len(str(text))
    estimate = a * char_count + b
    return int(round(estimate))