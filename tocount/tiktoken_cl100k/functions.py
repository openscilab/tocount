# -*- coding: utf-8 -*-
"""TikToken cl100k functions."""
from ..params import TIKTOKEN_CL100K_LINEAR_MODELS

def _linear_estimator(text: str, model: str = "english") -> int:
    """Perform linear estimation."""
    params = TIKTOKEN_CL100K_LINEAR_MODELS[model]
    a = params["a"]
    b = params["b"]
    char_count = len(text)
    estimate = a * char_count + b
    return int(round(estimate))

def linear_tokens_estimator_english(text: str) -> int:
    """Linear token estimator for the 'English' model."""
    return _linear_estimator(text, "english")

def linear_tokens_estimator_all(text: str) -> int:
    """Linear token estimator for the 'all' model."""
    return _linear_estimator(text, "all")
