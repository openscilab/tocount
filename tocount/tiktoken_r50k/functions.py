# -*- coding: utf-8 -*-
"""TikToken R50K functions."""
from ..params import TIKTOKEN_R50K_LINEAR_MODELS

def _linear_estimator(text: str, model: str = "English") -> int:
    """Perform linear estimation."""
    params = TIKTOKEN_R50K_LINEAR_MODELS[model]
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
