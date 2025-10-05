# -*- coding: utf-8 -*-
"""TikToken CL100K functions."""
from ..params import TIKTOKEN_CL100K_LINEAR_MODELS


def _linear_estimator(text: str, model: str = "english") -> int:
    """
    Perform linear estimation.

    :param text: input text
    :param model: model name
    """
    params = TIKTOKEN_CL100K_LINEAR_MODELS[model]

    model_params = params["model"]
    input_scaler_params = params["input_scaler"]
    output_scaler_params = params["output_scaler"]

    a = model_params["a"]
    b = model_params["b"]

    x_mean = input_scaler_params["mean"]
    x_scale = input_scaler_params["scale"]

    y_mean = output_scaler_params["mean"]
    y_scale = output_scaler_params["scale"]

    char_count = len(text)
    scaled_char_count = (char_count - x_mean) / x_scale
    scaled_estimate = a * scaled_char_count + b
    estimate = (scaled_estimate * y_scale) + y_mean
    return int(round(max(0, estimate)))


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
