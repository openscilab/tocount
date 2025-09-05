# -*- coding: utf-8 -*-
"""Tocount parameters and constants."""

TOCOUNT_VERSION = "0.1"

INVALID_TEXT_ESTIMATOR_MESSAGE = "Invalid value. `estimator` must be an instance of TextEstimator enum."
INVALID_TEXT_MESSAGE = "Invalid value. `text` must be a string."
INVALID_LINEAR_MODEL_MESSAGE = "Invalid value. `model` must be a valid linear model name."

LINEAR_MODELS = {
    "english": {"a": 0.22027472695240083, "b": 1.3098454987590542},
    "all":     {"a": 0.24897308965467127, "b": 4.543082651055883}
}