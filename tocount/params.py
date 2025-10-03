# -*- coding: utf-8 -*-
"""Tocount parameters and constants."""

TOCOUNT_VERSION = "0.1"

INVALID_TEXT_ESTIMATOR_MESSAGE = "Invalid value. `estimator` must be an instance of TextEstimator enum."
INVALID_TEXT_MESSAGE = "Invalid value. `text` must be a string."

TIKTOKEN_R50K_LINEAR_MODELS = {
    "english": {"a": 0.22027472695240083, "b": 1.3098454987590542},
    "all":     {"a": 0.24897308965467127, "b": 4.5430826510558830}
}

TIKTOKEN_CL100K_LINEAR_MODELS = {
    "english": {"a": 0.20632774595922751, "b": 1.31582377652722826},
    "all":     {"a": 0.22359382657517404, "b": 4.81058433875418601}
}