# -*- coding: utf-8 -*-
"""Tocount parameters and constants."""

TOCOUNT_VERSION = "0.2"

INVALID_TEXT_ESTIMATOR_MESSAGE = "Invalid value. `estimator` must be an instance of TextEstimator enum."
INVALID_TEXT_MESSAGE = "Invalid value. `text` must be a string."

# --- Model Parameters ---
# The model coefficients ('a', 'b') are pre-scaled to operate directly on the
# raw character count. They represent the simplified result of a full
# StandardScaler pipeline, whose original parameters ('input_scaler',
# 'output_scaler') are retained below for reproducibility.

TIKTOKEN_R50K_LINEAR_MODELS = {
    "english": {
        "model": {"a": 0.22027472695240083, "b": 1.3098454987590542},
        "input_scaler": {"mean": 847.1859533518088, "scale": 4824.545962963612},
        "output_scaler": {"mean": 191.91873679585714, "scale": 1122.038549166423}
    },
    "all": {
        "model": {"a": 0.24897308965467127, "b": 4.543082651055883},
        "input_scaler": {"mean": 863.9105273550211, "scale": 4579.146073191748},
        "output_scaler": {"mean": 250.55580827419274, "scale": 1317.8399144012787}
    }
}

TIKTOKEN_CL100K_LINEAR_MODELS = {
    "english": {
        "model": {"a": 0.2063277459592275, "b": 1.3158237765272283},
        "input_scaler": {"mean": 928.0135145581235, "scale": 4839.455147131051},
        "output_scaler": {"mean": 198.34363306972855, "scale": 1087.618915250561}
    },
    "all": {
        "model": {"a": 0.22359382657517404, "b": 4.810584338754186},
        "input_scaler": {"mean": 874.1646054463054, "scale": 4486.7423868301485},
        "output_scaler": {"mean": 213.8142892920311, "scale": 1078.2629716972262}
    }
}
