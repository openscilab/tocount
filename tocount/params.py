# -*- coding: utf-8 -*-
"""Tocount parameters and constants."""

TOCOUNT_VERSION = "0.3"

INVALID_TEXT_ESTIMATOR_MESSAGE = "Invalid value. `estimator` must be an instance of TextEstimator enum."
INVALID_TEXT_MESSAGE = "Invalid value. `text` must be a string."

# --- Model Parameters ---
# The model coefficients ('a', 'b') are pre-scaled to operate directly on the
# raw character count. They represent the simplified result of a full
# StandardScaler pipeline, whose original parameters ('input_scaler',
# 'output_scaler') are retained below for reproducibility.

TIKTOKEN_R50K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.22027472695240083, "b": 1.30984549875905421},
        "input_scaler": {"mean": 847.18595335180884, "scale": 4824.54596296361160},
        "output_scaler": {"mean": 191.91873679585714, "scale": 1122.03854916642285}
    },
    "all": {
        "coefficient": {"a": 0.24897308965467127, "b": 4.54308265105588305},
        "input_scaler": {"mean": 863.91052735502114, "scale": 4579.14607319174774},
        "output_scaler": {"mean": 250.55580827419274, "scale": 1317.83991440127875}
    }
}

TIKTOKEN_CL100K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.20632774595922751, "b": 1.31582377652722826},
        "input_scaler": {"mean": 928.01351455812346, "scale": 4839.45514713105058},
        "output_scaler": {"mean": 198.34363306972855, "scale": 1087.61891525056103}
    },
    "all": {
        "coefficient": {"a": 0.22359382657517404, "b": 4.81058433875418601},
        "input_scaler": {"mean": 874.16460544630535, "scale": 4486.74238683014846},
        "output_scaler": {"mean": 213.81428929203110, "scale": 1078.26297169722625}
    }
}

TIKTOKEN_O200K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.20354485735834993, "b": 2.08764234347103361},
        "input_scaler": {"mean": 923.75157809972700, "scale": 4843.14030162006293},
        "output_scaler": {"mean": 194.41034579748791, "scale": 1073.00614112992844}
    },
    "all": {
        "coefficient": {"a": 0.21511955690162138, "b": 1.71656955330649552},
        "input_scaler": {"mean": 859.61614585211419, "scale": 4397.61706792694440},
        "output_scaler": {"mean": 191.96748588283666, "scale": 1006.41246761102514}
    }
}
