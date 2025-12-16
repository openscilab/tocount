# -*- coding: utf-8 -*-
"""Tocount parameters and constants."""

TOCOUNT_VERSION = "0.4"

INVALID_TEXT_ESTIMATOR_MESSAGE = "Invalid value. `estimator` must be an instance of TextEstimator enum."
INVALID_TEXT_MESSAGE = "Invalid value. `text` must be a string."

# --- Model Parameters ---
# The model coefficients ('a', 'b') are pre-scaled to operate directly on the
# raw character count. They represent the simplified result of a full
# StandardScaler pipeline, whose original parameters ('input_scaler',
# 'output_scaler') are retained below for reproducibility.

TIKTOKEN_R50K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.24220021827364216, "b": -1.52512159607669773},
        "input_scaler": {"mean": 944.83131738824351942, "scale": 5021.64260895033748966},
        "output_scaler": {"mean": 254.89806628634627828, "scale": 1446.92651162795687014}
    },
    "all": {
        "coefficient": {"a": 0.26949633800191791, "b": 17.71983908874145186},
        "input_scaler": {"mean": 807.95457802727003127, "scale": 4239.81308570276723913},
        "output_scaler": {"mean": 320.20259580719863379, "scale": 1446.23196906494854375}
    }
}

TIKTOKEN_CL100K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.21207829974544795, "b": 3.61015453257535057},
        "input_scaler": {"mean": 944.83131738824351942, "scale": 5021.64260895033748966},
        "output_scaler": {"mean": 208.97662180989638614, "scale": 1160.24193688094055688}
    },
    "all": {
        "coefficient": {"a": 0.22389270545161979, "b": 14.24559780994757219},
        "input_scaler": {"mean": 807.95457802727003127, "scale": 4239.81308570276723913},
        "output_scaler": {"mean": 221.90881687060880267, "scale": 1055.65522548552621629}
    }
}

TIKTOKEN_O200K_LINEAR_MODELS = {
    "english": {
        "coefficient": {"a": 0.20934150948723654, "b": 3.23697987353031991},
        "input_scaler": {"mean": 944.83131738824351942, "scale": 5021.64260895033748966},
        "output_scaler": {"mean": 205.52642979710270765, "scale": 1144.67974411186628458}
    },
    "all": {
        "coefficient": {"a": 0.21634871429041430, "b": 8.52848758076195246},
        "input_scaler": {"mean": 807.95457802727003127, "scale": 4239.81308570276723913},
        "output_scaler": {"mean": 194.13328834356752850, "scale": 993.46453791503881803}
    }
}
