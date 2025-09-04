from ..params import LINEAR_MODELS
"""Linear token estimator."""

def linear_tokens_estimator(text: str, model: str = "English") -> int:
    """
    linear token estimator:
    
    :param text: input text
    :param model: model name (case-insensitive), must exist in params.LINEAR_MODELS
    :return: token estimate
    """
    key = (model).strip().lower()
    if key not in LINEAR_MODELS:
        raise ValueError(f"Unknown linear model '{model}'.")

    p = LINEAR_MODELS[key]
    a = float(p["a"])
    b = float(p["b"])

    char_count = len(str(text))
    estimate = a * char_count + b
    return int(round(estimate))