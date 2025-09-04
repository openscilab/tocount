# -*- coding: utf-8 -*-
"""Tocount modules."""

from .params import TOCOUNT_VERSION
from .heuristic.functions import estimate_text_tokens, TextEstimator
from .linear import linear_tokens_estimator
__version__ = TOCOUNT_VERSION