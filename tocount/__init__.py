# -*- coding: utf-8 -*-
"""Tocount modules."""

from .params import TOCOUNT_VERSION, TextEstimator
from .functions import estimate_text_tokens
__version__ = TOCOUNT_VERSION

__all__ = ["TextEstimator", "estimate_text_tokens"]
