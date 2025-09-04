# -*- coding: utf-8 -*-
"""Heuristic token estimators."""

from .rule_based import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4
from .keywords import PROGRAMMING_LANGUAGES, PROGRAMMING_LANGUAGES_KEYWORDS
from .functions import estimate_text_tokens, TextEstimator