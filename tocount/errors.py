# -*- coding: utf-8 -*-
"""Tocount errors."""

class TocountError(Exception):
    """Base exception for all Tocount errors."""

    pass

class TocountValidationError(TocountError, ValueError):
    """Base class for validation errors in Tocount."""

    pass
