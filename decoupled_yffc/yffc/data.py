"""Compatibility exports for split data modules."""

from .data import *

__all__ = [name for name in globals() if not name.startswith('__')]
