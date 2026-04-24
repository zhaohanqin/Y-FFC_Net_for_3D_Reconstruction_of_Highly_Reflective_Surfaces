"""Compatibility exports for split loss modules."""

from .losses import *

__all__ = [name for name in globals() if not name.startswith('__')]
