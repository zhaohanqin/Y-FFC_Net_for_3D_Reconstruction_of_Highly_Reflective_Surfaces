"""Decoupled Y-FFC copy. Import from .yffc when used as a package."""

from .yffc import *

__all__ = [name for name in globals() if not name.startswith('__')]
