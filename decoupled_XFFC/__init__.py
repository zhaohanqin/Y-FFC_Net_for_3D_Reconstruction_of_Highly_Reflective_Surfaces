"""Decoupled X-FFC copy. Import from .XFFC when used as a package."""

from .XFFC import *

__all__ = [name for name in globals() if not name.startswith('__')]
