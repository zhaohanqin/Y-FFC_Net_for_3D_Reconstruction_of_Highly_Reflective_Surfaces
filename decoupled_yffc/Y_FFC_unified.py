"""Decoupled copy of the original Y_FFC_unified.py entrypoint.

This file is stored under decoupled_yffc/ and does not replace the original
repository-root Y_FFC_unified.py.
"""

try:
    from yffc import *
    from yffc.cli import display_basic_model_info, display_model_summary, main
except ModuleNotFoundError:
    from .yffc import *
    from .yffc.cli import display_basic_model_info, display_model_summary, main

__all__ = [name for name in globals() if not name.startswith('__')]


if __name__ == "__main__":
    main()
