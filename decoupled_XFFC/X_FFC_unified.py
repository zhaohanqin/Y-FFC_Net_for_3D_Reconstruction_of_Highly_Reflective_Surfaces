"""Decoupled copy of the original X_FFC_unified.py entrypoint.

This file is stored under decoupled_XFFC/ and does not replace the original
repository-root X_FFC_unified.py. Run it from this directory:

    python X_FFC_unified.py test
"""

if __package__:
    from .XFFC import *
    from .XFFC.cli import display_basic_model_info, display_model_summary, main
else:
    from XFFC import *
    from XFFC.cli import display_basic_model_info, display_model_summary, main

__all__ = [name for name in globals() if not name.startswith('__')]


if __name__ == "__main__":
    main()
