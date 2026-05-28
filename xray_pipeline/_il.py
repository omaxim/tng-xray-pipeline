"""
Import helper for illustris_python.

Resolution order:
  1. Regular import (works if illustris_python is installed or on sys.path)
  2. ILLUSTRIS_PYTHON_PATH environment variable
  3. Hard-coded fallback for MPIA Vera cluster

Import this once at package initialisation; all other modules use::

    from ._il import il
"""

import os
import sys

_FALLBACK_PATHS = [
    os.environ.get('ILLUSTRIS_PYTHON_PATH', ''),
    '/vera/u/maoweyssi/Code/VeraWorkspace/illustris_python',
]

try:
    import illustris_python as il
except ImportError:
    for _p in _FALLBACK_PATHS:
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    import illustris_python as il   # raises clearly if still not found

__all__ = ['il']
