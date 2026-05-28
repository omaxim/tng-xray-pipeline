"""
Import helper for illustris_python.

The real illustris_python lives one level inside its git clone
(clone_root/illustris_python/__init__.py), so we add the clone root to
sys.path unconditionally — before any import attempt — so the correct
version always wins over any ambient stub or namespace package.

Resolution order:
  1. ILLUSTRIS_PYTHON_PATH environment variable (clone root)
  2. Hard-coded fallback for MPIA Vera cluster

Import this once at package initialisation; all other modules use::

    from ._il import il
"""

import os
import sys

_PATHS = [
    os.environ.get('ILLUSTRIS_PYTHON_PATH', ''),
    '/vera/u/maoweyssi/Code/VeraWorkspace/illustris_python',
]

# Prepend unconditionally so the correct clone wins over any ambient install
for _p in reversed(_PATHS):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import illustris_python as il

if not hasattr(il, 'groupcat'):
    raise ImportError(
        f'illustris_python at {il.__file__!r} has no groupcat. '
        f'Set ILLUSTRIS_PYTHON_PATH to the illustris_python clone root.'
    )

__all__ = ['il']
