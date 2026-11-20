"""Pytest collection-time setup.

Our own packages (``hiexp``, ``hiexp_prompt``, ``ace_react``) ship via
``pyproject.toml`` and are picked up from the editable install — no
``sys.path`` manipulation needed for them.

The one exception is upstream ACE under ``third_party/ace``: it has no
``setup.py`` / ``pyproject.toml`` so ``import ace`` only works when that
directory is on ``sys.path``. Tests that touch ACE's controller flow
need this; tests that don't are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY_ACE = str(_PROJECT_ROOT / "third_party" / "ace")
if _THIRD_PARTY_ACE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_ACE)
