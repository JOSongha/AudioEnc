"""pytest configuration for experiments/ analysis tests.

Adds repo root to sys.path so `experiments.*` imports resolve via Python 3
namespace packages (no __init__.py required in experiments/).
"""

import sys
from pathlib import Path


# Repo root = two levels up from this file (tests/experiments/conftest.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
