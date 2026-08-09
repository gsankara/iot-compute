"""Make the examples runnable straight from a checkout.

``pip install -e .`` is the tidy way, but an example that only works after an
install is an example most people never run.  Importing this module first puts
the repository root on the path when ``iotc`` is not already importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # already installed, or already on the path
    import iotc  # noqa: F401
except ImportError:  # pragma: no cover - convenience path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import iotc  # noqa: F401
