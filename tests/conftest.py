import sys
from pathlib import Path

# Ensure editable package root is first and preload C++ extension once per session.
ROOT = Path(__file__).resolve().parents[1] / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import az._az_core  # noqa: F401,E402 — avoid duplicate DLL init under pytest
