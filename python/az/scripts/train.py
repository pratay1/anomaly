from __future__ import annotations

import os
import sys

# Must be set before torch is imported (SGD pulls in torch._dynamo on Windows).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from az.gui.app import main


def run():
    return main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(run())
