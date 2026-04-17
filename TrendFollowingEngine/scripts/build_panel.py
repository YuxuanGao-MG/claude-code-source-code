"""Build the training panel for a window of trading days."""

from __future__ import annotations

import sys

from trend_engine.cli import main


if __name__ == "__main__":
    sys.exit(main(["build-panel"] + sys.argv[1:]))
