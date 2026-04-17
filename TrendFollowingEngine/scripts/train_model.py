"""Train the ensemble model from a prebuilt panel parquet."""

from __future__ import annotations

import sys

from trend_engine.cli import main


if __name__ == "__main__":
    sys.exit(main(["train"] + sys.argv[1:]))
