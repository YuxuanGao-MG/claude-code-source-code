"""Convenience wrapper around `trend-engine scan --predict`.

Run this from cron after the US close. Prints a CSV of today's flagged
candidates to stdout.
"""

from __future__ import annotations

import sys

from trend_engine.cli import main


if __name__ == "__main__":
    sys.exit(main(["scan", "--predict"] + sys.argv[1:]))
