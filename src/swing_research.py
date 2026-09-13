"""Compatibility wrapper for the legacy swing research module.

The active NSE-wide research workflow uses ``swing_tournament.py``.
This module remains importable for older callers and delegates execution to
that current tournament runner.
"""
from __future__ import annotations

import argparse

from swing_tournament import main as tournament_main


def main(config_path: str = "config/swing.yaml") -> None:
    """Run the current NSE-wide daily swing tournament."""
    tournament_main(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/swing.yaml")
    args = parser.parse_args()
    main(args.config)
