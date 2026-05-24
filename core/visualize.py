"""
Unified visualization CLI for EQE.

Examples:
    python core/visualize.py benchmark-summary
    python core/visualize.py feature-reduction --dataset breast_cancer --explainer shap
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph_utilities import benchmark_summary, feature_reduction


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EQE visualize",
        description="Generate EQE presentation figures.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="benchmark-summary",
        choices=["benchmark-summary", "feature-reduction"],
        help="Visualization to generate.",
    )
    parser.add_argument(
        "remaining",
        nargs=argparse.REMAINDER,
        help="Options passed to the selected visualization.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "feature-reduction":
        feature_reduction.main(args.remaining)
        return
    if args.command == "benchmark-summary":
        benchmark_summary.main(args.remaining)
        return
    raise ValueError(f"Unknown visualization command: {args.command}")


if __name__ == "__main__":
    main()
