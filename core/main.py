"""
core/main.py
============
Entry point for the EQE evaluation framework.

Run from the project root with:
    python -m core.main
or:
    python core/main.py
"""

from __future__ import annotations

import argparse

from core.test_framework import ExperimentOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following fields:

        - ``dataset``    : str  — dataset identifier (default: "breast_cancer")
        - ``k``          : int  — cognitive complexity limit K (default: 4)
        - ``explainers`` : list[str] — explainer names (default: ["lime","shap"])
        - ``n_explain``  : int  — number of test instances to explain (default: all)
        - ``test_size``  : float — fraction used for testing (default: 0.2)
        - ``seed``       : int  — random seed (default: 42)
        - ``quiet``      : bool — suppress progress output (default: False)
    """
    parser = argparse.ArgumentParser(
        prog="EQE",
        description=(
            "Evaluating the Quality of Explanations — "
            "Complexity-Calibrated Local Concordance (CCC) benchmark."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["adult", "breast_cancer", "diabetes", "german_credit"],
        help="Dataset to use for the experiment (default: breast_cancer).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        metavar="K",
        help="Cognitive complexity limit K: max features per explanation (default: 4).",
    )
    parser.add_argument(
        "--explainers",
        nargs="+",
        default=["lime", "shap"],
        choices=["lime", "shap", "maple", "l2x"],
        help="Explainer(s) to benchmark (default: lime shap).",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=None,
        metavar="N",
        help="Limit explanation generation to the first N test instances. "
             "Omit to use the full test set.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate the pipeline and run the experiment.

    Parses CLI arguments, constructs :class:`~core.test_framework.ExperimentOrchestrator`,
    calls :meth:`~core.test_framework.ExperimentOrchestrator.run_experiment`,
    and prints the results table to stdout.
    """
    args = parse_args()

    orchestrator = ExperimentOrchestrator(
        dataset_name=args.dataset,
        k_features=args.k,
        test_size=args.test_size,
        random_state=args.seed,
        n_explain=args.n_explain,
        explainers=args.explainers,
        verbose=not args.quiet,
    )

    results = orchestrator.run_experiment()

    print("\n" + "=" * 60)
    print("  Complexity-Calibrated Local Concordance — Results")
    print("=" * 60)
    print(results.to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
