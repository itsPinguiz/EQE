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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Permette di eseguire anche: `python core/main.py`
sys.path.append(str(Path(__file__).parent.parent))

from core.config import DEFAULT_CONFIG_PATH, load_config
from core.test_framework import ExperimentOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following fields:

        - ``config`` : str — path to the YAML config file (default: config.yml)
    """
    parser = argparse.ArgumentParser(
        prog="EQE",
        description=(
            "Evaluating the Quality of Explanations — "
            "Complexity-Calibrated Local Concordance (CCC) benchmark."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file (default: config.yml).",
    )
    return parser.parse_args()


def _normalize_models(models_cfg: Any) -> tuple[list[str] | None, dict[str, dict]]:
    if not models_cfg:
        return None, {}

    names = []
    params: dict[str, dict] = {}
    for item in models_cfg:
        if isinstance(item, str):
            names.append(item)
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("Ogni modello deve avere un campo 'name'.")
            names.append(name)
            params[name.lower()] = item.get("params", {})
            continue
        raise ValueError("Modelli non validi nel config.yml.")

    return names, params


def _normalize_explainers(explainers_cfg: Any) -> tuple[list[str] | None, int | None, str]:
    if not explainers_cfg:
        return None, 100, "sample"

    names: list[str] = []
    background_size = 100
    background_strategy = "sample"

    for item in explainers_cfg:
        if isinstance(item, str):
            names.append(item)
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("Ogni explainer deve avere un campo 'name'.")
            names.append(name)
            if name.lower() == "shap":
                background_size = item.get("background_size", background_size)
                background_strategy = item.get("background_strategy", background_strategy)
            continue
        raise ValueError("Explainer non validi nel config.yml.")

    return names, background_size, background_strategy


def _render_markdown_table(df) -> str:
    headers = list(df.columns)
    rows = df.values.tolist()

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(map(str, row)) + " |" for row in rows]

    return "\n".join([header_row, separator_row, *data_rows])


def _render_markdown_report(df, metadata: dict[str, Any], include_header: bool) -> str:
    table = _render_markdown_table(df)
    if not include_header:
        return table

    lines = ["# EQE Results", "", "## Experiment", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Results", "", table])
    return "\n".join(lines)


def main() -> None:
    """Instantiate the pipeline and run the experiment.

    Parses CLI arguments, constructs :class:`~core.test_framework.ExperimentOrchestrator`,
    calls :meth:`~core.test_framework.ExperimentOrchestrator.run_experiment`,
    and prints the results table to stdout.
    """
    args = parse_args()
    config = load_config(args.config)

    experiment = config.get("experiment", {})
    models_cfg = config.get("models", None)
    explainers_cfg = config.get("explainers", None)

    models, model_params = _normalize_models(models_cfg)
    explainers, shap_background_size, shap_background_strategy = _normalize_explainers(
        explainers_cfg
    )

    orchestrator = ExperimentOrchestrator(
        dataset_name=experiment.get("dataset", "breast_cancer"),
        k_features=experiment.get("k_features", experiment.get("k", 4)),
        test_size=experiment.get("test_size", 0.2),
        random_state=experiment.get("random_state", experiment.get("seed", 42)),
        n_explain=experiment.get("n_explain"),
        models=models,
        model_params=model_params,
        explainers=explainers,
        metrics=config.get("metrics"),
        shap_background_size=shap_background_size,
        shap_background_strategy=shap_background_strategy,
        verbose=experiment.get("verbose", True),
    )

    results = orchestrator.run_experiment()

    print("\n" + "=" * 60)
    print("  Complexity-Calibrated Local Concordance — Results")
    print("=" * 60)
    print(results.to_string(index=False))
    print("=" * 60 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_cfg = config.get("results", {})
    prefix = results_cfg.get("filename_prefix", "results")
    include_header = results_cfg.get("include_header", True)

    output_path = results_dir / f"{prefix}_{timestamp}.md"
    metadata = {
        "dataset": experiment.get("dataset", "breast_cancer"),
        "k_features": experiment.get("k_features", experiment.get("k", 4)),
        "n_explain": experiment.get("n_explain"),
        "models": ", ".join(models or ["xgboost", "neuralnetwork"]),
        "explainers": ", ".join(explainers or ["lime", "shap"]),
        "metrics": ", ".join(config.get("metrics", ["ccc_mse"])),
        "timestamp": timestamp,
    }
    output_path.write_text(
        _render_markdown_report(results, metadata, include_header),
        encoding="utf-8",
    )
    print(f"Tabella salvata in: {output_path}")


if __name__ == "__main__":
    main()
