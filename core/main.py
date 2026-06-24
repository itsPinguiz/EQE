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

import numpy as np
import pandas as pd

# Permette di eseguire anche: `python core/main.py`
sys.path.append(str(Path(__file__).parent.parent))

from core.test_framework import ExperimentOrchestrator
from core.utility.config import DEFAULT_CONFIG_PATH, load_config


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


def _normalize_explainers(explainers_cfg: Any) -> tuple[list[str] | None, dict[str, dict]]:
    if not explainers_cfg:
        return None, {}

    names: list[str] = []
    params: dict[str, dict] = {}

    for item in explainers_cfg:
        if isinstance(item, str):
            names.append(item)
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("Ogni explainer deve avere un campo 'name'.")
            names.append(name)
            explainer_params = dict(item.get("params", {}))
            explainer_params.update(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"name", "params"}
                }
            )
            params[name.lower()] = explainer_params
            continue
        raise ValueError("Explainer non validi nel config.yml.")

    return names, params


def _format_config_names(items_cfg: Any, default: list[str]) -> str:
    if not items_cfg:
        return ", ".join(default)

    names = []
    for item in items_cfg:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(item["name"])

    return ", ".join(names or default)


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


def _is_numeric_column(df: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[column])


def _format_results_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    formatted_rows = [
        [_format_markdown_value(cell) for cell in row]
        for row in df.values.tolist()
    ]
    widths = []

    for idx, header in enumerate(headers):
        row_width = max((len(row[idx]) for row in formatted_rows), default=0)
        widths.append(max(len(header), row_width))

    numeric_columns = [_is_numeric_column(df, header) for header in headers]
    header_row = "  ".join(
        _pad_markdown_cell(header, widths[idx], numeric_columns[idx])
        for idx, header in enumerate(headers)
    )
    separator_row = "  ".join("-" * width for width in widths)
    data_rows = [
        "  ".join(
            _pad_markdown_cell(cell, widths[idx], numeric_columns[idx])
            for idx, cell in enumerate(row)
        )
        for row in formatted_rows
    ]

    return "\n".join([header_row, separator_row, *data_rows])


def _pad_markdown_cell(value: str, width: int, align_right: bool) -> str:
    if align_right:
        return value.rjust(width)
    return value.ljust(width)


def _render_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    formatted_rows = [
        [_format_markdown_value(cell) for cell in row]
        for row in df.values.tolist()
    ]
    widths = []

    for idx, header in enumerate(headers):
        row_width = max((len(row[idx]) for row in formatted_rows), default=0)
        widths.append(max(len(header), row_width))

    numeric_columns = [_is_numeric_column(df, header) for header in headers]

    header_row = (
        "| "
        + " | ".join(
            _pad_markdown_cell(header, widths[idx], align_right=False)
            for idx, header in enumerate(headers)
        )
        + " |"
    )

    separator_cells = []
    for idx, is_numeric in enumerate(numeric_columns):
        dash_count = max(3, widths[idx])
        if is_numeric:
            separator_cells.append("-" * (dash_count - 1) + ":")
        else:
            separator_cells.append("-" * dash_count)
    separator_row = "| " + " | ".join(separator_cells) + " |"

    data_rows = [
        "| "
        + " | ".join(
            _pad_markdown_cell(cell, widths[idx], numeric_columns[idx])
            for idx, cell in enumerate(row)
        )
        + " |"
        for row in formatted_rows
    ]

    return "\n".join([header_row, separator_row, *data_rows])


def _aggregate_seed_results(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in df.columns or df["seed"].nunique() <= 1:
        return pd.DataFrame()

    group_columns = [
        column
        for column in ("dataset", "model", "explainer", "k_features")
        if column in df.columns
    ]
    value_columns = [
        column
        for column in df.columns
        if column not in {*group_columns, "seed", "n_explain"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not group_columns or not value_columns:
        return pd.DataFrame()

    aggregated = (
        df.groupby(group_columns)[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in aggregated.columns
    ]
    aggregated = aggregated.fillna(0.0)
    return aggregated


def _render_markdown_report(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    include_header: bool,
    aggregate_df: pd.DataFrame | None = None,
) -> str:
    table = _render_markdown_table(df)
    if not include_header:
        return table

    lines = ["# EQE Results", "", "## Experiment", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Aggregate by Seed", "", _render_markdown_table(aggregate_df)])
    lines.extend(["", "## Results", "", table])
    return "\n".join(lines)


def _build_run_matrix(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = (
        experiment.get("seeds")
        or experiment.get("random_states")
        or [experiment.get("random_state", experiment.get("seed", 42))]
    )
    if isinstance(seeds, int):
        seeds = [seeds]

    suite = experiment.get("suite")
    if suite:
        datasets = suite.get("datasets") or []
        k_values = suite.get("k_features") or suite.get("k") or []
        if not datasets or not k_values:
            raise ValueError("experiment.suite richiede 'datasets' e 'k_features'.")

        runs = []
        for dataset in datasets:
            for seed in seeds:
                # Passiamo l'intera lista di k_values per dataset, invece che fare N cicli
                runs.append({"dataset": dataset, "k_features": k_values, "seed": seed})
        return runs

    return [
        {
            "dataset": experiment.get("dataset", "breast_cancer"),
            "k_features": [experiment.get("k_features", experiment.get("k", 4))],
            "seed": seed,
        }
        for seed in seeds
    ]


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
    explainers, explainer_params = _normalize_explainers(explainers_cfg)

    runs = _build_run_matrix(experiment)
    all_results = []

    for run in runs:
        orchestrator = ExperimentOrchestrator(
            dataset_name=run["dataset"],
            k_features=run["k_features"],
            test_size=experiment.get("test_size", 0.2),
            random_state=run["seed"],
            n_explain=experiment.get("n_explain"),
            models=models,
            model_params=model_params,
            explainers=explainers,
            explainer_params=explainer_params,
            metrics=config.get("metrics"),
            n_jobs=experiment.get("n_jobs"),
            verbose=experiment.get("verbose", True),
        )

        all_results.append(orchestrator.run_experiment())

    results = pd.concat(all_results, ignore_index=True)
    aggregate_results = _aggregate_seed_results(results)

    if not aggregate_results.empty:
        # print("\nAggregate by seed")
        # print(_format_results_table(aggregate_results))
        pass
    # print("\n" + "=" * 60)
    # print("  Complexity-Calibrated Local Concordance — Results")
    # print("=" * 60)
    # print(_format_results_table(results))
    # print("=" * 60 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1 & 2: Archiviazione e impostazione del file di output
    latest_md = results_dir / "latest.md"
    archive_dir = results_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    if latest_md.exists():
        archive_path = archive_dir / f"results_{timestamp}.md"
        latest_md.rename(archive_path)
    
    output_path = latest_md
    
    results_cfg = config.get("results", {})
    include_header = results_cfg.get("include_header", True)
    
    suite = experiment.get("suite")
    metadata = {
        "n_explain": experiment.get("n_explain"),
        "models": ", ".join(models or ["xgboost", "neuralnetwork"]),
        "explainers": ", ".join(explainers or ["lime", "shap"]),
        "metrics": _format_config_names(config.get("metrics"), ["ccc_mse"]),
        "timestamp": timestamp,
        "runs": len(runs),
        "seeds": ", ".join(str(seed) for seed in sorted({run["seed"] for run in runs})),
    }
    if suite:
        metadata["datasets"] = ", ".join(suite.get("datasets", []))
        metadata["k_features"] = ", ".join(str(v) for v in suite.get("k_features", []))
    else:
        metadata["dataset"] = experiment.get("dataset", "breast_cancer")
        metadata["k_features"] = experiment.get("k_features", experiment.get("k", 4))
    
    output_path.write_text(
        _render_markdown_report(results, metadata, include_header, aggregate_results),
        encoding="utf-8",
    )
    print(f"Tabella salvata in: {output_path}")

    # 3 & 4: Generazione grafici
    print("Generazione dei grafici...")
    from core import visualize
    # Esegue benchmark-summary che usa results/latest.md di default
    visualize.main(["benchmark-summary"])
    print("Grafici generati in: results/figures/latest/")



if __name__ == "__main__":
    main()
