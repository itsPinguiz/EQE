"""
Generate presentation-ready summary charts from an EQE markdown results table.

Example:
    uv run python -m core.graph_utilities.benchmark_summary \
        --results results/SOTA/results_20260518_141724.md
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS_PATH = Path("results/archive/results_20260518_141724.md")
DEFAULT_OUTPUT_DIR = Path("results/figures/presentation")
EXPLAINER_COLORS = {
    "shap": "#2f80ed",
    "maple": "#1f9d55",
    "lime": "#d95f02",
}
RESULT_SECTION_TITLES = ("## Results", "## Aggregate by Seed")
METRIC_ALIASES = {
    "accuracy": ("accuracy", "accuracy_mean"),
    "ccc_mse": ("ccc_mse", "ccc_mse_mean"),
    "random_k_mse": ("random_k_mse", "random_k_mse_mean"),
    "sufficiency_mse": ("sufficiency_mse", "sufficiency_mse_mean"),
    "comprehensiveness_abs_drop": (
        "comprehensiveness_abs_drop",
        "comprehensiveness_abs_drop_mean",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EQE benchmark summary visualizer",
        description="Create summary charts from a saved EQE markdown results table.",
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def load_results_table(path: str | Path) -> pd.DataFrame:
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    lines = results_path.read_text(encoding="utf-8").splitlines()
    for section_title in RESULT_SECTION_TITLES:
        df = _read_markdown_table(lines, section_title)
        if df is None:
            continue
        normalized = _normalize_results_columns(df)
        if {"ccc_mse", "random_k_mse"}.issubset(normalized.columns):
            if section_title != "## Results":
                warnings.warn(
                    f"Using {section_title} from {results_path.name} because the primary results table is incomplete.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return _coerce_results_types(normalized)

    raise ValueError(f"No usable markdown results table found in: {results_path}")


def _read_markdown_table(lines: list[str], section_title: str) -> pd.DataFrame | None:
    table_lines: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == section_title:
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break

    if len(table_lines) < 3:
        return None

    header = _split_markdown_row(table_lines[0])
    rows = [
        _split_markdown_row(line)
        for line in table_lines[2:]
        if not set(line.replace("|", "").strip()) <= {"-", ":", " "}
    ]
    return pd.DataFrame(rows, columns=header)


def _normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical in normalized.columns:
            continue
        for alias in aliases:
            if alias in normalized.columns:
                normalized = normalized.rename(columns={alias: canonical})
                break
    return normalized


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _coerce_results_types(df: pd.DataFrame) -> pd.DataFrame:
    typed = df.copy()
    integer_columns = ["k_features", "n_explain"]
    float_columns = ["accuracy", "ccc_mse", "random_k_mse", "sufficiency_mse", "comprehensiveness_abs_drop"]

    for column in integer_columns:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="raise").astype(int)
    for column in float_columns:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="raise")

    return typed


def generate_summary_charts(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (_plot_ccc_mse_by_k, {"dataset", "model", "explainer", "k_features", "ccc_mse"}),
        (_plot_explainer_ranking, {"dataset", "explainer", "ccc_mse"}),
        (_plot_random_k_advantage, {"dataset", "explainer", "k_features", "ccc_mse", "random_k_mse"}),
    ]
    paths: list[Path] = []
    for plot_func, required_columns in plot_specs:
        missing = required_columns - set(df.columns)
        if missing:
            warnings.warn(
                f"Skipping {plot_func.__name__}: missing columns {sorted(missing)}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        paths.append(plot_func(df, output_path))
    return paths


def _plot_ccc_mse_by_k(df: pd.DataFrame, output_dir: Path) -> Path:
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(13.5, 8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    fig.set_constrained_layout_pads(h_pad=0.06, w_pad=0.04, hspace=0.08, wspace=0.08)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]
            for explainer in sorted(subset["explainer"].unique()):
                explainer_df = subset[subset["explainer"] == explainer].sort_values("k_features")
                ax.plot(
                    explainer_df["k_features"],
                    explainer_df["ccc_mse"],
                    marker="o",
                    linewidth=2.2,
                    color=EXPLAINER_COLORS.get(explainer, "#4b5563"),
                    label=explainer.upper(),
                )
            ax.set_title(f"{dataset} | {model}")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("CCC-MSE (log scale)")
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Top-K feature budget")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Faithfulness improves as the feature budget increases", fontweight="bold")

    path = output_dir / "ccc_mse_by_k.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_explainer_ranking(df: pd.DataFrame, output_dir: Path) -> Path:
    ranking = (
        df.groupby(["dataset", "explainer"], as_index=False)["ccc_mse"]
        .mean()
        .sort_values(["dataset", "ccc_mse"])
    )
    datasets = sorted(ranking["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.8), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([axes])

    for ax, dataset in zip(axes, datasets, strict=True):
        subset = ranking[ranking["dataset"] == dataset]
        colors = [EXPLAINER_COLORS.get(name, "#4b5563") for name in subset["explainer"]]
        ax.bar(
            subset["explainer"].str.upper(),
            subset["ccc_mse"],
            color=colors,
            edgecolor="#2b2f36",
        )
        ax.set_title(dataset)
        ax.set_yscale("log")
        ax.set_ylabel("Mean CCC-MSE (log scale)")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.25)
        for idx, value in enumerate(subset["ccc_mse"]):
            ax.text(idx, value * 1.08, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Explainer ranking by compact faithfulness", fontweight="bold")
    path = output_dir / "explainer_ranking_mean_ccc_mse.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_random_k_advantage(df: pd.DataFrame, output_dir: Path) -> Path:
    max_k = int(df["k_features"].max())
    subset = df[df["k_features"] == max_k].copy()
    subset["error_reduction"] = 100 * (1 - subset["ccc_mse"] / subset["random_k_mse"])
    advantage = (
        subset.groupby(["dataset", "explainer"], as_index=False)["error_reduction"]
        .mean()
        .sort_values(["dataset", "error_reduction"], ascending=[True, False])
    )
    datasets = sorted(advantage["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.8), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([axes])

    for ax, dataset in zip(axes, datasets, strict=True):
        dataset_df = advantage[advantage["dataset"] == dataset]
        colors = [EXPLAINER_COLORS.get(name, "#4b5563") for name in dataset_df["explainer"]]
        ax.bar(
            dataset_df["explainer"].str.upper(),
            dataset_df["error_reduction"],
            color=colors,
            edgecolor="#2b2f36",
        )
        ax.set_ylim(0, 105)
        ax.set_title(dataset)
        ax.set_ylabel("Error reduction vs random-K (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for idx, value in enumerate(dataset_df["error_reduction"]):
            ax.text(idx, min(value + 2, 101), f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle(f"Top-K beats random feature selection at K={max_k}", fontweight="bold")
    path = output_dir / f"top_k_vs_random_k_k{max_k}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_results_table(args.results)
    paths = generate_summary_charts(df, args.output_dir)
    for path in paths:
        print(f"Saved benchmark summary figure: {path}")


if __name__ == "__main__":
    main()
