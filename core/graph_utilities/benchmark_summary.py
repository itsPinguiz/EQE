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

DEFAULT_RESULTS_PATH = Path("results/latest.md")
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
    "ccc_mse_normalized": ("ccc_mse_normalized", "ccc_mse_normalized_mean"),
    "top_k_degradation_ratio": ("top_k_degradation_ratio", "top_k_degradation_ratio_mean"),
    "full_mse": ("full_mse", "full_mse_mean"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EQE benchmark summary visualizer",
        description="Create summary charts from a saved EQE markdown results table.",
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _default_output_dir_for_results(results_path: Path) -> Path:
    if results_path.name == "latest.md":
        return Path("results/figures/latest")
    return Path("results/figures") / results_path.stem


def load_results_table(path: str | Path) -> pd.DataFrame:
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    lines = results_path.read_text(encoding="utf-8").splitlines()
    # Prefer the "## Results" section (primary data)
    for section_title in RESULT_SECTION_TITLES:
        df = _read_markdown_table(lines, section_title)
        if df is None:
            continue
        normalized = _normalize_results_columns(df)
        # Check for required columns - use ccc_mse as minimum requirement
        if "ccc_mse" in normalized.columns:
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
    # Metrics that may contain NaN values (e.g., top_k_degradation_ratio for SHAP)
    # Use errors="coerce" to handle NaN gracefully
    float_columns = [
        "accuracy", "ccc_mse", "random_k_mse", "sufficiency_mse",
        "comprehensiveness_abs_drop", "ccc_mse_normalized", "top_k_degradation_ratio", "full_mse"
    ]

    for column in integer_columns:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="raise").astype(int)
    for column in float_columns:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="coerce")

    return typed


def generate_summary_charts(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (_plot_ccc_mse_by_k, {"dataset", "model", "explainer", "k_features", "ccc_mse"}),
        (_plot_explainer_ranking, {"dataset", "explainer", "ccc_mse"}),
        (_plot_random_k_advantage, {"dataset", "explainer", "k_features", "ccc_mse", "random_k_mse"}),
        (_plot_ccc_vs_full, {"dataset", "model", "explainer", "k_features", "ccc_mse", "full_mse"}),
        (_plot_seed_variability, {"dataset", "model", "explainer", "seed", "k_features", "ccc_mse"}),
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
                # Aggregate multiple seeds by mean for smooth continuous curves
                if "seed" in explainer_df.columns and explainer_df["seed"].nunique() > 1:
                    agg_dict = {"ccc_mse": "mean"}
                    if "random_k_mse" in explainer_df.columns:
                        agg_dict["random_k_mse"] = "mean"
                    explainer_df = explainer_df.groupby("k_features", as_index=False).agg(agg_dict)
                k_values = explainer_df["k_features"].values.astype(float)
                ccc_values = explainer_df["ccc_mse"].values
                
                # Create smooth interpolated curve using numpy
                k_smooth = np.linspace(k_values.min(), k_values.max(), 200)
                # Linear interpolation for smooth curve
                ccc_smooth = np.interp(k_smooth, k_values, ccc_values)
                
                ax.plot(
                    k_smooth,
                    ccc_smooth,
                    linewidth=2.5,
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
    # Aggregate by seed first if multiple seeds exist, then compute mean per dataset/explainer
    agg_columns = ["ccc_mse"]
    if "seed" in df.columns and df["seed"].nunique() > 1:
        # Group by dataset, model, explainer, k_features to get mean across seeds
        group_cols = ["dataset", "explainer", "k_features"]
        if "model" in df.columns:
            group_cols.insert(1, "model")
        df = df.groupby(group_cols, as_index=False)[agg_columns].mean()
    
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
    # Aggregate by seed first if multiple seeds exist
    if "seed" in df.columns and df["seed"].nunique() > 1:
        group_cols = ["dataset", "explainer", "k_features"]
        if "model" in df.columns:
            group_cols.insert(1, "model")
        agg_dict = {"ccc_mse": "mean"}
        if "random_k_mse" in df.columns:
            agg_dict["random_k_mse"] = "mean"
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
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


def _plot_ccc_vs_full(df: pd.DataFrame, output_dir: Path) -> Path:
    """Scatter plot showing degradation from top-K truncation.

    X-axis: full_mse (baseline fidelity). Y-axis: ccc_mse (truncated fidelity).
    Points near the diagonal indicate robust explanations.
    Note: SHAP points collapse to x≈0 due to local accuracy guarantee.
    """
    # Aggregate by seed first if necessary
    if "seed" in df.columns and df["seed"].nunique() > 1:
        group_cols = ["dataset", "model", "explainer", "k_features"]
        agg_dict = {"ccc_mse": "mean"}
        if "full_mse" in df.columns:
            agg_dict["full_mse"] = "mean"
        if "top_k_degradation_ratio" in df.columns:
            agg_dict["top_k_degradation_ratio"] = "mean"
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Filter for max K to show worst-case degradation
    max_k = int(df["k_features"].max())
    subset = df[df["k_features"] == max_k].copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each explainer with distinct colors
    for explainer in sorted(subset["explainer"].unique()):
        expl_df = subset[subset["explainer"] == explainer]
        color = EXPLAINER_COLORS.get(explainer, "#4b5563")

        # Skip points where full_mse is 0 (SHAP) to avoid log issues
        nonzero_mask = expl_df["full_mse"] > 1e-10

        ax.scatter(
            expl_df.loc[nonzero_mask, "full_mse"],
            expl_df.loc[nonzero_mask, "ccc_mse"],
            c=color,
            label=explainer.upper(),
            s=80,
            alpha=0.8,
            edgecolor="#2b2f36",
        )

        # Mark SHAP separately (all points at x≈0)
        if explainer == "shap" or not nonzero_mask.any():
            shap_df = expl_df[~nonzero_mask]
            if len(shap_df) > 0:
                ax.scatter(
                    shap_df["ccc_mse"] * 0 + 1e-12,  # Offset for visibility
                    shap_df["ccc_mse"],
                    c=color,
                    label=f"{explainer.upper()} (exact)",
                    s=80,
                    alpha=0.8,
                    marker="s",
                    edgecolor="#2b2f36",
                )

    # Reference line: perfect robustness (y = x)
    max_val = max(subset["ccc_mse"].max(), subset["full_mse"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect robustness")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Full MSE (log scale)")
    ax.set_ylabel("CCC MSE at K=max (log scale)")
    ax.set_title("Fidelity degradation from top-K truncation\n(SHAP points at x≈0 by design)")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=False)

    path = output_dir / "ccc_vs_full_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_seed_variability(df: pd.DataFrame, output_dir: Path) -> Path:
    """Dot plot showing CCC MSE variability across seeds.

    More informative than box plots when seed count is small (<5).
    Shows that mean aggregation is justified if dots are clustered.
    """
    if "seed" not in df.columns or df["seed"].nunique() <= 1:
        # No seed variability - return early
        return output_dir / "seed_variability_dotplot.png"

    # Aggregate by seed first, then compute mean across seeds
    group_cols = ["dataset", "model", "explainer", "seed", "k_features"]
    if "ccc_mse" in df.columns:
        df_mean = df.groupby(group_cols, as_index=False)["ccc_mse"].mean()

    # Get max K for worst-case analysis
    max_k = int(df_mean["k_features"].max())
    subset = df_mean[df_mean["k_features"] == max_k].copy()

    datasets = sorted(subset["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([axes])

    for ax, dataset in zip(axes, datasets, strict=True):
        dataset_df = subset[subset["dataset"] == dataset]
        colors = [EXPLAINER_COLORS.get(name, "#4b5563") for name in dataset_df["explainer"]]

        for idx, (explainer, group) in enumerate(dataset_df.groupby("explainer")):
            seeds = group["seed"].values
            ccc_values = group["ccc_mse"].values
            color = EXPLAINER_COLORS.get(explainer, "#4b5563")

            # Dot/strip plot
            y_positions = np.arange(len(seeds)) + idx * 0.15
            ax.scatter(
                ccc_values, y_positions,
                c=color, label=explainer.upper() if idx == 0 else None,
                s=60, alpha=0.8, edgecolor="#2b2f36",
            )

            # Connect dots to show trend
            ax.plot(
                ccc_values, y_positions,
                c=color, alpha=0.3, linewidth=1,
            )

        ax.set_xlabel("CCC MSE")
        ax.set_yticks([])
        ax.set_title(dataset)
        ax.grid(axis="x", linestyle="--", alpha=0.25)

    # Global mean marker
    global_mean = subset.groupby("explainer")["ccc_mse"].mean()
    for explainer, mean_val in global_mean.items():
        color = EXPLAINER_COLORS.get(explainer, "#4b5563")
        for ax, dataset in zip(axes, datasets, strict=True):
            dataset_mean = subset[subset["dataset"] == dataset].groupby("explainer")["ccc_mse"].mean()
            if explainer in dataset_mean.index:
                ax.axvline(dataset_mean[explainer], color=color, alpha=0.5, linestyle=":")

    # Single legend
    handles = [plt.scatter([], [], c=EXPLAINER_COLORS.get(e, "#4b5563"), label=e.upper())
               for e in sorted(subset["explainer"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Seed variability (strip plot) - Dots near cluster center indicate stable results", fontweight="bold")
    path = output_dir / "seed_variability_dotplot.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir_for_results(results_path)
    df = load_results_table(results_path)
    paths = generate_summary_charts(df, output_dir)
    for path in paths:
        print(f"Saved benchmark summary figure: {path}")


if __name__ == "__main__":
    main()
