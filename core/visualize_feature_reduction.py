"""
Generate presentation-ready visual examples of top-K feature reduction.

Example:
    uv run core/visualize_feature_reduction.py \
        --dataset breast_cancer \
        --model xgboost \
        --explainer shap \
        --instance-index 0 \
        --k 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.config import DEFAULT_CONFIG_PATH, load_config
from core.main import _normalize_explainers, _normalize_models
from core.test_framework import ExperimentOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EQE feature reduction visualizer",
        description="Create a plot showing full local explanation vs top-K reduction.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "adult"])
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "neuralnetwork"])
    parser.add_argument("--explainer", default="shap", choices=["lime", "shap", "maple"])
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def _safe_filename(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "-")


def _build_orchestrator(args: argparse.Namespace) -> ExperimentOrchestrator:
    config = load_config(args.config)
    experiment = config.get("experiment", {})
    models, model_params = _normalize_models(config.get("models"))
    _, explainer_params = _normalize_explainers(config.get("explainers"))

    return ExperimentOrchestrator(
        dataset_name=args.dataset,
        k_features=[args.k],
        test_size=experiment.get("test_size", 0.2),
        random_state=experiment.get("random_state", experiment.get("seed", 42)),
        n_explain=None,
        models=[args.model],
        model_params=model_params if models else {},
        explainers=[args.explainer],
        explainer_params=explainer_params,
        metrics=config.get("metrics"),
        n_jobs=1,
        verbose=False,
    )


def _compute_example(args: argparse.Namespace) -> dict:
    orchestrator = _build_orchestrator(args)
    orchestrator._load_dataset()
    orchestrator._train_models()

    if args.instance_index < 0 or args.instance_index >= orchestrator.X_test.shape[0]:
        raise ValueError(
            f"instance-index must be between 0 and {orchestrator.X_test.shape[0] - 1}."
        )

    model = orchestrator.models[args.model]
    x = orchestrator.X_test[args.instance_index : args.instance_index + 1]
    explainer = orchestrator._init_explainer(args.explainer, model)
    weights, intercepts = explainer.explain(x)

    contributions = weights[0]
    intercept = float(intercepts[0])
    f_proba = float(model.predict_proba(x)[0, 1])
    full_reconstruction = intercept + float(np.sum(contributions))

    ranked_indices = np.argsort(-np.abs(contributions))
    k = min(args.k, contributions.shape[0])
    top_k_indices = ranked_indices[:k]
    top_k_reconstruction = intercept + float(np.sum(contributions[top_k_indices]))
    top_k_mse = float((f_proba - top_k_reconstruction) ** 2)

    return {
        "feature_names": orchestrator.feature_names,
        "contributions": contributions,
        "ranked_indices": ranked_indices,
        "top_k_indices": set(int(idx) for idx in top_k_indices),
        "f_proba": f_proba,
        "intercept": intercept,
        "full_reconstruction": full_reconstruction,
        "top_k_reconstruction": top_k_reconstruction,
        "top_k_mse": top_k_mse,
        "accuracy": orchestrator.model_scores[args.model],
        "n_features": contributions.shape[0],
    }


def _plot_feature_reduction(args: argparse.Namespace, example: dict) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = example["feature_names"]
    contributions = example["contributions"]
    ranked_indices = example["ranked_indices"]
    top_k_indices = example["top_k_indices"]
    top_n = min(args.top_n, len(ranked_indices))
    shown_indices = ranked_indices[:top_n][::-1]

    labels = [feature_names[idx] for idx in shown_indices]
    values = contributions[shown_indices]
    colors = ["#1f9d55" if int(idx) in top_k_indices else "#b7bec8" for idx in shown_indices]

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.2, 1.0],
        height_ratios=[1.0, 0.55, 0.45],
    )
    ax_features = fig.add_subplot(grid[:, 0])
    ax_reconstruction = fig.add_subplot(grid[0, 1])
    ax_budget = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[2, 1])

    y_pos = np.arange(len(labels))
    ax_features.barh(y_pos, values, color=colors, edgecolor="#2b2f36", linewidth=0.5)
    ax_features.axvline(0, color="#2b2f36", linewidth=1)
    ax_features.set_yticks(y_pos)
    ax_features.set_yticklabels(labels, fontsize=9)
    ax_features.set_xlabel("Local contribution")
    ax_features.set_title(f"Top-{args.k} feature reduction ({args.explainer.upper()})")
    ax_features.grid(axis="x", linestyle="--", alpha=0.25)

    kept_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#1f9d55",
        edgecolor="#2b2f36",
        linewidth=0.8,
    )
    removed_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#b7bec8",
        edgecolor="#2b2f36",
        linewidth=0.8,
    )
    ax_features.legend(
        [kept_patch, removed_patch],
        [f"kept top-{args.k}", "discarded"],
        loc="lower right",
        frameon=False,
    )

    reconstruction_labels = ["black-box\nf(x)", "full\nexplanation", f"top-{args.k}\ng_K(x)"]
    reconstruction_values = [
        example["f_proba"],
        example["full_reconstruction"],
        example["top_k_reconstruction"],
    ]
    reconstruction_colors = ["#2f80ed", "#7b61ff", "#1f9d55"]
    ax_reconstruction.bar(
        reconstruction_labels,
        reconstruction_values,
        color=reconstruction_colors,
        edgecolor="#2b2f36",
        linewidth=0.5,
    )
    min_reconstruction = min(reconstruction_values)
    max_reconstruction = max(reconstruction_values)
    spread = max_reconstruction - min_reconstruction
    padding = max(0.02, spread * 0.35)
    y_min = max(0.0, min_reconstruction - padding)
    y_max = min(1.0, max_reconstruction + padding)
    if y_max - y_min < 0.08:
        center = (y_min + y_max) / 2
        y_min = max(0.0, center - 0.04)
        y_max = min(1.0, center + 0.04)

    ax_reconstruction.set_ylim(y_min, y_max)
    ax_reconstruction.set_ylabel("Positive-class probability")
    ax_reconstruction.set_title("Prediction reconstruction (zoomed)")
    ax_reconstruction.grid(axis="y", linestyle="--", alpha=0.25)
    for idx, value in enumerate(reconstruction_values):
        label_offset = (y_max - y_min) * 0.04
        ax_reconstruction.text(
            idx,
            min(value + label_offset, y_max - label_offset),
            f"{value:.3f}",
            ha="center",
        )

    kept = args.k
    discarded = example["n_features"] - kept
    ax_budget.barh(
        ["features"],
        [kept],
        color="#1f9d55",
        edgecolor="#2b2f36",
        linewidth=0.7,
        label="kept",
    )
    ax_budget.barh(
        ["features"],
        [discarded],
        left=[kept],
        color="#b7bec8",
        edgecolor="#2b2f36",
        linewidth=0.7,
        label="discarded",
    )
    ax_budget.set_xlim(0, example["n_features"])
    ax_budget.set_title("Cognitive budget")
    ax_budget.set_xlabel(f"{kept} kept / {example['n_features']} total")
    ax_budget.legend(loc="lower right", frameon=False)

    ax_summary.axis("off")
    summary_text = (
        f"top-K squared error: {example['top_k_mse']:.6f}\n"
        f"black-box f(x): {example['f_proba']:.3f}\n"
        f"top-{args.k} g_K(x): {example['top_k_reconstruction']:.3f}\n"
        f"model accuracy: {example['accuracy']:.4f}"
    )
    ax_summary.text(
        0.0,
        1.0,
        summary_text,
        fontsize=10,
        va="top",
        ha="left",
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f6f8fa",
            "edgecolor": "#d0d7de",
            "linewidth": 0.8,
        },
    )

    fig.suptitle(
        f"{args.dataset} | {args.model} | instance #{args.instance_index}",
        fontsize=14,
        fontweight="bold",
    )

    filename = (
        f"feature_reduction_"
        f"{_safe_filename(args.dataset)}_"
        f"{_safe_filename(args.model)}_"
        f"{_safe_filename(args.explainer)}_"
        f"k{args.k}_"
        f"i{args.instance_index}.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    example = _compute_example(args)
    output_path = _plot_feature_reduction(args, example)
    print(f"Saved feature-reduction figure: {output_path}")


if __name__ == "__main__":
    main()
