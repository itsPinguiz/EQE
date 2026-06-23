# EQE: Evaluating the Quality of Explanations

> Project P4 — Explainable and Trustworthy AI

This repository contains an experimental framework for evaluating the quality of
local explanations produced by Explainable AI methods on tabular classification
tasks.

The project focuses on a specific research gap: many explanation metrics measure
faithfulness, but do not ask whether the explanation remains faithful when it is
restricted to a small, human-readable number of features.

To address this, the framework implements **Complexity-Calibrated Local
Concordance**, reported as `ccc_mse`.

## Core Idea

For each explained instance, the framework:

1. generates a local explanation;
2. ranks features by attribution magnitude;
3. keeps only the top `K` features;
4. reconstructs the model prediction using the truncated explanation;
5. measures the MSE between the black-box probability and the top-`K`
   reconstruction.

Lower `ccc_mse` means the explanation remains more faithful under the cognitive
feature budget.

The framework also reports `random_k_mse`, a sanity-check baseline that keeps
`K` random features from the same explanation weights. This checks whether the
explainer-selected top-`K` features are more informative than random features.

## Implemented Benchmark

| Component | Current implementation |
| --- | --- |
| Datasets | `breast_cancer`, `adult` |
| Black-box models | `xgboost`, `neuralnetwork` |
| Explainers | `lime`, `shap`, `maple` |
| Metrics | `ccc_mse`, `full_mse`, `top_k_degradation_mse`, `compactness_ratio`, `sufficiency_mse`, `comprehensiveness_abs_drop`, `random_k_mse` |
| Main config | `config.yml` |
| Main results doc | `docs/SOTA_ANALYSIS.md` |

Current SOTA result:

```text
results/SOTA/results_20260518_141724.md
```

Summary: SHAP is the strongest explainer under `ccc_mse`; MAPLE is a useful
intermediate baseline; LIME is weakest overall. Top-`K` selection beats random-K
in every tested configuration.

## Repository Structure

```text
core/
  data_loader.py                 # dataset loading and preprocessing
  explainers.py                  # LIME, SHAP, MAPLE wrappers
  graph_utilities/
    benchmark_summary.py         # summary charts from benchmark results
    feature_reduction.py         # slide-ready top-K visualization
  main.py                        # CLI entry point
  metrics.py                     # ccc_mse and random_k_mse
  model.py                       # black-box model wrappers
  test_framework.py              # experiment orchestration
  third_party/MAPLE.py           # official MAPLE implementation
  visualize.py                   # visualization CLI
  utility/
    config.py                    # YAML config loader
    log.py                       # logging helpers

docs/
  PAPERS_REVIEW.md               # literature review notes
  PLAN.md                        # initial benchmark plan
  HYBRID_STRATEGY.md             # metric motivation
  SOTA_ANALYSIS.md               # current project analysis

results/
  SOTA/                          # selected benchmark outputs
  figures/                       # generated presentation figures
```

## Setup

The project uses Python `>=3.11` and is configured with `pyproject.toml`.

Using `uv`:

```bash
uv sync
```

Then run commands with:

```bash
uv run ...
```

## Running the Benchmark

The benchmark is controlled through `config.yml`.

Run the full configured experiment:

```bash
uv run core/main.py
```

or:

```bash
uv run python -m core.main --config config.yml
```

The output is printed to the terminal and saved as a markdown table under
`results/`.

## Configuring Experiments

The main knobs are in `config.yml`:

```yaml
experiment:
  suite:
    datasets: [breast_cancer, adult]
    k_features: [4, 5, 6, 7, 8, 9]
  seeds: [42, 123, 2026]
  n_explain: null
  n_jobs: null

models:
  - name: xgboost
  - name: neuralnetwork

explainers:
  - name: lime
  - name: shap
  - name: maple

metrics:
  - ccc_mse
  - full_mse
  - top_k_degradation_mse
  - compactness_ratio
  - sufficiency_mse
  - comprehensiveness_abs_drop
  - name: random_k_mse
```

When multiple seeds are configured, the saved markdown report includes both the
per-seed rows and an aggregate table with mean and standard deviation grouped by
dataset, model, explainer, and `K`.

For a faster smoke test, set:

```yaml
experiment:
  suite:
    datasets: [breast_cancer]
    k_features: [4, 9]
  n_explain: 20
  n_jobs: 2
```

## Generating Presentation Figures

To create the recommended summary charts from the SOTA results table:

```bash
uv run core/visualize.py benchmark-summary \
  --results results/SOTA/results_20260518_141724.md
```

This creates presentation PNGs under `results/figures/presentation/`.

To visualize the feature reduction process for a single instance:

```bash
uv run core/visualize.py feature-reduction \
  --dataset breast_cancer \
  --model xgboost \
  --explainer shap \
  --instance-index 0 \
  --k 4
```

This creates a PNG in `results/figures/` showing:

- local feature contributions;
- top-`K` features kept;
- discarded features;
- black-box prediction vs full explanation vs top-`K` reconstruction;
- cognitive budget summary.

Useful slide examples:

```bash
uv run core/visualize.py feature-reduction --dataset breast_cancer --model xgboost --explainer shap --instance-index 0 --k 4
uv run core/visualize.py feature-reduction --dataset breast_cancer --model xgboost --explainer shap --instance-index 0 --k 9
uv run core/visualize.py feature-reduction --dataset breast_cancer --model xgboost --explainer lime --instance-index 0 --k 4
uv run core/visualize.py feature-reduction --dataset breast_cancer --model xgboost --explainer maple --instance-index 0 --k 4
```

## Current Findings

The current SOTA run supports three main conclusions:

- **SHAP has the best explanation fidelity** under the top-`K` constraint.
- **MAPLE is a useful intermediate baseline**, often better than LIME but behind
  SHAP.
- **Random-K is consistently worse than top-K**, showing that explainer-selected
  features preserve more prediction signal than random feature subsets.

The metric is useful because it exposes trade-offs that model accuracy alone
does not. For example, on Breast Cancer the neural network is slightly more
accurate, but XGBoost produces more faithful SHAP explanations under the top-`K`
constraint.

See [docs/SOTA_ANALYSIS.md](docs/SOTA_ANALYSIS.md) for the full interpretation.

## Project Members

- Stefano Zizzi (346595)
- Michelangelo Ungolo (349109)
- Miriana Sette (349110)

---

Developed as part of the Explainable and Trustworthy AI curriculum.
