# EQE: Evaluating the Quality of Explanations

> Project P4 - Explainable and Trustworthy AI

EQE is an experimental framework for evaluating local explanations produced by
Explainable AI methods on tabular binary classification tasks.

The project focuses on a specific gap in XAI evaluation: many metrics measure
whether a complete explanation is faithful to the black-box model, but they do
not ask whether that faithfulness survives when the explanation is reduced to a
small number of human-readable features.

To study this trade-off, the framework introduces and benchmarks
**Complexity-Calibrated Local Concordance (CCC)**, reported as `ccc_mse`.

## Research Question

If a user can inspect only `K` features, which explainer still reconstructs the
black-box prediction faithfully?

This question is motivated by the cognitive constraint commonly summarized by
Miller's Law: people can only hold a small number of information units in working
memory at once. In this project, `K` is varied from 4 to 9 features.

## Main Metric: CCC

For each explained instance, EQE:

1. generates an additive local explanation;
2. ranks features by absolute attribution magnitude;
3. keeps only the top `K` feature contributions;
4. reconstructs the local prediction from the truncated explanation;
5. computes the MSE against the black-box positive-class probability.

Formally:

```text
g_K(x) = w_0 + sum(phi_i(x) for i in top_K(abs(phi(x))))

CCC = mean((f(x) - g_K(x))^2)
```

where:

- `f(x)` is the black-box model probability for the positive class;
- `w_0` is the explainer intercept or base value;
- `phi_i(x)` is the additive contribution of feature `i`;
- lower `ccc_mse` means better compact faithfulness.

CCC is defined over additive contributions, not over raw explainer
coefficients. The explainer wrappers therefore normalize their outputs before
the metric is computed: SHAP already returns additive contributions; MAPLE
local coefficients are converted as `phi_i(x) = beta_i * x_i`; LIME surrogate
coefficients are converted as `phi_i(x) = beta_i * z_i(x)`, where `z_i(x)` is
LIME's interpretable/scaled representation of the explained instance.

The framework also reports `random_k_mse`, a direct sanity-check baseline that
keeps `K` random features from the same explanation weights. This tests whether
the explainer's top-K features preserve more signal than arbitrary feature
subsets.

## Implemented Benchmark

| Component | Implementation |
| --- | --- |
| Datasets | [`breast_cancer`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [`adult`](https://archive.ics.uci.edu/dataset/2/adult) |
| Black-box models | `xgboost`, `neuralnetwork` |
| Explainers | `lime`, `shap`, `maple` |
| Feature budgets | `K = 4, 5, 6, 7, 8, 9` |
| Seeds | `42`, `123`, `2026` |
| Main metric | `ccc_mse` |
| Control baseline | `random_k_mse` |
| Other metrics | `full_mse`, `sufficiency_mse`, `comprehensiveness_abs_drop`, `ccc_mse_normalized`, `top_k_degradation_ratio` |
| Main config | `config.yml` |
| Latest results | `results/latest.md` |
| Technical report | `docs/relazione.md` |

The current run in `results/latest.md` was generated with `n_explain: 1000`,
two datasets, two models, three explainers, six `K` values, and three random
seeds.

## Datasets

| Config name | Source |
| --- | --- |
| `breast_cancer` | [UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| `adult` | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) |

## Project Workflow

The benchmark pipeline is:

1. load and preprocess the selected dataset;
2. train each configured black-box model;
3. generate local explanations for every model/explainer pair;
4. cache the additive explanation contributions and intercepts for the run;
5. compute all configured metrics for every `K`;
6. aggregate results across seeds;
7. save `results/latest.md` and regenerate summary figures.

`results/latest.md` is the active result file. When a new run starts, the
previous `latest.md` is moved into `results/archive/`.

## Current Findings

The latest benchmark supports these conclusions:

- **SHAP is the strongest explainer under `ccc_mse`**. Its complete additive
  explanations have near-zero `full_mse`, and its top-K reconstructions improve
  sharply as `K` increases.
- **CCC exposes a hidden trade-off in SHAP**. Even when the full explanation is
  exact, small `K` values can still lose fidelity because many small
  contributions may be needed to reconstruct the prediction.
- **MAPLE is a useful intermediate baseline**. It is often more faithful than
  LIME, but its top-K ranking can be weak in some model/dataset settings.
- **LIME is the least reliable overall in this benchmark**, especially on the
  Adult dataset with the neural network model.
- **Top-K usually beats random-K**, showing that explainer-selected features
  carry more prediction signal than random feature subsets.

Model accuracy alone is not enough to choose an explanation method. For example,
the Breast Cancer neural network reaches slightly higher accuracy than XGBoost,
but XGBoost can produce more faithful SHAP explanations under the compact top-K
constraint.

See [docs/relazione.md](docs/relazione.md) for the full technical analysis.

## Repository Structure

```text
core/
  data_loader.py                 # Breast Cancer and Adult loading/preprocessing
  explainers.py                  # SHAP, LIME, MAPLE wrappers
  graph_utilities/
    benchmark_summary.py         # summary charts from markdown results
    feature_reduction.py         # top-K feature reduction visualizations
  main.py                        # benchmark CLI entry point
  metrics.py                     # CCC and comparison metrics
  model.py                       # XGBoost and neural network wrappers
  test_framework.py              # experiment orchestration
  third_party/MAPLE.py           # bundled MAPLE implementation
  utility/
    config.py                    # YAML config loader
    log.py                       # logging helpers
  visualize.py                   # visualization CLI

datasets/
  adult/                         # UCI Adult files
  BSWD/                          # Wisconsin Diagnostic Breast Cancer files

docs/
  relazione.md                   # technical report and comparative analysis
  PAPERS_REVIEW.md               # reviewed XAI literature
  report/TeX/                    # LaTeX report sources

results/
  latest.md                      # latest benchmark output
  archive/                       # previous benchmark outputs
  figures/latest/                # regenerated charts for latest.md
  figures/presentation/          # selected presentation figures
```

## Setup

The project uses Python `>=3.11` and is configured with `pyproject.toml`.

Using `uv`:

```bash
uv sync
```

Run commands through the environment with:

```bash
uv run ...
```

## Running the Benchmark

The full experiment is controlled by `config.yml`.

```bash
uv run python -m core.main --config config.yml
```

Equivalent console script:

```bash
uv run eqe --config config.yml
```

The benchmark writes:

- `results/latest.md`;
- archived previous results under `results/archive/`;
- summary charts under `results/figures/latest/`;
- logs under `results/experiment.log`.

Set `experiment.verbose: false` for clean progress-only output. In that mode,
the main process owns the progress display: one bar tracks pipeline stages
and one bar tracks explanation samples. Set `experiment.verbose: true` when
you want detailed logs instead.

## Configuring Experiments

The most important knobs are in `config.yml`:

```yaml
experiment:
  suite:
    datasets: [breast_cancer, adult]
    k_features: [4, 5, 6, 7, 8, 9]
  seeds: [42, 123, 2026]
  n_explain: 1000
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
  - sufficiency_mse
  - comprehensiveness_abs_drop
  - name: random_k_mse
    params:
      repeats: 30
  - ccc_mse_normalized
  - top_k_degradation_ratio
```

For a faster smoke test, reduce the suite and sample count:

```yaml
experiment:
  suite:
    datasets: [breast_cancer]
    k_features: [4, 9]
  seeds: [42]
  n_explain: 20
  n_jobs: 2
```

## Generating Figures

Generate summary charts from the latest benchmark:

```bash
uv run python -m core.visualize benchmark-summary --results results/latest.md
```

or:

```bash
uv run eqe-visualize benchmark-summary --results results/latest.md
```

Generate a single top-K feature reduction example:

```bash
uv run python -m core.visualize feature-reduction \
  --dataset breast_cancer \
  --model xgboost \
  --explainer shap \
  --instance-index 0 \
  --k 4
```

This creates a figure showing:

- the ranked local feature contributions;
- which features are kept under the top-K budget;
- which features are discarded;
- the black-box probability vs the full and top-K reconstructions.

Useful presentation examples:

```bash
uv run python -m core.visualize feature-reduction --dataset breast_cancer --model xgboost --explainer shap --instance-index 0 --k 4
uv run python -m core.visualize feature-reduction --dataset breast_cancer --model xgboost --explainer shap --instance-index 0 --k 9
uv run python -m core.visualize feature-reduction --dataset breast_cancer --model xgboost --explainer lime --instance-index 0 --k 4
uv run python -m core.visualize feature-reduction --dataset breast_cancer --model xgboost --explainer maple --instance-index 0 --k 4
```

## Metric Notes

| Metric | Meaning | Direction |
| --- | --- | --- |
| `ccc_mse` | MSE between black-box probability and top-K additive reconstruction | lower is better |
| `full_mse` | MSE using the full additive explanation | lower is better |
| `random_k_mse` | Same reconstruction as CCC, but with random K features | lower is better |
| `ccc_mse_normalized` | CCC divided by the variance of black-box probabilities | lower is better |
| `top_k_degradation_ratio` | `ccc_mse / full_mse`; undefined for near-exact full explanations | lower is better |
| `sufficiency_mse` | Prediction drift when only top-K input features are kept | lower is better |
| `comprehensiveness_abs_drop` | Mean absolute prediction change after removing top-K input features | higher is better |

## Known Limitations

- CCC assumes additive explanation outputs of the form
  `g(x) = w_0 + sum(phi_i(x))`.
- Raw local surrogate coefficients must be converted to additive
  contributions before CCC is computed. In particular, LIME uses its
  interpretable representation `z(x)`, so the contribution is
  `phi_i(x) = beta_i * z_i(x)`.
- Cross-dataset comparisons should prefer `ccc_mse_normalized`, because raw MSE
  depends on the prediction distribution.
- Perturbation metrics such as `sufficiency_mse` and
  `comprehensiveness_abs_drop` depend on the feature baseline used for masking.
- `top_k_degradation_ratio` is not informative for methods with near-zero
  `full_mse`, such as SHAP in this benchmark.

## Project Members

- Stefano Zizzi (346595)
- Michelangelo Ungolo (349109)
- Miriana Sette (349110)

Developed as part of the Explainable and Trustworthy AI curriculum.
