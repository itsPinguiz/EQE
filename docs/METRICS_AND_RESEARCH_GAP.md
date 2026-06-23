# Metrics and Research Gap

This document clarifies the formal notation used by the project and explains
how the experimental evaluation should be positioned against existing XAI
evaluation metrics.

## 1. Formal Notation

The project distinguishes three objects:

- `f(x)`: the black-box model prediction for instance `x`. In this project it
  is the positive-class probability returned by the trained classifier.
- `g(x)`: the local additive reconstruction induced by an explainer.
- `g_K(x)`: the same reconstruction after retaining only the top `K`
  explanatory feature contributions.

For each instance, the explainer returns a base value and one additive
contribution per interpretable feature:

```text
g(x) = phi_0 + sum_i phi_i(x)
```

where `phi_0` is the base value/intercept and `phi_i(x)` is the contribution of
feature `i` for the specific instance `x`.

The top-`K` feature set is:

```text
S_K(x) = indices of the K largest absolute contributions |phi_i(x)|
```

The truncated reconstruction is:

```text
g_K(x) = phi_0 + sum_{i in S_K(x)} phi_i(x)
```

The proposed metric is:

```text
ccc_mse = mean_x (f(x) - g_K(x))^2
```

Lower `ccc_mse` means that the explanation remains locally concordant with the
black-box model even after being compressed to `K` interpretable features.

Important implementation note: in the framework, `weights` are normalized as
additive contributions. They are not multiplied again by the raw feature values
inside `ccc_mse`. For MAPLE, local coefficients are converted into additive
contributions before metric computation.

## 2. State of the Art Metrics

Existing XAI evaluation metrics can be grouped by the property they measure.

| Family | Typical question | Examples | Main limitation for this project |
| --- | --- | --- | --- |
| Compactness / sparsity | How small is the explanation? | Number of selected features, sparsity, explanation size | Does not say whether the selected features are faithful to the model. |
| Local fidelity / concordance | Does the surrogate mimic the black-box locally? | Local MSE, local R2, LIME fidelity | Usually evaluates the full explanation, not the compressed top-`K` version. |
| Sufficiency | Are the selected features enough to preserve the prediction? | Keep only rationale/top features and measure prediction drift | Measures model behavior under masking, not whether explanation contributions are numerically calibrated. |
| Comprehensiveness | Does removing selected features change the prediction? | Remove rationale/top features and measure prediction drop | Shows feature influence, but not whether the additive explanation reconstructs `f(x)`. |
| Insertion / deletion / ROAR | Does performance change when important features are added or removed? | Deletion curves, insertion curves, remove-and-retrain | Strong faithfulness signal, but often expensive and sensitive to perturbation strategy. |
| Infidelity / sensitivity | Do attributions match model response to perturbations and remain stable? | Infidelity, sensitivity, local Lipschitz | Focuses on perturbation response or robustness, not cognitive top-`K` reconstruction. |
| Plausibility | Does the explanation match human rationales? | IOU, token F1, human rationale overlap | Human agreement does not guarantee faithfulness to the model. |

The project uses these families as comparison points rather than claiming that
`ccc_mse` replaces them.

## 3. Research Gap

The project targets the intersection of faithfulness and compactness.

Existing metrics often answer one of these questions:

- Is the explanation short?
- Is the full local explanation faithful?
- Are the top features influential when the input is perturbed?
- Is the explanation stable or plausible to humans?

The gap is narrower:

```text
Does a local additive explanation remain faithful to the black-box prediction
when it is forced into a fixed top-K feature budget?
```

This matters because a full explanation may reconstruct the model well only by
using many small contributions. Such an explanation can have good full fidelity
but poor top-`K` fidelity. Conversely, a compact explanation can be easy to read
but fail to reconstruct the prediction.

`ccc_mse` is designed to expose that trade-off directly.

## 4. Methodology

The project operationalizes Complexity-Calibrated Local Concordance as an MSE
between the black-box prediction and the top-`K` additive reconstruction:

```text
ccc_mse@K = mean_x (f(x) - g_K(x))^2
```

The name reflects two components:

- `Local Concordance`: the score compares the black-box output `f(x)` with a
  local reconstruction produced from the explanation.
- `Complexity-Calibrated`: the reconstruction is evaluated under a fixed
  feature budget `K`.

This should be presented as the project's proposed operational metric, not as a
universally established standard metric.

## 5. Experimental Comparison

The benchmark now supports the following comparison metrics:

| Metric | Meaning | Better |
| --- | --- | --- |
| `ccc_mse` | Error between `f(x)` and top-`K` additive reconstruction `g_K(x)` | Lower |
| `full_mse` | Error between `f(x)` and full additive reconstruction `g(x)` | Lower |
| `top_k_degradation_mse` | Extra error introduced by truncating `g(x)` to `g_K(x)` | Lower |
| `compactness_ratio` | Fraction of available features retained by top-`K` | Lower |
| `sufficiency_mse` | Prediction drift when only top-`K` input features are kept | Lower |
| `comprehensiveness_abs_drop` | Prediction change when top-`K` input features are removed | Higher |
| `random_k_mse` | Error from keeping `K` random contributions instead of top-`K` | Lower |

For `sufficiency_mse` and `comprehensiveness_abs_drop`, non-retained or removed
features are replaced with the training-set feature mean after preprocessing.
This baseline is simple and reproducible, but it should be reported because
perturbation-based metrics are sensitive to the masking strategy.

The key comparisons are:

- `ccc_mse` vs `compactness_ratio`: compactness alone does not imply
  faithfulness.
- `ccc_mse` vs `full_mse`: shows whether the complete explanation is faithful.
- `ccc_mse` vs `top_k_degradation_mse`: isolates the loss caused by the top-`K`
  truncation.
- `ccc_mse` vs `sufficiency_mse` and `comprehensiveness_abs_drop`: contrasts
  additive reconstruction fidelity with perturbation-based faithfulness.
- `ccc_mse` vs `random_k_mse`: checks whether explainer-selected top features
  preserve more signal than arbitrary feature subsets.

## 6. Robustness Across Seeds

The experiment configuration supports multiple seeds:

```yaml
experiment:
  seeds: [42, 123, 2026]
```

The output report contains both per-seed rows and an aggregate table with mean
and standard deviation grouped by:

```text
dataset, model, explainer, k_features
```

This addresses run-to-run variability in data splits, model training, MAPLE
sampling, and random baselines.
