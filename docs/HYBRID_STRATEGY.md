### Complexity-Calibrated Local Concordance

**The Hybrid Intersection:** Quantitative Faithfulness (Functionally-grounded) + Explanation Conciseness (Human-grounded / Cognitive limit).

**The Targeted Gap:** Existing metrics often evaluate either faithfulness or
compactness, but not the fidelity of the explanation after it has been forced
into a small top-`K` feature budget. The project focuses on the question:

```text
Does a local additive explanation remain faithful to the black-box prediction
when only its top-K feature contributions are retained?
```

**Theoretical Definition:** Let `f(x)` be the black-box positive-class
probability. Let `g(x)` be the additive local reconstruction induced by an
explainer:

```text
g(x) = phi_0 + sum_i phi_i(x)
```

where `phi_0` is the explainer base value/intercept and `phi_i(x)` is the
feature contribution for instance `x`. The top-`K` reconstruction is:

```text
g_K(x) = phi_0 + sum_{i in S_K(x)} phi_i(x)
```

where `S_K(x)` contains the `K` largest absolute contributions. The implemented
score is:

```text
ccc_mse = mean_x (f(x) - g_K(x))^2
```

Lower `ccc_mse` means the explanation preserves local concordance with the
black-box model under a fixed complexity budget.

**Practical Experimental Design:**

*   **Datasets:** Tabular binary classification datasets (`breast_cancer`,
    `adult`).
*   **AI Task:** Predict the positive-class probability.
*   **Human Task:** None required directly for metric calculation; `K` acts as a
    proxy for a human-readable feature budget.
*   **Score Calculation:** Extract additive contributions from the explainer,
    retain the top `K` by absolute magnitude, reconstruct `g_K(x)`, and compute
    MSE against `f(x)`.
*   **Comparison Metrics:** Report `full_mse`, `top_k_degradation_mse`,
    `compactness_ratio`, `sufficiency_mse`, `comprehensiveness_abs_drop`, and
    `random_k_mse` to show what information `ccc_mse` adds over existing
    evaluation families.

See `docs/METRICS_AND_RESEARCH_GAP.md` for the full notation, state-of-the-art
metric overview, research gap, and evaluation plan.
