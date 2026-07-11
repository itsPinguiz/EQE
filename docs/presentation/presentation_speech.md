# Presentation Speech

## Speaker Split

- Stefano: slides 1-7, introduction and CCC metric.
- Michelangelo: slides 8-15, benchmark setup and main empirical results.
- Miriana: slides 16-23, comparison with other metrics, takeaways, and closing.

---

## Stefano - Slides 1-7

### Slide 1 - Title

Good morning. We are Stefano Zizzi, Michelangelo Ungolo, and Miriana Sette.

Today we present our project, **Evaluating the Quality of Explanations under a Cognitive Feature Budget**.

The main idea is simple: in Explainable AI, we often evaluate whether a full explanation is faithful to the model. But in practice, users rarely see the full explanation. They usually see only a small number of features. So our question is: what happens to fidelity when the explanation is made short enough for a human to inspect?

### Slide 2 - Problem

Local explanation methods such as LIME, SHAP, and MAPLE usually assign one contribution to each input feature.

Many evaluation metrics check the complete explanation, meaning all feature contributions are included. However, real interfaces, dashboards, and reports normally display only the most important features.

This creates a gap between **full fidelity** and **displayed fidelity**. An explanation can reconstruct the model well when all features are used, but become much less faithful when only the top features are shown.

So the core question of the project is: **how much model fidelity is preserved when only the top-K contributions are displayed?**

### Slide 3 - Table of Contents: Metric

We start from the metric, because this is the main contribution of the project.

### Slide 4 - Contribution

To study this problem, we introduce **Complexity-Calibrated Local Concordance**, or **CCC**.

CCC is a budget-aware metric. Instead of asking whether the complete explanation reconstructs the model output, it asks whether the explanation still reconstructs the model output after keeping only K features.

We evaluate K from 5 to 9, following the idea of a limited cognitive budget. Lower CCC means lower reconstruction error, so the displayed explanation is more faithful to the black-box prediction.

The goal is not to replace existing metrics, but to complement them. CCC adds a specific view: fidelity under a human-sized feature budget.

### Slide 5 - CCC Definition

Formally, for each instance we start from the black-box probability, called `f(x)`.

The explainer gives us an intercept `w0` and a vector of feature contributions, `phi`.

We select `S_K`, the set of the K largest contributions in absolute value. Then we reconstruct the prediction using only those K terms:

`g_K(x) = w0 + sum of phi_i over the selected features`.

CCC is the mean squared error between the black-box prediction and this truncated reconstruction.

So if CCC is low, the short explanation is still close to the model output. If CCC is high, the displayed explanation loses fidelity.

### Slide 6 - Common Explainer Format

Since we compare different explainers, we first normalize them into the same additive format.

For SHAP, this is straightforward, because SHAP values are already additive contributions.

For LIME, the local coefficients are multiplied by the interpretable representation of the instance. This gives instance-level additive contributions.

For MAPLE, the coefficients are multiplied by the corresponding feature values.

The important assumption is that all methods are evaluated through the same structure:

`g(x) = w0 + sum of phi_i(x)`.

This makes the comparison consistent across LIME, SHAP, and MAPLE.

### Slide 7 - Table of Contents: Benchmark

Now that the metric is defined, Michelangelo will explain how we used it in the benchmark.

---

## Michelangelo - Slides 8-15

### Slide 8 - Experimental Setup

The benchmark uses two tabular datasets: UCI Adult and UCI Breast Cancer Wisconsin Diagnostic.

We test two black-box classifiers: XGBoost and a neural network.

For explanations, we compare three local methods: LIME, SHAP KernelExplainer, and MAPLE.

For each dataset, model, and explainer combination, we compute CCC for K equal to 5, 6, 7, 8, and 9. We also repeat the experiments over three random seeds: 42, 123, and 2026.

This gives us a controlled setting where we can observe how compact fidelity changes as the feature budget increases.

### Slide 9 - Benchmark Pipeline

The benchmark pipeline is reproducible.

First, we load and scale the dataset. Then we train each black-box classifier.

After that, we generate local explanations for each model and explainer pair. These explanations are converted into the common additive format described before.

Finally, we compute several metrics: CCC, full MSE, random-K MSE, sufficiency, and comprehensiveness.

The orchestration is implemented in `core/test_framework.py`, while the metric definitions are implemented in `core/metrics.py`.

### Slide 10 - Model Accuracy

This slide shows the mean model accuracy across the three seeds.

The classifiers perform well overall. For example, XGBoost reaches about 0.874 accuracy on Adult, and the neural network reaches about 0.977 on Breast Cancer.

But the key point is that accuracy alone does not tell us which explainer is better.

A model can be accurate and still difficult to explain compactly. CCC measures the quality of the displayed explanation, not just the classifier performance.

### Slide 11 - Table of Contents: Results

Now we move to the main results.

### Slide 12 - CCC Curves

This figure shows CCC as K increases.

The lower the curve, the more faithful the compact explanation is. In general, SHAP has the lowest curves, especially as K increases.

This is expected because SHAP has a strong additive reconstruction property. But the important observation is that CCC is not always zero at small K. Even if the full explanation is faithful, the compact version can still lose information.

MAPLE is usually intermediate. In some cases it improves with K, while in others it stays almost flat.

LIME is the most unstable, and in some Breast Cancer settings the error does not decrease monotonically.

### Slide 13 - Representative CCC Values

This table gives representative aggregate CCC values.

On Adult with the neural network, SHAP decreases from about 0.007 at K equal to 5 to almost zero at K equal to 9.

MAPLE is almost flat in the Adult cases, meaning that increasing K does not recover much additional reconstruction signal.

LIME on Adult with the neural network is much worse, with CCC above 3 even at K equal to 9.

On Breast Cancer, SHAP again improves strongly as K increases. MAPLE improves more gradually. LIME is interesting because it can improve at first and then get worse.

### Slide 14 - Explainer Patterns

The results show three main patterns.

First, SHAP gives the best compact reconstructions overall. Its error decreases quickly as more features are included.

Second, MAPLE behaves like an intermediate baseline. It can recover some signal, but its top-ranked features are not always strongly concentrated.

Third, LIME is the most sensitive method. Its local surrogate does not guarantee that the selected additive terms reconstruct the black-box probability well.

This is exactly the type of behavior CCC is designed to expose.

### Slide 15 - Why LIME Can Be Non-Monotonic

LIME can be non-monotonic because adding one more feature does not necessarily reduce the squared reconstruction error.

There are several reasons for this. LIME fits a local sparse surrogate and does not enforce local accuracy. The selected feature set is not guaranteed to be nested across different K values. Also, Breast Cancer features are highly correlated, so added terms can interact with the reconstruction direction.

In the example shown here, Breast Cancer with the neural network improves from K equal to 5 to K equal to 7, but then worsens at K equal to 8 and 9.

So this is not necessarily a bug. It shows that LIME's local weights can be unstable under a compact additive reconstruction metric.

I will now pass to Miriana, who will compare CCC with the other metrics and conclude the presentation.

---

## Miriana - Slides 16-23

### Slide 16 - CCC Versus Full MSE

This slide shows why CCC adds information beyond full MSE.

The example is SHAP on Breast Cancer with XGBoost. The full MSE is zero for every K, because the complete SHAP reconstruction is exact.

However, CCC is not zero when we keep only the top features. At K equal to 5, CCC is 0.0160, and it decreases as K increases.

This means that full fidelity and compact fidelity are different concepts. Full MSE says the complete explanation is faithful. CCC says how much fidelity remains when the explanation is shortened.

### Slide 17 - Top-K Versus Random-K

The random-K baseline is an internal sanity check.

If the explainer ranking is meaningful, the top-K features should reconstruct the prediction better than a random subset of K features.

In most settings, this is exactly what we observe. Top-K CCC is much lower than random-K MSE, especially for SHAP.

When the gap is smaller, it suggests that the explainer's feature ranking is less informative.

### Slide 18 - Metric Comparison

CCC answers a different question from the other metrics.

Full MSE evaluates the complete additive reconstruction. CCC evaluates the reconstruction after top-K truncation.

Sufficiency and comprehensiveness are perturbation metrics: they modify the input and observe how the model prediction changes.

CCC does not perturb the input. Instead, it directly asks whether the displayed additive terms reconstruct the black-box probability.

So CCC is especially useful when the explanation is actually shown to a user as a short list of feature contributions.

### Slide 19 - Table of Contents: Takeaways

We now summarize the main takeaways.

### Slide 20 - What CCC Adds

CCC separates three ideas that are often mixed together: model accuracy, full explanation fidelity, and compact explanation fidelity.

It also gives a full curve across different values of K. This curve tells us how quickly the top-ranked features recover the model prediction.

In practice, CCC could be summarized using the area under the CCC curve, or by finding the smallest K that reaches an acceptable error threshold.

This makes CCC useful both as an offline benchmark metric and as a possible runtime warning in deployed explanation tools.

### Slide 21 - Limitations and Extensions

There are also limitations.

CCC is designed for additive explanations, so it does not directly apply to rule lists, counterfactual explanations, or prototypes.

It also depends on the scale and calibration of predicted probabilities, so raw CCC values should be interpreted carefully when moving across datasets.

Another limitation is that a fixed K is only a proxy for cognitive complexity. For example, one-hot encoded features may not correspond to human-level concepts.

As future work, the benchmark could be extended to more datasets, more model families, and adaptive values of K.

### Slide 22 - Conclusion

The main conclusion is that compact faithfulness is a distinct evaluation target.

An explanation can be faithful when all features are included, but less faithful when only the displayed top-K features are used.

CCC measures exactly this displayed-explanation fidelity.

In our benchmark, SHAP gives the strongest compact reconstructions, MAPLE is intermediate, and LIME is more sensitive to K and local instability.

More generally, explainer selection should not depend on a single metric. CCC should be used together with perturbation metrics when practical decisions depend on both additive reconstruction and input-masking behavior.

### Slide 23 - Backmatter

Thank you for listening. We are happy to answer any questions.
