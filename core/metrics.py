"""
core/metrics.py
===============
Defines the abstract interface for XAI evaluation metrics and the
novel **Complexity-Calibrated Local Concordance** metric.

Theoretical Background
----------------------
Standard local explainability metrics (e.g., SHAP local accuracy) assume
that *all* features are available. In practice, human working memory is
bounded (Miller's Law: ~7 ± 2 items). This module operationalises a
cognitive limit K, measuring the mathematical robustness of the explanation
(Complexity-Calibrated Local Concordance) when the explanation is 
**strictly truncated** to the top-K most important features.

Formal Definition
-----------------
Let f(x)  be the black-box probability for the positive class.
Let w_0   be the explainer's intercept (base value).
Let w     be the explainer's feature weight vector of length F.
Let S_K   be the index set of the K features with largest |w_i|.

Truncated prediction:
    g_K(x) = w_0 + sum_{i in S_K} w_i * x_i

Complexity-Calibrated Local Concordance (mean over N test instances):
    Score = (1/N) * sum_{j=1}^{N} (f(x^j) - g_K(x^j))^2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class EvaluationMetric(ABC):
    """Abstract interface for an XAI evaluation metric.

    Every concrete metric must implement :meth:`compute`, which accepts
    the necessary data and returns a scalar score (lower is better unless
    documented otherwise by the subclass).
    """

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> float:
        """Compute and return the scalar evaluation score.

        Parameters
        ----------
        *args, **kwargs
            Concrete sub-classes define their own signature. Refer to the
            respective subclass documentation for the exact expected inputs.

        Returns
        -------
        float
            The computed metric score.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Return a concise, human-readable description of the metric.

        Should include the metric name and any relevant hyper-parameters
        (e.g., the value of K for :class:`ComplexityCalibratedConcordance`).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the metric (used for registry/labels)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete Implementation
# ---------------------------------------------------------------------------

class ComplexityCalibratedConcordance(EvaluationMetric):
    """Complexity-Calibrated Local Concordance (CCC) metric.

    Measures the mean squared error between the black-box model's positive-
    class probabilities and the truncated local explanation's prediction,
    where the explanation is restricted to at most ``k_features`` features.

    A **lower** score indicates that the explainer can faithfully approximate
    the local decision boundary using only K features — i.e., it achieves
    high fidelity within the human cognitive limit.

    Parameters
    ----------
    k_features : int, optional
        The cognitive complexity limit K (i.e., the maximum number of
        features allowed in the explanation). Defaults to ``4``, following
        the conservative estimate of human working memory capacity used in
        the XAI literature.

    Attributes
    ----------
    k_features : int
        Stored value of the cognitive limit K.

    Examples
    --------
    >>> metric = ComplexityCalibratedConcordance(k_features=4)
    >>> score = metric.compute(
    ...     f_proba=model.predict_proba(X_test)[:, 1],
    ...     weights=shap_weights,       # shape (n_samples, n_features)
    ...     intercepts=shap_intercepts, # shape (n_samples,)
    ...     X=X_test,                   # shape (n_samples, n_features)
    ... )
    >>> print(f"CCC Score (MSE): {score:.4f}")
    """

    def __init__(self, k_features: int = 4) -> None:
        if k_features < 1:
            raise ValueError(f"k_features must be >= 1, got {k_features}.")
        self.k_features = k_features

    def compute(
        self,
        f_proba: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        intercepts: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
    ) -> float:
        """Compute the Complexity-Calibrated Local Concordance score.

        Algorithm
        ---------
        For each test instance j:
          1. Identify S_K: indices of the top-K features by absolute weight.
          2. Zero-out all weights outside S_K (truncation).
          3. Compute the truncated local prediction:
                g_K(x^j) = intercepts[j] + dot(weights_truncated[j], X[j])
          4. Compute the squared error: (f_proba[j] - g_K(x^j))^2

        Return the mean squared error across all N instances.

        Parameters
        ----------
        f_proba : np.ndarray of shape (n_samples,)
            Positive-class probability predictions from the black-box model
            f(x), i.e., ``model.predict_proba(X)[:, 1]``.
        weights : np.ndarray of shape (n_samples, n_features)
            Per-instance feature attribution weights produced by the explainer
            (e.g., SHAP values or LIME coefficients).
        intercepts : np.ndarray of shape (n_samples,)
            Per-instance intercept / base value from the explainer (w_0).
            For SHAP, this is the global ``expected_value``; for LIME, it is
            the linear model's intercept for each local instance.
        X : np.ndarray of shape (n_samples, n_features)
            The raw feature matrix of the test instances. Values should be
            in the same scale as used during explanation generation.

        Returns
        -------
        float
            Mean squared error between f(x) and g_K(x) across N instances.
            A perfect concordance score is 0.0.

        Raises
        ------
        ValueError
            If ``k_features`` exceeds the number of features in ``weights``.
        """
        if self.k_features > weights.shape[1]:
            raise ValueError(f"k_features ({self.k_features}) is greater than the number of features ({weights.shape[1]}).")

        # 1 & 2: Identify and truncate the feature weights (S_K)
        weights_truncated = self._truncate_weights(weights)

        # 3: Calculate the truncated local prediction: g_K(x) = w_0 + sum(w_i)
        # Note: SHAP values and LIME local coefficients are already the full attributions 
        # for the specific instance. They should NOT be multiplied by the raw X values.
        g_K = intercepts + np.sum(weights_truncated, axis=1)

        # 4: Return the Mean Squared Error (MSE)
        mse = np.mean((f_proba - g_K) ** 2)
        return float(mse)

    def _truncate_weights(
        self,
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Zero-out all but the top-K features (by absolute magnitude) per row.

        This is the core truncation step that enforces the cognitive complexity
        limit. For each instance (row), only the ``k_features`` entries with
        the largest absolute values are retained; all others are set to zero.

        Parameters
        ----------
        weights : np.ndarray of shape (n_samples, n_features)
            Full attribution weight matrix from the explainer.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Weight matrix with all but the top-K entries zeroed out per row.
        """
        truncated = np.copy(weights)
        
        # Sort the weight indices in descending order based on their absolute values
        sorted_indices = np.argsort(-np.abs(truncated), axis=1)
        
        # Extract the indices of the features that fall outside the cognitive window K
        indices_to_zero = sorted_indices[:, self.k_features:]
        
        # Zero out the weights of these less important features
        row_indices = np.arange(weights.shape[0])[:, None]
        truncated[row_indices, indices_to_zero] = 0.0
        
        return truncated

    def __repr__(self) -> str:
        return f"ComplexityCalibratedConcordance(k_features={self.k_features})"

    @property
    def name(self) -> str:
        return "ccc_mse"


class RandomKConcordance(EvaluationMetric):
    """Random-K baseline for additive explanation concordance.

    This baseline keeps ``k_features`` randomly selected contributions per
    instance instead of the explainer's top-K contributions. It uses the same
    explanation weights and intercepts as the main CCC metric, making it a
    direct sanity check for whether top-K selection beats random feature
    selection.
    """

    def __init__(
        self,
        k_features: int = 4,
        repeats: int = 30,
        random_state: int | None = 42,
    ) -> None:
        if k_features < 1:
            raise ValueError(f"k_features must be >= 1, got {k_features}.")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}.")

        self.k_features = k_features
        self.repeats = repeats
        self.random_state = random_state

    def compute(
        self,
        f_proba: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        intercepts: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
    ) -> float:
        if self.k_features > weights.shape[1]:
            raise ValueError(
                f"k_features ({self.k_features}) is greater than the number "
                f"of features ({weights.shape[1]})."
            )

        if self.k_features == weights.shape[1]:
            g_k = intercepts + np.sum(weights, axis=1)
            return float(np.mean((f_proba - g_k) ** 2))

        rng = np.random.default_rng(self.random_state)
        scores = []

        for _ in range(self.repeats):
            random_order = rng.random(weights.shape)
            selected = np.argpartition(random_order, self.k_features - 1, axis=1)[
                :, : self.k_features
            ]

            truncated = np.zeros_like(weights)
            row_indices = np.arange(weights.shape[0])[:, None]
            truncated[row_indices, selected] = weights[row_indices, selected]

            g_k = intercepts + np.sum(truncated, axis=1)
            scores.append(np.mean((f_proba - g_k) ** 2))

        return float(np.mean(scores))

    def __repr__(self) -> str:
        return (
            "RandomKConcordance("
            f"k_features={self.k_features}, "
            f"repeats={self.repeats}, "
            f"random_state={self.random_state})"
        )

    @property
    def name(self) -> str:
        return "random_k_mse"


METRIC_REGISTRY = {
    "ccc_mse": ComplexityCalibratedConcordance,
    "random_k_mse": RandomKConcordance,
}


def build_metrics(
    metric_names: Iterable[str | dict[str, Any]] | None,
    *,
    k_features: int,
    random_state: int | None = 42,
) -> list[EvaluationMetric]:
    """Instantiate metrics from the registry.

    Parameters
    ----------
    metric_names : Iterable[str] | None
        Metric names to instantiate. If None, defaults to ["ccc_mse"].
    k_features : int
        Cognitive limit K, passed to metrics that need it.
    """
    names = list(metric_names) if metric_names is not None else ["ccc_mse"]
    metrics: list[EvaluationMetric] = []

    for metric_cfg in names:
        if isinstance(metric_cfg, str):
            name = metric_cfg
            params: dict[str, Any] = {}
        elif isinstance(metric_cfg, dict):
            name = metric_cfg.get("name")
            if not name:
                raise ValueError("Ogni metrica deve avere un campo 'name'.")
            params = dict(metric_cfg.get("params", {}))
            params.update(
                {
                    key: value
                    for key, value in metric_cfg.items()
                    if key not in {"name", "params"}
                }
            )
        else:
            raise ValueError("Metriche non valide nel config.yml.")

        key = name.lower()
        metric_cls = METRIC_REGISTRY.get(key)
        if metric_cls is None:
            raise ValueError(f"Metrica non supportata: {name}")

        params.setdefault("k_features", k_features)
        if key == "random_k_mse":
            params.setdefault("random_state", random_state)

        metrics.append(metric_cls(**params))

    return metrics
