"""
core/test_framework.py
======================
Provides the :class:`ExperimentOrchestrator`, the central controller of the
evaluation pipeline. It coordinates dataset loading, model training,
explanation generation, and metric computation in a reproducible, end-to-end
workflow.

Pipeline Overview
-----------------
  1. _load_dataset()         → X_train, X_test, y_train, y_test
  2. _train_models()         → fitted BlackBoxModel instances
  3. _generate_explanations()→ per-instance weights & intercepts
  4. compute CCC metric for each (model, explainer) combo
  5. Aggregate & return results as a DataFrame
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from core.metrics import ComplexityCalibratedConcordance
from core.model import BlackBoxModel, NeuralNetworkModel, XGBoostModel

DatasetName  = Literal["adult", "breast_cancer", "diabetes", "german_credit"]
ExplainerType = Literal["lime", "shap", "maple", "l2x"]


class ExperimentOrchestrator:
    """Coordinates the end-to-end XAI evaluation pipeline.

    Parameters
    ----------
    dataset_name : DatasetName
        Identifier of the dataset to load. One of ``"adult"``,
        ``"breast_cancer"``, ``"diabetes"``, or ``"german_credit"``.
    k_features : int, optional
        Cognitive complexity limit K. Defaults to ``4``.
    test_size : float, optional
        Fraction of data held out for testing. Defaults to ``0.2``.
    random_state : int, optional
        Global seed for reproducibility. Defaults to ``42``.
    n_explain : int or None, optional
        Number of test instances to explain. ``None`` uses all. Defaults
        to ``None``.
    explainers : list[ExplainerType] or None, optional
        Explainer identifiers to evaluate. Defaults to ``["lime", "shap"]``.
    verbose : bool, optional
        Print progress messages if ``True``. Defaults to ``True``.

    Attributes
    ----------
    X_train, X_test : np.ndarray
        Feature matrices after preprocessing.
    y_train, y_test : np.ndarray
        Label vectors.
    feature_names : list[str]
        Human-readable feature names.
    models : dict[str, BlackBoxModel]
        Mapping from model name to fitted instance.
    results : pd.DataFrame
        CCC scores keyed by (dataset, model, explainer, K).

    Examples
    --------
    >>> orch = ExperimentOrchestrator("breast_cancer", k_features=4)
    >>> results = orch.run_experiment()
    >>> print(results)
    """

    def __init__(
        self,
        dataset_name: DatasetName,
        k_features: int = 4,
        test_size: float = 0.2,
        random_state: int = 42,
        n_explain: int | None = None,
        explainers: list[ExplainerType] | None = None,
        verbose: bool = True,
    ) -> None:
        self.dataset_name  = dataset_name
        self.k_features    = k_features
        self.test_size     = test_size
        self.random_state  = random_state
        self.n_explain     = n_explain
        self.explainers: list[ExplainerType] = explainers or ["lime", "shap"]
        self.verbose       = verbose

        self.X_train: npt.NDArray[np.float64] | None = None
        self.X_test:  npt.NDArray[np.float64] | None = None
        self.y_train: npt.NDArray[np.int_]    | None = None
        self.y_test:  npt.NDArray[np.int_]    | None = None
        self.feature_names: list[str]                = []
        self.models: dict[str, BlackBoxModel]        = {}
        self.results: pd.DataFrame                   = pd.DataFrame()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dataset(self, name: DatasetName) -> None:
        """Load, preprocess, and split the specified dataset.

        Datasets are sourced from ``sklearn.datasets`` (Breast Cancer WDBC)
        or UCI repositories / local CSV files (Adult, Diabetes, German Credit).

        Preprocessing steps (in order):
        1. Drop rows with missing values.
        2. One-hot encode nominal categorical columns.
        3. Standard-scale continuous features (zero mean, unit variance).
        4. Stratified train/test split using ``self.test_size``.

        Sets ``self.X_train``, ``self.X_test``, ``self.y_train``,
        ``self.y_test``, and ``self.feature_names`` upon completion.

        Parameters
        ----------
        name : DatasetName
            One of ``"adult"``, ``"breast_cancer"``, ``"diabetes"``,
            ``"german_credit"``.

        Raises
        ------
        ValueError
            If ``name`` is not a recognised dataset identifier.
        FileNotFoundError
            If a required CSV file cannot be found on disk.
        """
        raise NotImplementedError

    def _train_models(self) -> None:
        """Instantiate and fit all black-box models on the training split.

        Initialises :class:`~core.model.XGBoostModel` and
        :class:`~core.model.NeuralNetworkModel` with default hyper-parameters
        and calls their :meth:`~core.model.BlackBoxModel.fit` methods.

        Populates ``self.models`` with keys ``"xgboost"`` and
        ``"neural_network"``.  Prints test-set accuracy for each model when
        ``self.verbose`` is ``True``.

        Raises
        ------
        RuntimeError
            If :meth:`_load_dataset` has not been called first.
        """
        raise NotImplementedError

    def _generate_explanations(
        self,
        explainer_type: ExplainerType,
        model: BlackBoxModel,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate per-instance feature attribution explanations.

        Runs the requested explainer over the test set (or the first
        ``self.n_explain`` instances) and returns attribution weights and
        intercepts required to compute g_K(x).

        Implementation Notes
        --------------------
        **LIME:** Use ``lime.lime_tabular.LimeTabularExplainer`` with
        ``self.X_train`` as background. For each instance, call
        ``explain_instance`` with ``num_features=n_features`` (full set)
        and extract coefficients and intercept from the linear surrogate.

        **SHAP:** Use ``shap.TreeExplainer`` for XGBoost and
        ``shap.KernelExplainer`` (with a k-means background summary) for
        the MLP. Broadcast ``expected_value`` as the shared intercept.

        Parameters
        ----------
        explainer_type : ExplainerType
            One of ``"lime"`` or ``"shap"``.
        model : BlackBoxModel
            Fitted black-box model to explain.

        Returns
        -------
        weights : np.ndarray of shape (n_samples, n_features)
            Per-instance attribution weight vectors (w).
        intercepts : np.ndarray of shape (n_samples,)
            Per-instance base values / intercepts (w_0).

        Raises
        ------
        ValueError
            If ``explainer_type`` is not a supported identifier.
        RuntimeError
            If :meth:`_train_models` has not been called first.
        """
        raise NotImplementedError

    def _log(self, message: str) -> None:
        """Print a formatted log message if ``self.verbose`` is True.

        Parameters
        ----------
        message : str
            The message to display.
        """
        if self.verbose:
            print(f"[EQE] {message}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_experiment(self) -> pd.DataFrame:
        """Execute the full evaluation pipeline end-to-end.

        Steps
        -----
        1. :meth:`_load_dataset` — load and preprocess ``self.dataset_name``.
        2. :meth:`_train_models` — fit XGBoost and MLP black-box models.
        3. For each ``(model, explainer)`` pair:
           a. :meth:`_generate_explanations` — obtain attribution weights.
           b. Compute CCC score via
              :class:`~core.metrics.ComplexityCalibratedConcordance`.
        4. Collect results into a DataFrame with columns:
           ``["dataset", "model", "explainer", "k_features", "ccc_score"]``.
        5. Store in ``self.results`` and return.

        Returns
        -------
        pd.DataFrame
            One row per (model, explainer) pair with the CCC metric score.

        Raises
        ------
        RuntimeError
            If any pipeline step fails (propagated from sub-methods).
        """
        raise NotImplementedError