"""
core/model.py
=============
Defines the abstract interface for black-box models and its concrete
implementations (XGBoost, MLP). All models share a common API so that
the ExperimentOrchestrator can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BlackBoxModel(ABC):
    """Abstract interface for a trainable, opaque classification model.

    All concrete sub-classes must implement :meth:`fit`, :meth:`predict_proba`,
    and :meth:`predict`. This allows the rest of the pipeline to be agnostic
    about the underlying architecture.

    Attributes
    ----------
    model : Any
        The underlying estimator object (set by each sub-class).
    is_fitted : bool
        Flag indicating whether :meth:`fit` has been called successfully.
    """

    def __init__(self) -> None:
        self.model = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int_],
    ) -> "BlackBoxModel":
        """Train the model on the provided data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y_train : np.ndarray of shape (n_samples,)
            Binary class labels (0 or 1).

        Returns
        -------
        BlackBoxModel
            The fitted instance (allows method chaining).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return class probability estimates for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Probability of each class for every sample. For binary
            classification, column index 1 corresponds to the positive class.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before this method.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        """Return hard class predictions for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class label (0 or 1) for each sample.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before this method.
        """
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet.

        Should be called at the start of :meth:`predict` and
        :meth:`predict_proba` to guard against premature inference.

        Raises
        ------
        RuntimeError
            If ``self.is_fitted`` is ``False``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling predict."
            )


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------

class XGBoostModel(BlackBoxModel):
    """Gradient-boosted tree ensemble using XGBoost.

    Wraps :class:`xgboost.XGBClassifier` with the :class:`BlackBoxModel`
    interface. Tree ensembles create highly non-linear, non-orthogonal decision
    boundaries that are a challenging testbed for local linear explainers.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded directly to :class:`xgboost.XGBClassifier`
        (e.g., ``n_estimators``, ``max_depth``, ``learning_rate``).

    Examples
    --------
    >>> m = XGBoostModel(n_estimators=200, max_depth=4)
    >>> m.fit(X_train, y_train)
    >>> proba = m.predict_proba(X_test)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            **kwargs,
        )

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int_],
    ) -> "XGBoostModel":
        """Train the XGBoost classifier.

        Delegates directly to ``XGBClassifier.fit``. Sets ``self.is_fitted``
        to ``True`` upon completion.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
        y_train : np.ndarray of shape (n_samples,)

        Returns
        -------
        XGBoostModel
            The fitted instance.
        """
        raise NotImplementedError

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return class probability estimates from the XGBoost classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        raise NotImplementedError

    def predict(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        """Return hard class predictions from the XGBoost classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        raise NotImplementedError


class NeuralNetworkModel(BlackBoxModel):
    """Multi-Layer Perceptron classifier using scikit-learn's MLPClassifier.

    Uses two fully-connected hidden layers (100 units each) with ReLU
    activations, matching the architecture commonly referenced in XAI
    benchmarking literature (e.g., OpenXAI, XAI-BENCH).

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...], optional
        Number of neurons per hidden layer. Defaults to ``(100, 100)``.
    max_iter : int, optional
        Maximum number of training iterations. Defaults to ``500``.
    random_state : int, optional
        Seed for reproducibility. Defaults to ``42``.
    **kwargs
        Additional keyword arguments forwarded to :class:`sklearn.neural_network.MLPClassifier`.

    Examples
    --------
    >>> m = NeuralNetworkModel(hidden_layer_sizes=(100, 100), max_iter=300)
    >>> m.fit(X_train, y_train)
    >>> proba = m.predict_proba(X_test)
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 100),
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int_],
    ) -> "NeuralNetworkModel":
        """Train the MLP classifier.

        Delegates directly to ``MLPClassifier.fit``. Sets ``self.is_fitted``
        to ``True`` upon completion.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
        y_train : np.ndarray of shape (n_samples,)

        Returns
        -------
        NeuralNetworkModel
            The fitted instance.
        """
        raise NotImplementedError

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return class probability estimates from the MLP.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        raise NotImplementedError

    def predict(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        """Return hard class predictions from the MLP.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        raise NotImplementedError
