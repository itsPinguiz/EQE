from abc import ABC, abstractmethod
import os
import numpy as np
import numpy.typing as npt
import warnings
from sklearn.model_selection import train_test_split

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Try importing XAI libraries safely so the module doesn't crash if one is missing during dev
try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None

from core.model import BlackBoxModel
from core.third_party.MAPLE import MAPLE

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.
    Ensures that every explainer produces attribution weights and intercepts
    in a standardized format expected by the ComplexityCalibratedConcordance metric.
    """
    
    def __init__(self, model: BlackBoxModel):
        self.model = model
        
    @abstractmethod
    def explain(self, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generates feature attribution weights and base intercepts for the given instances.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The test instances to explain.
            
        Returns
        -------
        weights : np.ndarray of shape (n_samples, n_features)
            The attribution assigned to each feature for each instance.
        intercepts : np.ndarray of shape (n_samples,)
            The base predictive value (w_0) for each instance before adding feature contributions.
        """
        raise NotImplementedError("Subclasses must implement the explain method.")


class ShapExplainer(BaseExplainer):
    """Wrapper for SHAP (KernelExplainer mode)."""
    
    def __init__(
        self,
        model: BlackBoxModel,
        background_data: npt.NDArray[np.float64],
        background_size: int | None = 100,
        background_strategy: str = "sample",
        silent: bool = True,
    ):
        super().__init__(model)
        if shap is None:
            raise ImportError("Please install 'shap' library to use ShapExplainer.")
        self.silent = silent

        if background_size is not None and background_size < 1:
            raise ValueError("background_size must be >= 1 or None.")

        background = background_data
        if background_size is not None and background_data.shape[0] > background_size:
            if background_strategy == "kmeans":
                background = shap.kmeans(background_data, background_size)
            elif background_strategy == "sample":
                background = shap.sample(background_data, background_size)
            else:
                raise ValueError("background_strategy must be 'sample' or 'kmeans'.")
        
        # Convert memmap to regular array for SHAP compatibility
        background = np.asarray(background)
        
        # We need a named instance method to allow ProcessPoolExecutor pickling
        self.explainer = shap.KernelExplainer(
            self._predict_proba_pos_class, 
            background
        )

    def _predict_proba_pos_class(self, x):
        """Wrapper method to return only positive class probabilities.
        Implemented as an instance method to allow multiprocessing pickling."""
        # Convert memmap to regular array for SHAP compatibility
        x = np.asarray(x)
        return self.model.predict_proba(x)[:, 1]
        
    def explain(self, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Convert memmap to regular array for SHAP compatibility
        X = np.asarray(X)
        # Calculate SHAP values (feature attributions)
        try:
            shap_out = self.explainer.shap_values(X, silent=self.silent)
        except TypeError:
            shap_out = self.explainer.shap_values(X)
        weights = self._normalize_shap_values(shap_out)
        
        # In SHAP, the intercept is the 'expected_value' across the background dataset.
        base_value = self._normalize_expected_value(self.explainer.expected_value)
        
        # Replicate the base_value for all test instances
        intercepts = np.full(X.shape[0], base_value)
        
        return weights, intercepts

    def _normalize_shap_values(
        self,
        shap_values: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        if isinstance(shap_values, list):
            if len(shap_values) < 2:
                raise ValueError("SHAP values list is empty or has only one class.")
            values = shap_values[1]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                if values.shape[0] < 2:
                    raise ValueError("SHAP values array has only one class.")
                values = values[1]

        if values.ndim != 2:
            raise ValueError("SHAP values must have shape (n_samples, n_features).")

        return np.asarray(values, dtype=float)

    def _normalize_expected_value(self, expected_value) -> float:
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            if len(expected_value) < 2:
                raise ValueError("SHAP expected_value has only one class.")
            return float(expected_value[1])
        return float(expected_value)


class LimeTabularExplainerWrapper(BaseExplainer):
    """Wrapper for LIME (Local Interpretable Model-agnostic Explanations)."""
    
    def __init__(self, model: BlackBoxModel, background_data: npt.NDArray[np.float64], feature_names: list[str] = None, n_jobs: int = 1):
        super().__init__(model)
        if lime is None:
            raise ImportError("Please install 'lime' library to use LimeTabularExplainerWrapper.")
            
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(background_data.shape[1])]
            
        # Initialize LIME. It requires the training data to compute feature statistics.
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_data,
            feature_names=feature_names,
            mode='classification'
        )
        self.n_jobs = n_jobs
        
    def _explain_single(self, i: int, x_row: np.ndarray, n_features: int) -> tuple[np.ndarray, float]:
        """Explain a single instance. Used for parallel execution."""
        exp = self.explainer.explain_instance(
            data_row=x_row,
            predict_fn=self.model.predict_proba,
            num_features=n_features
        )
        
        weights = np.zeros(n_features)
        for feature_idx, weight in exp.as_map()[1]:
            weights[feature_idx] = weight
        
        return weights, exp.intercept[1]
        
    def explain(self, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n_samples, n_features = X.shape
        weights = np.zeros((n_samples, n_features))
        intercepts = np.zeros(n_samples)
        
        # Use joblib for parallel processing if n_jobs > 1
        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._explain_single)(i, X[i], n_features)
                for i in range(n_samples)
            )
            for i, (w, intercept) in enumerate(results):
                weights[i] = w
                intercepts[i] = intercept
        else:
            # Sequential fallback
            for i in range(n_samples):
                weights[i], intercepts[i] = self._explain_single(i, X[i], n_features)
            
        return weights, intercepts


class MapleExplainer(BaseExplainer):
    """Wrapper for the official MAPLE implementation bundled in third_party.

    MAPLE returns local linear coefficients. The project metric expects
    per-feature additive contributions, so this wrapper converts each local
    coefficient beta_i into beta_i * x_i and keeps MAPLE's local intercept.
    """

    def __init__(
        self,
        model: BlackBoxModel,
        background_data: npt.NDArray[np.float64],
        validation_size: float | int = 100,
        training_size: int | None = None,
        fe_type: str = "rf",
        n_estimators: int = 200,
        max_features: float = 0.5,
        min_samples_leaf: int = 10,
        regularization: float = 0.001,
        random_state: int = 42,
        n_jobs: int = 1,
        **_: object,
    ):
        super().__init__(model)
        self.random_state = random_state
        self.n_jobs = n_jobs

        X_train, X_val = self._split_background(
            background_data=background_data,
            validation_size=validation_size,
            training_size=training_size,
            random_state=random_state,
        )

        mr_train = self.model.predict_proba(X_train)[:, 1]
        mr_val = self.model.predict_proba(X_val)[:, 1]

        # The official MAPLE code does not expose random_state in its forest
        # constructors. Seeding numpy before construction makes sklearn's
        # random_state=None path reproducible.
        np.random.seed(random_state)
        self.explainer = MAPLE(
            X_train=X_train,
            MR_train=mr_train,
            X_val=X_val,
            MR_val=mr_val,
            fe_type=fe_type,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            regularization=regularization,
        )

    def _split_background(
        self,
        background_data: npt.NDArray[np.float64],
        validation_size: float | int,
        training_size: int | None,
        random_state: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if background_data.shape[0] < 2:
            raise ValueError("MAPLE needs at least two background rows.")

        if isinstance(validation_size, float):
            if not 0 < validation_size < 1:
                raise ValueError("MAPLE validation_size as a float must be between 0 and 1.")
            test_size = validation_size
        else:
            if validation_size < 1:
                raise ValueError("MAPLE validation_size as an int must be >= 1.")
            test_size = min(validation_size, background_data.shape[0] - 1)

        X_maple_train, X_val = train_test_split(
            background_data,
            test_size=test_size,
            random_state=random_state,
        )

        if training_size is not None:
            if training_size < 1:
                raise ValueError("MAPLE training_size must be >= 1 or null.")
            if training_size < X_maple_train.shape[0]:
                rng = np.random.default_rng(random_state)
                train_indices = rng.choice(
                    X_maple_train.shape[0],
                    size=training_size,
                    replace=False,
                )
                X_maple_train = X_maple_train[train_indices]

        return X_maple_train, X_val

    def explain(
        self,
        X: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n_samples, n_features = X.shape
        weights = np.zeros((n_samples, n_features))
        intercepts = np.zeros(n_samples)

        # Use joblib for parallel processing if n_jobs > 1
        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._explain_single)(i, X[i])
                for i in range(n_samples)
            )
            for i, (w, intercept) in enumerate(results):
                weights[i] = w
                intercepts[i] = intercept
        else:
            # Sequential fallback
            for i in range(n_samples):
                exp = self.explainer.explain(X[i])
                coefs = np.asarray(exp["coefs"], dtype=float)
                intercepts[i] = coefs[0]
                weights[i] = coefs[1:] * X[i]

        return weights, intercepts

    def _explain_single(self, i: int, x_row: np.ndarray) -> tuple[np.ndarray, float]:
        """Explain a single instance. Used for parallel execution."""
        exp = self.explainer.explain(x_row)
        coefs = np.asarray(exp["coefs"], dtype=float)
        return coefs[1:] * x_row, float(coefs[0])
