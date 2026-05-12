from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import warnings

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
        pass


class ShapExplainer(BaseExplainer):
    """Wrapper for SHAP (KernelExplainer mode)."""
    
    def __init__(
        self,
        model: BlackBoxModel,
        background_data: npt.NDArray[np.float64],
        background_size: int | None = 100,
        background_strategy: str = "sample",
    ):
        super().__init__(model)
        if shap is None:
            raise ImportError("Please install 'shap' library to use ShapExplainer.")

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
            
        # We use KernelExplainer. We wrap the predict_proba to only return probabilities for class 1 (positive class).
        self.explainer = shap.KernelExplainer(
            lambda x: self.model.predict_proba(x)[:, 1], 
            background
        )
        
    def explain(self, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Calculate SHAP values (feature attributions)
        shap_out = self.explainer.shap_values(X)
        weights = np.array(shap_out)
        
        # In SHAP, the intercept is the 'expected_value' across the background dataset.
        base_value = float(self.explainer.expected_value)
        
        # Replicate the base_value for all test instances
        intercepts = np.full(X.shape[0], base_value)
        
        return weights, intercepts


class LimeTabularExplainerWrapper(BaseExplainer):
    """Wrapper for LIME (Local Interpretable Model-agnostic Explanations)."""
    
    def __init__(self, model: BlackBoxModel, background_data: npt.NDArray[np.float64], feature_names: list[str] = None):
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
        
    def explain(self, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n_samples, n_features = X.shape
        weights = np.zeros((n_samples, n_features))
        intercepts = np.zeros(n_samples)
        
        # LIME can only explain one instance at a time, so we must iterate
        for i in range(n_samples):
            # LIME requires the full predict_proba function (returning all classes)
            exp = self.explainer.explain_instance(
                data_row=X[i],
                predict_fn=self.model.predict_proba,
                num_features=n_features # Request weights for all features
            )
            
            # Extract weights for the positive class (assuming index 1)
            for feature_idx, weight in exp.as_map()[1]:
                weights[i, feature_idx] = weight
            
            # The intercept is stored inside LIME's internal surrogate linear model
            intercepts[i] = exp.intercept[1]
            
        return weights, intercepts


