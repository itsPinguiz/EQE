from abc import ABC, abstractmethod
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class BlackBoxModel(ABC):
    """
    Abstract base class for all black-box models to be explained.
    """
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the black-box model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        pass


class XGBoostModel(BlackBoxModel):
    def __init__(self, random_state: int = 42, **kwargs):
        # eval_metric='logloss' avoids warnings in newer XGBoost versions
        self.model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            **kwargs,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class NeuralNetworkModel(BlackBoxModel):
    def __init__(self, random_state: int = 42, **kwargs):
        # A simple Multi-Layer Perceptron (2 hidden layers of 100 nodes)
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            max_iter=500,
            random_state=random_state,
            **kwargs,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)