import numpy as np
from sklearn.metrics import accuracy_score

from core.model import XGBoostModel, NeuralNetworkModel
from core.utility.log import ExperimentLogger
from core.data_loader import BreastCancerLoader, AdultIncomeLoader

# from core.metrics import ComplexityCalibratedConcordance

logger = ExperimentLogger.get_logger("Orchestrator")

class ExperimentOrchestrator:
    def __init__(self, dataset_name: str, k_features: int = 4):
        self.dataset_name = dataset_name
        self.k_features = k_features
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = []
        self.models = {}
        
    def _load_dataset(self) -> None:
        """Delega il caricamento dei dati all'apposito DataLoader."""
        logger.info(f"Loading local dataset: {self.dataset_name}...")
        
        if self.dataset_name == "breast_cancer":
            loader = BreastCancerLoader()
        elif self.dataset_name == "adult":
            loader = AdultIncomeLoader()
        else:
            error_msg = "Dataset non supportato. Usa 'breast_cancer' o 'adult'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Il DataLoader fa tutto il lavoro sporco (lettura, encoding, split, scaling)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.load_data()
        self.feature_names = loader.feature_names
        
        logger.info(f"  -> Dati caricati. Train: {self.X_train.shape[0]} righe | Test: {self.X_test.shape[0]} righe")

    def _train_models(self) -> None:
        """Initializes and trains the Black-Box models."""
        logger.info("Training Black-Box models...")
        self.models['XGBoost'] = XGBoostModel()
        self.models['NeuralNetwork'] = NeuralNetworkModel()
        
        for name, model in self.models.items():
            model.train(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            logger.info(f"  -> [{name}] Test Accuracy: {acc:.4f}")

    def _generate_explanations(self) -> None:
        """Placeholder for generating explanations and calculating the metric."""
        logger.info(f"Esecuzione Explainers con limite cognitivo K={self.k_features}...")
        logger.warning("  -> WORK IN PROGRESS: Implementazione metriche e XAI...")

    def run_experiment(self) -> None:
        """Executes the full experimental pipeline."""
        logger.info(f"NIZIO ESPERIMENTO: Limite Cognitivo K={self.k_features}")
        self._load_dataset()
        self._train_models()
        self._generate_explanations()
        logger.info(f"ESPERIMENTO CONCLUSO")