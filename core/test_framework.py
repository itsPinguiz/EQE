import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from core.data_loader import AdultIncomeLoader, BreastCancerLoader
from core.explainers import LimeTabularExplainerWrapper, ShapExplainer
from core.metrics import build_metrics
from core.model import NeuralNetworkModel, XGBoostModel
from core.utility.log import ExperimentLogger

logger = ExperimentLogger.get_logger("Orchestrator")

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "neuralnetwork": NeuralNetworkModel,
}

EXPLAINER_REGISTRY = {
    "shap": ShapExplainer,
    "lime": LimeTabularExplainerWrapper,
}


class ExperimentOrchestrator:
    def __init__(
        self,
        dataset_name: str,
        k_features: int = 4,
        test_size: float = 0.2,
        random_state: int = 42,
        n_explain: int | None = None,
        models: Iterable[str] | None = None,
        model_params: dict[str, dict] | None = None,
        explainers: Iterable[str] | None = None,
        metrics: Iterable[str] | None = None,
        shap_background_size: int | None = 100,
        shap_background_strategy: str = "sample",
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.n_explain = n_explain
        self.models_config = list(models) if models is not None else ["xgboost", "neuralnetwork"]
        self.model_params = model_params or {}
        self.explainers = list(explainers) if explainers is not None else ["lime", "shap"]
        self.metrics = list(metrics) if metrics is not None else None
        self.shap_background_size = shap_background_size
        self.shap_background_strategy = shap_background_strategy
        self.verbose = verbose

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = []
        self.models = {}
        self.model_scores = {}

        if not self.verbose:
            logger.setLevel(logging.WARNING)
        
    def _load_dataset(self) -> None:
        """Delega il caricamento dei dati all'apposito DataLoader."""
        logger.info(f"Loading local dataset: {self.dataset_name}...")
        
        if self.dataset_name == "breast_cancer":
            loader = BreastCancerLoader(test_size=self.test_size, random_state=self.random_state)
        elif self.dataset_name == "adult":
            loader = AdultIncomeLoader(test_size=self.test_size, random_state=self.random_state)
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
        for model_name in self.models_config:
            model_key = model_name.lower()
            model_cls = MODEL_REGISTRY.get(model_key)
            if model_cls is None:
                raise ValueError(f"Modello non supportato: {model_name}")

            params = dict(self.model_params.get(model_key, {}))
            params.setdefault("random_state", self.random_state)
            model = model_cls(**params)
            self.models[model_name] = model

        for name, model in self.models.items():
            model.train(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            self.model_scores[name] = float(acc)
            logger.info(f"  -> [{name}] Test Accuracy: {acc:.4f}")

    def _generate_explanations(self) -> pd.DataFrame:
        """Generate explanations and compute CCC for each model/explainer pair."""
        logger.info(f"Esecuzione Explainers con limite cognitivo K={self.k_features}...")

        X_explain = self.X_test
        if self.n_explain is not None:
            X_explain = self.X_test[: self.n_explain]

        metrics = build_metrics(self.metrics, k_features=self.k_features)
        results = []

        for model_name, model in self.models.items():
            f_proba = model.predict_proba(X_explain)[:, 1]

            for explainer_name in self.explainers:
                explainer_key = explainer_name.lower()
                explainer_cls = EXPLAINER_REGISTRY.get(explainer_key)
                if explainer_cls is None:
                    raise ValueError(f"Explainer non supportato: {explainer_name}")

                try:
                    if explainer_key == "shap":
                        explainer = explainer_cls(
                            model,
                            background_data=self.X_train,
                            background_size=self.shap_background_size,
                            background_strategy=self.shap_background_strategy,
                        )
                    else:
                        explainer = explainer_cls(
                            model,
                            background_data=self.X_train,
                            feature_names=self.feature_names,
                        )

                    weights, intercepts = explainer.explain(X_explain)
                    metric_scores = {}
                    for metric in metrics:
                        metric_scores[metric.name] = metric.compute(
                            f_proba=f_proba,
                            weights=weights,
                            intercepts=intercepts,
                            X=X_explain,
                        )
                except Exception as exc:
                    logger.error(
                        f"  -> [{model_name} | {explainer_name}] errore durante l'explain: {exc}"
                    )
                    continue

                results.append(
                    {
                        "dataset": self.dataset_name,
                        "model": model_name,
                        "explainer": explainer_name,
                        "k_features": self.k_features,
                        "n_explain": X_explain.shape[0],
                        "accuracy": self.model_scores.get(model_name),
                        **metric_scores,
                    }
                )

                for metric_name, score in metric_scores.items():
                    logger.info(
                        f"  -> [{model_name} | {explainer_name}] {metric_name}: {score:.6f}"
                    )

        return pd.DataFrame(results)

    def run_experiment(self) -> pd.DataFrame:
        """Executes the full experimental pipeline."""
        logger.info(f"INIZIO ESPERIMENTO: Limite Cognitivo K={self.k_features}")
        self._load_dataset()
        self._train_models()
        results = self._generate_explanations()
        logger.info("ESPERIMENTO CONCLUSO")
        return results