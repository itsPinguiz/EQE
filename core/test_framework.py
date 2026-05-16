import concurrent.futures
import multiprocessing
import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing
import logging
from typing import Iterable, Sequence

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
        k_features: Sequence[int] | int = 4,
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
        # Accept a single K or a list of K values
        self.k_features_list = [k_features] if isinstance(k_features, int) else list(k_features)
        
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
            self.models[model_key] = model

        for name, model in self.models.items():
            model.train(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            self.model_scores[name] = float(acc)
            logger.info(f"  -> [{name}] Test Accuracy: {acc:.4f}")

    def _generate_explanations(self) -> pd.DataFrame:
        """Deprecated: kept for compatibility.

        Use `generate_explanations` + `compute_metrics` instead which split the
        responsibilities: one function only generates explainers' outputs while
        the other computes and formats metric results.
        """
        X_explain = self.X_test
        if self.n_explain is not None:
            X_explain = self.X_test[: self.n_explain]

        # Backwards-compatible wrapper that uses the new split flow.
        explanations = self.generate_explanations(X_explain)
        return self.compute_metrics(explanations, X_explain, current_k=self.k_features_list[-1])

    def _init_explainer(self, explainer_key: str, model):
        """Factory that initializes explainers with the correct arguments.

        Centralising explainer initialization keeps `_generate_explanations`
        clean and allows explainers to evolve without touching the
        orchestrator logic.
        """
        explainer_key = explainer_key.lower()
        explainer_cls = EXPLAINER_REGISTRY.get(explainer_key)
        if explainer_cls is None:
            raise ValueError(f"Explainer non supportato: {explainer_key}")

        if explainer_key == "shap":
            return explainer_cls(
                model,
                background_data=self.X_train,
                background_size=self.shap_background_size,
                background_strategy=self.shap_background_strategy,
            )

        # Default wrapper (LIME-like)
        return explainer_cls(
            model,
            background_data=self.X_train,
            feature_names=self.feature_names,
        )

    def generate_explanations(self, X_explain: np.ndarray) -> dict:
        """Generate and return raw explanations for all model/explainer pairs.

        Returns a dict keyed by (model_name, explainer_name) with values:
            {
                "weights": np.ndarray (n_samples, n_features),
                "intercepts": np.ndarray (n_samples,),
                "f_proba": np.ndarray (n_samples,),
            }
        """
        logger.info(f"Generating explanations in parallel (Caching)...")

        explanations: dict = {}

        for model_name, model in self.models.items():
            try:
                f_proba = model.predict_proba(X_explain)[:, 1]
            except Exception as exc:
                logger.error(f"Error calculating probabilities for {model_name}: {exc}")
                continue

            for explainer_name in self.explainers:
                try:
                    explainer = self._init_explainer(explainer_name, model)
                    
                    # 1. PARALLELIZATION (True Multi-Processing, Joblib fallback)
                    # Split X_explain into chunks based on the number of available CPU cores
                    n_cores = multiprocessing.cpu_count() or 4
                    chunks = np.array_split(X_explain, n_cores)
                    
                    # Using joblib.Parallel allows serializing lambdas (used natively by LIME)
                    # bounding deadlocks and pickling problems automatically.
                    results = Parallel(n_jobs=n_cores, backend="loky")(
                        delayed(explainer.explain)(chunk) for chunk in chunks
                    )
                    
                    # Reassemble the results keeping the original order
                    weights = np.vstack([res[0] for res in results])
                    intercepts = np.concatenate([res[1] for res in results])
                    
                except Exception as exc:
                    logger.error(f"  -> [{model_name} | {explainer_name}] error during explain: {exc}")
                    continue

                explanations[(model_name, explainer_name)] = {
                    "weights": weights,
                    "intercepts": intercepts,
                    "f_proba": f_proba,
                }

        return explanations

    def compute_metrics(self, explanations: dict, X_explain: np.ndarray, current_k: int) -> pd.DataFrame:
        """Compute configured metrics for the supplied explanations dict.

        This function is independent from how explanations were produced and
        can be re-used to compute different K or metric sets against cached
        explanations.
        """
        metrics = build_metrics(self.metrics, k_features=current_k)
        results = []

        for (model_name, explainer_name), payload in explanations.items():
            weights = payload["weights"]
            intercepts = payload["intercepts"]
            f_proba = payload["f_proba"]

            metric_scores = {}
            for metric in metrics:
                try:
                    metric_scores[metric.name] = metric.compute(
                        f_proba=f_proba,
                        weights=weights,
                        intercepts=intercepts,
                        X=X_explain,
                    )
                except Exception as exc:
                    logger.error(f"  -> [{model_name} | {explainer_name}] errore nel calcolo della metrica {metric.name} (K={current_k}): {exc}")
                    metric_scores[metric.name] = float("nan")

            results.append(
                {
                    "dataset": self.dataset_name,
                    "model": model_name,
                    "explainer": explainer_name,
                    "k_features": current_k,
                    "n_explain": X_explain.shape[0],
                    "accuracy": self.model_scores.get(model_name),
                    **metric_scores,
                }
            )

            for metric_name, score in metric_scores.items():
                logger.info(f"  -> [{model_name} | {explainer_name}] {metric_name} (K={current_k}): {score:.6f}")

        return pd.DataFrame(results)

    def run_experiment(self) -> pd.DataFrame:
        """Executes the full experimental pipeline with Caching and K-loop optimization."""
        logger.info(f"STARTING EXPERIMENT: Cognitive Limits K={self.k_features_list}")
        self._load_dataset()
        self._train_models()
        
        # Data Extraction for Explanation
        X_explain = self.X_test
        if self.n_explain is not None:
            X_explain = self.X_test[: self.n_explain]
            
        # 1. Weight Caching: Generate SHAP/LIME ONCE
        explanations_cache = self.generate_explanations(X_explain)
        
        # 2. K-Loop Optimization (Instant Compute iterating over cache)
        all_results = []
        for current_k in self.k_features_list:
            logger.info(f"=== Computing metrics on cached weights for K={current_k} ===")
            k_result_df = self.compute_metrics(explanations_cache, X_explain, current_k)
            all_results.append(k_result_df)
            
        final_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        logger.info("EXPERIMENT CONCLUDED")
        return final_results