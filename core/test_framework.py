import time
from joblib import Parallel, delayed
import multiprocessing
import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from core.data_loader import AdultIncomeLoader, BreastCancerLoader
from core.explainers import LimeTabularExplainerWrapper, MapleExplainer, ShapExplainer
from core.metrics import build_metrics
from core.model import BlackBoxModel, NeuralNetworkModel, XGBoostModel
from core.utility.log import ExperimentLogger
from core.utility.progress import progress_bar, tqdm_joblib

run_logger = ExperimentLogger.get_logger("Run")
data_logger = ExperimentLogger.get_logger("Data")
model_logger = ExperimentLogger.get_logger("Model")
explain_logger = ExperimentLogger.get_logger("Explain")
metric_logger = ExperimentLogger.get_logger("Metric")

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "neuralnetwork": NeuralNetworkModel,
}

EXPLAINER_REGISTRY = {
    "shap": ShapExplainer,
    "lime": LimeTabularExplainerWrapper,
    "maple": MapleExplainer,
}

HIGHER_IS_BETTER_METRICS = {"comprehensiveness_abs_drop"}


def _apply_log_level(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    for stage_logger in (
        run_logger,
        data_logger,
        model_logger,
        explain_logger,
        metric_logger,
    ):
        stage_logger.setLevel(level)


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
        explainer_params: dict[str, dict] | None = None,
        metrics: Iterable[str] | None = None,
        shap_background_size: int | None = None,
        shap_background_strategy: str | None = None,
        n_jobs: int | None = None,
        verbose: bool = True,
        show_progress: bool | None = None,
        progress_queue=None,
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
        self.explainer_params = explainer_params or {}
        self.metrics = list(metrics) if metrics is not None else None
        self.shap_background_size = shap_background_size
        self.shap_background_strategy = shap_background_strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = (not verbose) if show_progress is None else show_progress
        self.progress_queue = progress_queue

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_names = []
        self.models = {}
        self.model_scores = {}
        self.feature_baseline = None

        _apply_log_level(self.verbose)

    def _emit_progress(self, event_type: str, amount: int = 1, label: str | None = None) -> None:
        if self.progress_queue is not None:
            event = {"type": event_type, "amount": amount}
            if label:
                event["label"] = label
            self.progress_queue.put(event)
        
    def _load_dataset(self) -> None:
        """Delega il caricamento dei dati all'apposito DataLoader."""
        data_logger.info(f"Loading dataset | name={self.dataset_name}")
        
        if self.dataset_name == "breast_cancer":
            loader = BreastCancerLoader(test_size=self.test_size, random_state=self.random_state)
        elif self.dataset_name == "adult":
            loader = AdultIncomeLoader(test_size=self.test_size, random_state=self.random_state)
        else:
            error_msg = "Dataset non supportato. Usa 'breast_cancer' o 'adult'."
            data_logger.error(error_msg)
            raise ValueError(error_msg)

        # Il DataLoader fa tutto il lavoro sporco (lettura, encoding, split, scaling)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.load_data()
        self.feature_names = loader.feature_names
        self.feature_baseline = np.mean(self.X_train, axis=0)
        
        data_logger.info(
            "Loaded dataset | "
            f"train={self.X_train.shape[0]} | "
            f"test={self.X_test.shape[0]} | "
            f"features={len(self.feature_names)}"
        )
        self._emit_progress("stage", label=f"{self.dataset_name}/{self.random_state} data")

    def _train_models(self) -> None:
        """Initializes and trains the Black-Box models."""
        model_logger.info(f"Training models | count={len(self.models_config)}")
        for model_name in self.models_config:
            model_key = model_name.lower()
            model_cls = MODEL_REGISTRY.get(model_key)
            if model_cls is None:
                raise ValueError(f"Modello non supportato: {model_name}")

            params = dict(self.model_params.get(model_key, {}))
            params.setdefault("random_state", self.random_state)
            model = model_cls(**params)
            self.models[model_key] = model

        model_items = progress_bar(
            self.models.items(),
            total=len(self.models),
            desc=f"{self.dataset_name}/{self.random_state} models",
            disable=not self.show_progress,
            unit="model",
        )
        for name, model in model_items:
            model.train(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            self.model_scores[name] = float(acc)
            model_logger.info(f"Trained | model={name} | accuracy={acc:.4f}")
            self._emit_progress(
                "stage",
                label=f"{self.dataset_name}/{self.random_state} {name}",
            )

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

        params = dict(self.explainer_params.get(explainer_key, {}))

        if explainer_key == "shap":
            params.setdefault(
                "background_size",
                100 if self.shap_background_size is None else self.shap_background_size,
            )
            params.setdefault(
                "background_strategy",
                "sample"
                if self.shap_background_strategy is None
                else self.shap_background_strategy,
            )
            return explainer_cls(
                model,
                background_data=self.X_train,
                **params,
            )

        if explainer_key == "maple":
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs or 1)
            params.setdefault("show_progress", self.show_progress and self.n_jobs == 1)
            params.setdefault("progress_queue", self.progress_queue)
            return explainer_cls(
                model,
                background_data=self.X_train,
                **params,
            )

        # Default wrapper (LIME-like)
        params.setdefault("n_jobs", self.n_jobs or 1)
        params.setdefault("show_progress", self.show_progress and self.n_jobs == 1)
        params.setdefault("progress_queue", self.progress_queue)
        return explainer_cls(
            model,
            background_data=self.X_train,
            feature_names=self.feature_names,
            **params,
        )

    def _explain_single_pair(
        self,
        model_name: str,
        model: BlackBoxModel,
        explainer_name: str,
        X_explain: np.ndarray,
    ) -> tuple[str, str, dict | None]:
        """Generate explanations for a single (model, explainer) pair.

        Returns a tuple (model_name, explainer_name, payload) where payload is None on error.
        This function is designed to be called in parallel via joblib.
        """
        _apply_log_level(self.verbose)
        try:
            self._emit_progress(
                "active",
                label=f"{self.dataset_name}/{self.random_state} {model_name}/{explainer_name}",
            )
            started_at = time.perf_counter()
            explain_logger.info(
                "Started | "
                f"model={model_name} | "
                f"explainer={explainer_name}"
            )
            explainer = self._init_explainer(explainer_name, model)
            weights, intercepts = explainer.explain(X_explain)
            elapsed = time.perf_counter() - started_at
            explain_logger.info(
                "Finished | "
                f"model={model_name} | "
                f"explainer={explainer_name} | "
                f"shape={weights.shape[0]}x{weights.shape[1]} | "
                f"time={elapsed:.1f}s"
            )
            if explainer_name.lower() == "shap":
                self._emit_progress(
                    "sample",
                    amount=X_explain.shape[0],
                    label="SHAP",
                )
            self._emit_progress(
                "stage",
                label=f"{self.dataset_name}/{self.random_state} {model_name}/{explainer_name}",
            )
            return model_name, explainer_name, {
                "weights": weights,
                "intercepts": intercepts,
            }
        except Exception as exc:
            explain_logger.error(
                "Explanation error | "
                f"model={model_name} | "
                f"explainer={explainer_name} | "
                f"error={exc}"
            )
            return model_name, explainer_name, None

    def generate_explanations(self, X_explain: np.ndarray) -> dict:
        """Generate and return raw explanations for all model/explainer pairs.

        Returns a dict keyed by (model_name, explainer_name) with values:
            {
                "weights": np.ndarray (n_samples, n_features),
                "intercepts": np.ndarray (n_samples,),
                "f_proba": np.ndarray (n_samples,),
            }
        """
        explain_logger.info(
            "Generating explanations | "
            f"models={len(self.models)} | "
            f"explainers={len(self.explainers)} | "
            f"samples={X_explain.shape[0]} | "
            f"cache=true"
        )

        # Pre-compute f_proba for all models (shared across explainers)
        f_probas: dict[str, np.ndarray] = {}
        for model_name, model in self.models.items():
            try:
                f_probas[model_name] = model.predict_proba(X_explain)[:, 1]
            except Exception as exc:
                explain_logger.error(f"Probability error | model={model_name} | error={exc}")

        # Build task list: one task per (model, explainer) pair
        tasks = [
            (model_name, model, explainer_name, X_explain)
            for model_name, model in self.models.items()
            for explainer_name in self.explainers
        ]
        self._emit_progress("sample_total", amount=len(tasks) * X_explain.shape[0])

        # Parallelize at the outer level (model, explainer pairs)
        n_jobs = self.n_jobs or -1  # -1 = all cores
        progress = progress_bar(
            total=len(tasks),
            desc=f"{self.dataset_name}/{self.random_state} explainers",
            disable=not self.show_progress,
            unit="pair",
        )
        with tqdm_joblib(progress):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._explain_single_pair)(model_name, model, explainer_name, X_explain)
                for model_name, model, explainer_name, X_explain in tasks
            )

        explanations: dict = {}
        for model_name, explainer_name, payload in results:
            if payload is not None:
                explanations[(model_name, explainer_name)] = {
                    "weights": payload["weights"],
                    "intercepts": payload["intercepts"],
                    "f_proba": f_probas.get(model_name),
                }

        return explanations

    def compute_metrics(self, explanations: dict, X_explain: np.ndarray, current_k: int) -> pd.DataFrame:
        """Compute configured metrics for the supplied explanations dict.

        This function is independent from how explanations were produced and
        can be re-used to compute different K or metric sets against cached
        explanations.
        """
        metrics = build_metrics(
            self.metrics,
            k_features=current_k,
            random_state=self.random_state,
        )
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
                        model=self.models.get(model_name),
                        baseline=self.feature_baseline,
                    )
                except Exception as exc:
                    metric_logger.error(
                        "Metric error | "
                        f"model={model_name} | "
                        f"explainer={explainer_name} | "
                        f"metric={metric.name} | "
                        f"K={current_k} | "
                        f"error={exc}"
                    )
                    metric_scores[metric.name] = float("nan")

            results.append(
                {
                    "dataset": self.dataset_name,
                    "model": model_name,
                    "explainer": explainer_name,
                    "seed": self.random_state,
                    "k_features": current_k,
                    "n_explain": X_explain.shape[0],
                    "accuracy": self.model_scores.get(model_name),
                    **metric_scores,
                }
            )

            for metric_name, score in metric_scores.items():
                metric_logger.debug(
                    "Score | "
                    f"K={current_k} | "
                    f"model={model_name} | "
                    f"explainer={explainer_name} | "
                    f"{metric_name}={score:.6f}"
                )

        return pd.DataFrame(results)

    def _log_metric_summary(self, result_df: pd.DataFrame, current_k: int) -> None:
        if result_df.empty:
            metric_logger.warning(f"No metric rows produced | K={current_k}")
            return

        summary_parts = []
        for metric_name in self._metric_result_columns(result_df):
            if metric_name in HIGHER_IS_BETTER_METRICS:
                best_idx = result_df[metric_name].idxmax()
                direction = "max"
            else:
                best_idx = result_df[metric_name].idxmin()
                direction = "min"
            best_row = result_df.loc[best_idx]
            summary_parts.append(
                f"{metric_name}: best_{direction}={best_row[metric_name]:.6f} "
                f"({best_row['model']}/{best_row['explainer']})"
            )

        metric_logger.info(f"K={current_k} summary | " + " | ".join(summary_parts))

    def _metric_result_columns(self, result_df: pd.DataFrame) -> list[str]:
        metadata_columns = {
            "dataset",
            "model",
            "explainer",
            "seed",
            "k_features",
            "n_explain",
            "accuracy",
        }
        return [column for column in result_df.columns if column not in metadata_columns]

    def run_experiment(self) -> pd.DataFrame:
        """Executes the full experimental pipeline with Caching and K-loop optimization."""
        run_logger.info(
            "Starting experiment | "
            f"dataset={self.dataset_name} | "
            f"seed={self.random_state} | "
            f"K={self.k_features_list} | "
            f"models={','.join(self.models_config)} | "
            f"explainers={','.join(self.explainers)}"
        )
        self._load_dataset()
        self._train_models()
        
        # Data Extraction for Explanation
        X_explain = self.X_test
        if self.n_explain is not None:
            X_explain = self.X_test[: self.n_explain]
            
        # 1. Contribution caching: generate explainer outputs once
        explanations_cache = self.generate_explanations(X_explain)
        
        # 2. K-Loop Optimization (Instant Compute iterating over cache)
        all_results = []
        k_iter = progress_bar(
            self.k_features_list,
            desc=f"{self.dataset_name}/{self.random_state} metrics",
            disable=not self.show_progress,
            unit="K",
        )
        for current_k in k_iter:
            k_result_df = self.compute_metrics(explanations_cache, X_explain, current_k)
            self._log_metric_summary(k_result_df, current_k)
            all_results.append(k_result_df)
            self._emit_progress(
                "stage",
                label=f"{self.dataset_name}/{self.random_state} K={current_k}",
            )
            
        final_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        run_logger.info(f"Experiment concluded | rows={final_results.shape[0]}")
        return final_results
