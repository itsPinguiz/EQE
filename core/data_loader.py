import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.utility.log import ExperimentLogger

logger = ExperimentLogger.get_logger("DataLoader")

class BaseDataLoader(ABC):
    """Classe base astratta per caricare e preprocessare i dataset."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: List[str] = []

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Deve restituire (X_train, X_test, y_train, y_test) già preprocessati e scalati.
        """
        pass

    def _split_and_scale(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Metodo di utility comune per dividere e scalare i dati."""
        self._validate_feature_names(X.shape[1])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def _validate_feature_names(self, n_features: int) -> None:
        if len(self.feature_names) != n_features:
            raise ValueError(
                f"Feature names mismatch: {len(self.feature_names)} names for {n_features} columns."
            )

        generic_names = {
            name
            for name in self.feature_names
            if re.fullmatch(r"feature[_ ]?\d+", name.strip(), flags=re.IGNORECASE)
        }
        if generic_names:
            preview = ", ".join(sorted(generic_names)[:5])
            raise ValueError(f"Generic feature names are not allowed: {preview}")

class BreastCancerLoader(BaseDataLoader):
    WDBC_NAMES_PATH = Path("datasets/BSWD/wdbc.names")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = Path("datasets/BSWD/wdbc.data")
        if path.exists():
            df = pd.read_csv(path, header=None)
            df = df.drop(0, axis=1) # Rimuovi ID

            # M (Malignant) -> 1, B (Benign) -> 0
            y = df[1].map({'M': 1, 'B': 0}).values
            X_df = df.drop(1, axis=1)

            self.feature_names = self._read_wdbc_feature_names()
            X = X_df.values
        else:
            logger.warning(
                f"File non trovato: {path}. Uso sklearn.datasets.load_breast_cancer()"
            )
            dataset = load_breast_cancer(as_frame=True)
            X_df = dataset.data
            y = dataset.target.values
            self.feature_names = dataset.feature_names.tolist()
            X = X_df.values

        return self._split_and_scale(X, y)

    def _read_wdbc_feature_names(self) -> list[str]:
        if not self.WDBC_NAMES_PATH.exists():
            raise FileNotFoundError(f"WDBC metadata file not found: {self.WDBC_NAMES_PATH}")

        text = self.WDBC_NAMES_PATH.read_text(encoding="utf-8")
        feature_section = text.split(
            "Ten real-valued features are computed for each cell nucleus:",
            maxsplit=1,
        )[-1].split("Several of the papers listed above", maxsplit=1)[0]
        base_features = re.findall(
            r"^\s+[a-j]\)\s+([^(]+?)(?:\s+\(|\s*$)",
            feature_section,
            re.MULTILINE,
        )
        base_features = [feature.strip().lower() for feature in base_features]

        if len(base_features) != 10:
            raise ValueError(
                f"Expected 10 WDBC base features from metadata, found {len(base_features)}."
            )

        return [
            f"{stat} {feature}"
            for stat in ("mean", "se", "worst")
            for feature in base_features
        ]

class AdultIncomeLoader(BaseDataLoader):
    ADULT_NAMES_PATH = Path("datasets/adult/adult.names")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_path = Path("datasets/adult/adult.data")
        test_path = Path("datasets/adult/adult.test")

        if not data_path.exists() and not test_path.exists():
            logger.error("File non trovati: datasets/adult/adult.data o datasets/adult/adult.test")
            raise FileNotFoundError(
                "Assicurati che esista almeno uno tra datasets/adult/adult.data o datasets/adult/adult.test"
            )
            
        columns = self._read_adult_columns()
        
        frames = []
        if data_path.exists():
            frames.append(self._read_adult_file(data_path, columns))
        if test_path.exists():
            frames.append(self._read_adult_file(test_path, columns, skip_header=True))

        df = pd.concat(frames, ignore_index=True).dropna()
        
        # >50K -> 1, <=50K -> 0
        y = (df['income'].str.strip().str.replace('.', '', regex=False) == '>50K').astype(int).values
        X_df = df.drop('income', axis=1)
        
        X_df = pd.get_dummies(X_df, drop_first=True)
        self.feature_names = X_df.columns.tolist()
        X = X_df.values
        
        return self._split_and_scale(X, y)

    def _read_adult_file(
        self,
        path: Path,
        columns: list[str],
        skip_header: bool = False,
    ) -> pd.DataFrame:
        return pd.read_csv(
            path,
            header=0 if skip_header else None,
            names=columns,
            na_values=" ?",
            skipinitialspace=True,
        )

    def _read_adult_columns(self) -> list[str]:
        if not self.ADULT_NAMES_PATH.exists():
            raise FileNotFoundError(f"Adult metadata file not found: {self.ADULT_NAMES_PATH}")

        feature_columns = []
        for line in self.ADULT_NAMES_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|") or ":" not in stripped:
                continue
            feature_columns.append(stripped.split(":", maxsplit=1)[0])

        if len(feature_columns) != 14:
            raise ValueError(
                f"Expected 14 Adult feature columns from metadata, found {len(feature_columns)}."
            )

        return [*feature_columns, "income"]
