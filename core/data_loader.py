import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

class BreastCancerLoader(BaseDataLoader):
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = Path("datasets/BSWD/wdbc.data")
        if path.exists():
            df = pd.read_csv(path, header=None)
            df = df.drop(0, axis=1) # Rimuovi ID

            # M (Malignant) -> 1, B (Benign) -> 0
            y = df[1].map({'M': 1, 'B': 0}).values
            X_df = df.drop(1, axis=1)

            self.feature_names = [f"feature_{i}" for i in range(1, 31)]
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

class AdultIncomeLoader(BaseDataLoader):
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_path = Path("datasets/adult/adult.data")
        test_path = Path("datasets/adult/adult.test")

        if not data_path.exists() and not test_path.exists():
            logger.error("File non trovati: datasets/adult/adult.data o datasets/adult/adult.test")
            raise FileNotFoundError(
                "Assicurati che esista almeno uno tra datasets/adult/adult.data o datasets/adult/adult.test"
            )
            
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
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