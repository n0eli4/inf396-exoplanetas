"""Carga de los datos de curvas de luz de Kepler."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(name: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Carga 'train' o 'test'.

    Devuelve (X, y):
      X: array (n_estrellas, 3197) con el flujo de cada estrella.
      y: array (n_estrellas,) con 1 = tiene exoplaneta, 0 = no tiene.
         (El CSV original usa 2 y 1 respectivamente; aquí se pasa a 1 y 0.)
    """
    path = data_dir / f"exo{name.capitalize()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ver data/README.md para descargarlo.")
    df = pd.read_csv(path)
    y = (df["LABEL"].to_numpy() == 2).astype(int)
    X = df.drop(columns="LABEL").to_numpy(dtype=np.float64)
    return X, y


def load_train_test() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve X_train, y_train, X_test, y_test."""
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_test, y_test
