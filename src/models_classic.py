"""Modelos clásicos (Random Forest y XGBoost) sobre las features físicas."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

FEATURE_SETS = {
    "estadisticos": "stat_",
    "fft": "fft_",
    "bls": "bls_",
    "todas": "",
}


def select_columns(features: pd.DataFrame, family: str) -> list[str]:
    """Columnas de features de una familia ('estadisticos', 'fft', 'bls' o 'todas')."""
    prefix = FEATURE_SETS[family]
    return [c for c in features.columns if c != "label" and c.startswith(prefix)]


def make_random_forest(seed: int = 0) -> RandomForestClassifier:
    """Random Forest con ponderación de clases balanceada en cada árbol."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost que ajusta scale_pos_weight = (negativos / positivos) con los datos de entrenamiento.

    El peso se calcula en fit() para que dependa de cada fold de entrenamiento y no de todo el dataset.
    """

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed

    def fit(self, X, y):
        pos = max(int(np.sum(y == 1)), 1)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=np.sum(y == 0) / pos,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_xgboost(seed: int = 0) -> BalancedXGB:
    return BalancedXGB(seed=seed)


MODELS = {"random_forest": make_random_forest, "xgboost": make_xgboost}
