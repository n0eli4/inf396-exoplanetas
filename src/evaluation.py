"""Métricas y protocolo de evaluación robusto al desbalance de clases."""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

METRICS = ("recall", "precision", "f1", "pr_auc")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula recall, precisión, F1 y PR-AUC.

    y_score: probabilidad (o puntaje) de la clase positiva.
    threshold: umbral para convertir el puntaje en predicción 0/1 (usado en recall, precisión y F1).
    PR-AUC no depende del umbral: resume todos los umbrales posibles.
    """
    y_pred = (y_score >= threshold).astype(int)
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def repeated_cv(
    make_model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    threshold: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validación cruzada estratificada repetida.

    make_model: función sin argumentos que devuelve un modelo NUEVO con fit(X, y) y
        predict_proba(X) (interfaz de scikit-learn). Se crea uno nuevo por fold para que
        ningún fold herede lo aprendido por otro.
    Devuelve:
        folds: DataFrame con una fila por fold (repeat, fold y las 4 métricas).
        oof: array (n_repeats, n_estrellas) con la predicción "fuera de fold" de cada
             estrella en cada repetición (cada estrella se predice sin haber sido
             usada para entrenar en ese fold). Sirve para calcular intervalos por bootstrap.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(y)))
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(k, n_splits)
        model = make_model()
        model.fit(X[tr], y[tr])
        score = model.predict_proba(X[va])[:, 1]
        oof[repeat, va] = score
        rows.append({"repeat": repeat, "fold": fold, **compute_metrics(y[va], score, threshold)})
    return pd.DataFrame(rows), oof


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "pr_auc",
    n_boot: int = 1000,
    threshold: float = 0.5,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Intervalo de confianza por bootstrap estratificado.

    Remuestrea con reemplazo los positivos y los negativos por separado, de modo que cada
    remuestra conserve el mismo número de planetas (si no, con pocos positivos algunas
    remuestras quedarían con 0 y la métrica no se podría calcular).
    Devuelve (valor observado, límite inferior, límite superior).
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    observed = compute_metrics(y_true, y_score, threshold)[metric]
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values[b] = compute_metrics(y_true[idx], y_score[idx], threshold)[metric]
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return observed, low, high


def summarize_cv(folds: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar de cada métrica a través de los folds."""
    return folds[list(METRICS)].agg(["mean", "std"]).T


def report_test(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5, n_boot: int = 1000) -> pd.DataFrame:
    """Métricas sobre el test oficial con intervalo de confianza bootstrap al 95%."""
    rows = {}
    for m in METRICS:
        obs, lo, hi = bootstrap_ci(y_true, y_score, m, n_boot=n_boot, threshold=threshold)
        rows[m] = {"valor": obs, "ic95_inf": lo, "ic95_sup": hi}
    return pd.DataFrame(rows).T
