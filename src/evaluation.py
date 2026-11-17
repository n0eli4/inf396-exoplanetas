"""Métricas y protocolo de evaluación robusto al desbalance de clases."""
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
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


def choose_threshold(y: np.ndarray, oof: np.ndarray) -> float:
    """Umbral que maximiza F1 sobre las predicciones fuera de fold de la CV.

    oof: array (n_repeats, n_estrellas) devuelto por repeated_cv; se juntan todas las repeticiones.
    El umbral se elige solo con el conjunto de entrenamiento y luego se aplica tal cual al test,
    para no ajustar nada mirando los 5 positivos del test.
    """
    oof = np.atleast_2d(oof)
    y_all = np.tile(y, oof.shape[0])
    precision, recall, thresholds = precision_recall_curve(y_all, oof.ravel())
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])   # el último punto de la curva no tiene umbral


def paired_diff(folds: pd.DataFrame, folds_ref: pd.DataFrame, metric: str = "pr_auc") -> pd.Series:
    """Diferencia fold a fold (folds - folds_ref) entre dos configuraciones evaluadas con la misma partición.

    Como ambas usan los mismos folds (mismo seed), la dificultad propia de cada fold se cancela y la
    diferencia es mucho menos ruidosa que comparar medias. Solo usa los (repeat, fold) presentes en ambas,
    así una CV 5x1 se puede comparar contra la primera repetición de una 5x5.
    """
    a = folds.set_index(["repeat", "fold"])[metric]
    b = folds_ref.set_index(["repeat", "fold"])[metric]
    common = a.index.intersection(b.index)
    return a.loc[common] - b.loc[common]


def paired_tests(diff: pd.Series, n_splits: int = 5) -> dict:
    """Contrastes de hipótesis sobre diferencias pareadas por fold (salida de paired_diff). H0: diferencia media = 0.

    - t-test corregido para CV repetida (Nadeau y Bengio, 2003): los folds comparten datos de entrenamiento,
      así que sus diferencias no son independientes y el t-test común subestima la varianza (da p-valores
      demasiado chicos). La corrección multiplica la varianza por (1/n + n_val/n_entr) en vez de 1/n, con
      n_val/n_entr = 1/(n_splits - 1).
    - Wilcoxon de rangos con signo: no supone normalidad, pero tampoco corrige la dependencia entre folds
      (es optimista); se reporta como complemento.
    Ambos son bilaterales.
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + 1 / (n_splits - 1)) * var) if var > 0 else np.nan
    p_t = 2 * stats.t.sf(abs(t), df=n - 1) if var > 0 else np.nan
    p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    return {"delta": d.mean(), "t_corregido": t, "p_t_corregido": p_t, "p_wilcoxon": p_w, "n_folds": n}


def holm(pvalues) -> np.ndarray:
    """Corrección de Holm-Bonferroni para comparaciones múltiples (controla la probabilidad de algún falso hallazgo)."""
    p = np.asarray(pvalues, dtype=float)
    orden = np.argsort(p)
    ajust = np.empty_like(p)
    m = len(p)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - rango) * p[i]))
        ajust[i] = acumulado
    return ajust


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
