"""Vistas plegadas en fase de las curvas de luz (mejora M1).

Si una estrella tiene un planeta de período P, al "plegar" la curva en P (tomar el tiempo módulo P) todas las
repeticiones del tránsito caen en la misma fase y se apilan. Promediando por intervalos de fase, el ruido baja
y el tránsito se ve mucho más claro que en la curva completa, donde la CNN tenía que descubrir sola la repetición.

Siguiendo a Shallue y Vanderburg (2018) se construyen dos vistas por estrella, ambas centradas en el tránsito
del mejor ajuste BLS:
  - global: la órbita completa (fase de -0,5 a 0,5) en N_GLOBAL intervalos,
  - local: un acercamiento de ±LOCAL_DURACIONES duraciones alrededor del tránsito en N_LOCAL intervalos.
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares
from joblib import Parallel, delayed

from src.features import CADENCE_MIN, DURATIONS, PERIODS, _time_axis

N_GLOBAL, N_LOCAL, LOCAL_DURACIONES = 201, 61, 3.0


def bls_best(z: np.ndarray, cadence_min: float = CADENCE_MIN, periods=PERIODS, durations=DURATIONS) -> tuple:
    """Mejor ajuste BLS de una curva: (período, duración, instante del primer tránsito), en días."""
    t = _time_axis(len(z), cadence_min)
    res = BoxLeastSquares(t, z).power(periods, durations, objective="snr")
    i = int(np.argmax(res.power))
    return res.period[i], res.duration[i], res.transit_time[i]


def _binned(phase: np.ndarray, z: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    """Media de z por intervalo de fase entre lo y hi. Los intervalos vacíos se rellenan interpolando.

    Se usa la media y no la mediana: con períodos largos cada intervalo junta solo 2-3 pasos por la órbita y
    la mediana descartaría un tránsito de 2-3 puntos (pasaba con la estrella 28). Los picos hacia arriba ya
    vienen recortados por preprocess, así que la media no se dispara por valores atípicos positivos.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = z[idx == b]
        if len(sel):
            out[b] = sel.mean()
    ok = ~np.isnan(out)
    if ok.sum() == 0:
        return np.zeros(n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return np.interp(centers, centers[ok], out[ok])


def fold_views(z: np.ndarray, period: float, duration: float, t0: float, cadence_min: float = CADENCE_MIN) -> np.ndarray:
    """Vista global (N_GLOBAL valores) y local (N_LOCAL) concatenadas, con el tránsito en el centro.

    Los valores siguen en unidades de la desviación robusta (la curva ya viene normalizada por preprocess).
    """
    t = _time_axis(len(z), cadence_min)
    fase = ((t - t0) / period + 0.5) % 1.0 - 0.5            # en [-0,5; 0,5), tránsito en 0
    glob = _binned(fase, z, -0.5, 0.5, N_GLOBAL)
    medio = LOCAL_DURACIONES * duration / period            # media ventana local, en unidades de fase
    medio = min(medio, 0.5)
    loc = _binned(fase, z, -medio, medio, N_LOCAL)
    return np.r_[glob, loc]


def _one(z, cadence_min):
    period, duration, t0 = bls_best(z, cadence_min)
    return fold_views(z, period, duration, t0, cadence_min), (period, duration, t0)


def all_fold_views(Z: np.ndarray, cadence_min: float = CADENCE_MIN, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Vistas plegadas de todas las curvas (en paralelo). Devuelve (vistas, parámetros BLS: período, duración, t0)."""
    out = Parallel(n_jobs=n_jobs)(delayed(_one)(z, cadence_min) for z in Z)
    return np.stack([v for v, _ in out]), np.array([p for _, p in out])
