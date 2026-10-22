"""Extracción de features por estrella a partir de curvas ya preprocesadas.

Tres familias:
  - estadísticos genéricos de la distribución del flujo,
  - espectro de Fourier (FFT),
  - periodograma Box Least Squares (BLS).

Todas reciben Z: array (n_estrellas, n_puntos) producido por src.preprocessing.preprocess.
"""
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

CADENCE_MIN = 30.0  # supuesto: cadencia larga de Kepler (30 min entre mediciones)
PERIODS = np.linspace(0.5, 30.0, 3000)      # días
DURATIONS = np.array([0.05, 0.1, 0.2, 0.3])  # días


def _time_axis(n_points: int, cadence_min: float) -> np.ndarray:
    """Eje de tiempo en días: el dataset no trae marcas de tiempo, se asume cadencia uniforme."""
    return np.arange(n_points) * cadence_min / (60 * 24)


def stat_features(Z: np.ndarray) -> pd.DataFrame:
    """Estadísticos que resumen la distribución del flujo de cada estrella."""
    return pd.DataFrame({
        "stat_min": Z.min(axis=1),
        "stat_p1": np.percentile(Z, 1, axis=1),
        "stat_p5": np.percentile(Z, 5, axis=1),
        "stat_std": Z.std(axis=1),
        "stat_skew": skew(Z, axis=1),
        "stat_kurtosis": kurtosis(Z, axis=1),
        "stat_n_below_3": (Z < -3).sum(axis=1),
        "stat_n_below_5": (Z < -5).sum(axis=1),
    })


def fft_features(Z: np.ndarray, cadence_min: float = CADENCE_MIN) -> pd.DataFrame:
    """Features del espectro de potencia (FFT).

    peak_freq: frecuencia (ciclos/día) del pico más fuerte, excluyendo la frecuencia 0.
    peak_ratio: potencia del pico / mediana del espectro (qué tan destacado es).
    low_frac / high_frac: fracción de la potencia total en frecuencias bajas (< 0,5 ciclos/día)
        y altas (> 5 ciclos/día).
    """
    n = Z.shape[1]
    dt_days = cadence_min / (60 * 24)
    freqs = np.fft.rfftfreq(n, d=dt_days)
    power = np.abs(np.fft.rfft(Z - Z.mean(axis=1, keepdims=True), axis=1)) ** 2
    power, freqs = power[:, 1:], freqs[1:]  # se descarta la frecuencia 0
    peak_idx = power.argmax(axis=1)
    peak = power[np.arange(len(Z)), peak_idx]
    total = power.sum(axis=1)
    return pd.DataFrame({
        "fft_peak_freq": freqs[peak_idx],
        "fft_peak_ratio": peak / np.maximum(np.median(power, axis=1), 1e-12),
        "fft_low_frac": power[:, freqs < 0.5].sum(axis=1) / total,
        "fft_high_frac": power[:, freqs > 5.0].sum(axis=1) / total,
    })


def _bls_one(t, z, periods, durations):
    """BLS de una sola estrella. Devuelve el mejor ajuste (la 'caja' que mejor calza)."""
    res = BoxLeastSquares(t, z).power(periods, durations, objective="snr")
    i = int(np.argmax(res.power))
    period = res.period[i]
    return (
        period,
        res.duration[i],
        res.depth[i],
        res.power[i],                                   # SNR del mejor ajuste
        res.power[i] / max(np.median(res.power), 1e-12),  # contraste frente al periodograma típico
        int((t[-1] - t[0]) // period),                  # cuántos tránsitos caben en la curva
    )


def bls_features(
    Z: np.ndarray,
    cadence_min: float = CADENCE_MIN,
    periods: np.ndarray = PERIODS,
    durations: np.ndarray = DURATIONS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Features del periodograma BLS: período, duración, profundidad y potencia del mejor tránsito."""
    t = _time_axis(Z.shape[1], cadence_min)
    rows = Parallel(n_jobs=n_jobs)(delayed(_bls_one)(t, z, periods, durations) for z in Z)
    return pd.DataFrame(rows, columns=[
        "bls_period", "bls_duration", "bls_depth", "bls_snr", "bls_power_ratio", "bls_n_transits",
    ])


def all_features(Z: np.ndarray, cadence_min: float = CADENCE_MIN, n_jobs: int = -1) -> pd.DataFrame:
    """Concatena estadísticos, FFT y BLS en un solo DataFrame (una fila por estrella)."""
    return pd.concat(
        [stat_features(Z), fft_features(Z, cadence_min), bls_features(Z, cadence_min, n_jobs=n_jobs)],
        axis=1,
    )
