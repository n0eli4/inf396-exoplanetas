"""Preprocesamiento de curvas de luz: detrending, recorte de picos y normalización."""
import numpy as np
from scipy.ndimage import median_filter


def detrend_median(X: np.ndarray, window: int = 101) -> np.ndarray:
    """Quita la tendencia lenta de cada curva restando su mediana móvil.

    window: tamaño (en puntos) de la ventana del filtro de mediana. Debe ser mucho mayor
    que la duración de un tránsito para que el filtro no lo absorba como "tendencia".
    """
    X = np.atleast_2d(X)
    trend = median_filter(X, size=(1, window), mode="nearest")
    return X - trend


def clip_upward_outliers(X: np.ndarray, n_sigma: float = 4.0) -> np.ndarray:
    """Recorta picos hacia arriba por encima de n_sigma desviaciones robustas.

    Un tránsito solo baja el brillo, así que los picos positivos son ruido
    (rayos cósmicos, erupciones). Los valores negativos no se tocan.
    """
    X = np.atleast_2d(X)
    sigma = robust_std(X)[:, None]
    return np.minimum(X, n_sigma * sigma)


def robust_std(X: np.ndarray) -> np.ndarray:
    """Desviación estándar robusta por curva: 1,4826 * MAD (mediana de desviaciones absolutas).

    Si el MAD es 0 (más de la mitad de los puntos idénticos a la mediana, como ocurre en
    algunas estrellas con oscilaciones enormes tras el detrending), se usa la desviación
    estándar común para no dividir por ~0 al normalizar.
    """
    X = np.atleast_2d(X)
    med = np.median(X, axis=1, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(X - med), axis=1)
    return np.where(sigma > 0, sigma, X.std(axis=1))


def normalize(X: np.ndarray) -> np.ndarray:
    """Centra cada curva en su mediana y la divide por su desviación robusta."""
    X = np.atleast_2d(X)
    med = np.median(X, axis=1, keepdims=True)
    sigma = np.maximum(robust_std(X), 1e-12)[:, None]
    return (X - med) / sigma


def preprocess(X: np.ndarray, window: int = 101, n_sigma: float = 4.0) -> np.ndarray:
    """Pipeline completo: detrending -> recorte de picos -> normalización."""
    X = detrend_median(X, window)
    X = clip_upward_outliers(X, n_sigma)
    return normalize(X)
