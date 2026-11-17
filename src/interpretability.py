"""Interpretabilidad de la CNN 1D (saliencia y Grad-CAM) y su comparación con los tránsitos del BLS."""
import numpy as np
import torch
from astropy.timeseries import BoxLeastSquares

from src.features import CADENCE_MIN, DURATIONS, PERIODS, _time_axis
from src.models_cnn import _to_tensor


def saliency(clf, X: np.ndarray) -> np.ndarray:
    """Mapa de saliencia: |d logit / d entrada| para cada punto de cada curva.

    Indica qué puntos cambiarían más la predicción si se movieran un poco. Es ruidoso punto a punto,
    por eso conviene suavizarlo (ver smooth) antes de graficarlo.
    """
    net = clf.net_.eval()
    x = _to_tensor(X).requires_grad_(True)
    net(x).sum().backward()                      # cada curva solo influye en su propio logit
    return x.grad.abs().squeeze(1).numpy()


def grad_cam(clf, X: np.ndarray, layer: int = -2) -> np.ndarray:
    """Grad-CAM 1D sobre una capa de clf.net_.features (por defecto la ReLU del último bloque, antes del pooling).

    Pondera cada filtro de esa capa por el gradiente medio del logit respecto de su activación, suma los filtros
    y aplica ReLU: quedan las zonas de la curva que empujan la predicción hacia "con planeta". Se interpola al
    largo de la curva original y se normaliza a [0, 1] por curva.
    """
    net = clf.net_.eval()
    store = {}
    handle = net.features[layer].register_forward_hook(lambda m, i, o: store.update(act=o))
    x = _to_tensor(X)
    logits = net(x)
    act = store["act"]
    grads = torch.autograd.grad(logits.sum(), act)[0]
    handle.remove()
    weights = grads.mean(dim=2, keepdim=True)                    # (n, canales, 1)
    cam = torch.relu((weights * act).sum(dim=1)).detach().numpy()  # (n, largo reducido)
    length = X.shape[1]
    pos = np.linspace(0, cam.shape[1] - 1, length)
    cam = np.stack([np.interp(pos, np.arange(cam.shape[1]), c) for c in cam])
    return cam / np.maximum(cam.max(axis=1, keepdims=True), 1e-12)


def smooth(M: np.ndarray, width: int = 25) -> np.ndarray:
    """Media móvil por fila (para mostrar la saliencia, que es ruidosa punto a punto)."""
    kernel = np.ones(width) / width
    return np.stack([np.convolve(m, kernel, mode="same") for m in np.atleast_2d(M)])


def bls_transit_mask(z: np.ndarray, cadence_min: float = CADENCE_MIN, periods=PERIODS, durations=DURATIONS) -> tuple:
    """Puntos de la curva que caen dentro de los tránsitos del mejor ajuste BLS.

    Devuelve (máscara booleana del largo de z, período en días, duración en días).
    """
    t = _time_axis(len(z), cadence_min)
    model = BoxLeastSquares(t, z)
    res = model.power(periods, durations, objective="snr")
    i = int(np.argmax(res.power))
    mask = model.transit_mask(t, res.period[i], res.duration[i], res.transit_time[i])
    return mask, res.period[i], res.duration[i]


def attention_on_transits(cam: np.ndarray, mask: np.ndarray) -> float:
    """Cuánto más se concentra la atención dentro de los tránsitos del BLS que si fuera uniforme.

    = (fracción de la masa de Grad-CAM dentro de la máscara) / (fracción de puntos dentro de la máscara).
    1 significa que la CNN no mira especialmente los tránsitos del BLS; valores altos, que los mira.
    """
    if mask.sum() == 0 or cam.sum() == 0:
        return np.nan
    return (cam[mask].sum() / cam.sum()) / mask.mean()
