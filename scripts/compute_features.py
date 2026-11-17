"""Calcula las features de train y test y las guarda en results/features_{train,test}.csv.

Uso (desde la raíz del repo):
    python scripts/compute_features.py
Tarda unos minutos (el BLS es la parte lenta; usa todos los núcleos).

Variantes para el estudio de ablación y de sensibilidad (se guardan como
results/features_<variante>_{train,test}.csv y las usa scripts/run_ablation.py):
    python scripts/compute_features.py --variante sin_detrend
    python scripts/compute_features.py --variante todas      # todas las variantes, una tras otra
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_train_test
from src.features import CADENCE_MIN, PERIODS, all_features
from src.preprocessing import preprocess

# Cada variante cambia UNA cosa respecto de la configuración base (ventana 101, recorte 4σ, cadencia 30 min,
# períodos de 0,5 a 30 días).
VARIANTES = {
    "base": {},
    "sin_detrend": {"window": None},
    "sin_recorte": {"n_sigma": None},
    "ventana_301": {"window": 301},
    "cadencia_15": {"cadence_min": 15.0},
    "cadencia_60": {"cadence_min": 60.0},
    "periodos_15": {"periods": np.linspace(0.5, 15.0, 3000)},
}


def compute(variante: str) -> None:
    cfg = VARIANTES[variante]
    prep = {k: cfg[k] for k in ("window", "n_sigma") if k in cfg}
    X_train, y_train, X_test, y_test = load_train_test()
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        t0 = time.time()
        feats = all_features(
            preprocess(X, **prep),
            cadence_min=cfg.get("cadence_min", CADENCE_MIN),
            periods=cfg.get("periods", PERIODS),
        )
        feats["label"] = y
        sufijo = "" if variante == "base" else f"{variante}_"
        out = ROOT / "results" / f"features_{sufijo}{name}.csv"
        feats.to_csv(out, index=False)
        print(f"[{variante}] {name}: {feats.shape} -> {out.name} ({time.time() - t0:.0f} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variante", default="base", choices=[*VARIANTES, "todas"])
    args = parser.parse_args()
    variantes = [v for v in VARIANTES if v != "base"] if args.variante == "todas" else [args.variante]
    for v in variantes:
        compute(v)


if __name__ == "__main__":
    main()
