"""Calcula las vistas plegadas en fase (src/folding.py) de train y test y las guarda en results/folded_{train,test}.npz.

Uso (desde la raíz del repo):
    python scripts/compute_folded.py
Tarda lo mismo que el BLS de compute_features.py (~6 min con 12 hilos, ~10 min con 6 núcleos).
Cada .npz contiene: views (n_estrellas, 262), bls (n_estrellas, 3: período, duración, t0 en días) y label.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_train_test
from src.folding import all_fold_views
from src.preprocessing import preprocess


def main():
    X_train, y_train, X_test, y_test = load_train_test()
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        t0 = time.time()
        views, bls = all_fold_views(preprocess(X))
        out = ROOT / "results" / f"folded_{name}.npz"
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, views=views.astype(np.float32), bls=bls, label=y)
        tmp.replace(out)
        print(f"{name}: {views.shape} -> {out.name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
