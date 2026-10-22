"""Calcula las features de train y test y las guarda en results/features_{train,test}.csv.

Uso (desde la raíz del repo):
    python scripts/compute_features.py
Tarda unos minutos (el BLS es la parte lenta; usa todos los núcleos).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_train_test
from src.features import all_features
from src.preprocessing import preprocess


def main():
    X_train, y_train, X_test, y_test = load_train_test()
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        t0 = time.time()
        feats = all_features(preprocess(X))
        feats["label"] = y
        out = ROOT / "results" / f"features_{name}.csv"
        feats.to_csv(out, index=False)
        print(f"{name}: {feats.shape} -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
