"""Estudio de ablación: cambia UNA decisión de diseño a la vez y mide su efecto en CV.

Cada configuración guarda en results/07_ablacion/:
    <nombre>.csv      métricas por fold (repeat, fold, recall, precision, f1, pr_auc)
    <nombre>_oof.npy  predicciones fuera de fold, (n_repeats, n_estrellas), para elegir umbral y analizar errores
Si el .csv ya existe, la configuración se salta: el script se puede cortar y volver a lanzar.

Uso (desde la raíz del repo):
    python scripts/run_ablation.py --grupo clasicos   # RF y XGBoost, rápido (minutos)
    python scripts/run_ablation.py --grupo cnn        # CNN, ~7 min por configuración con 12 hilos
    python scripts/run_ablation.py                    # ambos
    python scripts/run_ablation.py --grupo cnn --solo cnn_base cnn_aug

Las variantes de preprocesamiento/cadencia de los modelos clásicos necesitan antes:
    python scripts/compute_features.py --variante todas
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
from src.evaluation import repeated_cv, summarize_cv
from src.models_classic import make_random_forest, make_xgboost, select_columns
from src.preprocessing import preprocess

OUT_DIR = ROOT / "results" / "07_ablacion"

# CNN: la base es la versión del informe de avance (sin augmentation, peso balanceado, tasa fija,
# ventana 101, recorte 4σ). Ordenadas por prioridad, por si no alcanza el tiempo para correrlas todas.
CNN_CONFIGS = {
    "cnn_base":        {},
    "cnn_aug":         {"model": {"augment": True}},
    "cnn_peso_sqrt":   {"model": {"class_weight": "sqrt"}},
    "cnn_peso_none":   {"model": {"class_weight": "none"}},
    "cnn_cosine":      {"model": {"cosine_lr": True}},
    "cnn_sin_detrend": {"prep": {"window": None}},
    "cnn_sin_recorte": {"prep": {"n_sigma": None}},
    "cnn_ventana_301": {"prep": {"window": 301}},
}

# Modelos clásicos (18 features): ponderación de clases y variantes de features de compute_features.py.
FEATURE_VARIANTS = ["base", "sin_detrend", "sin_recorte", "ventana_301", "cadencia_15", "cadencia_60", "periodos_15"]
CLASSIC_MODELS = {"rf": make_random_forest, "xgb": make_xgboost}


def classic_configs() -> dict:
    configs = {}
    for m in CLASSIC_MODELS:
        configs[f"{m}_base"] = {"model": m, "features": "base", "balanced": True}
        configs[f"{m}_sin_peso"] = {"model": m, "features": "base", "balanced": False}
        for v in FEATURE_VARIANTS[1:]:
            configs[f"{m}_{v}"] = {"model": m, "features": v, "balanced": True}
    return configs


def features_path(variant: str) -> Path:
    sufijo = "" if variant == "base" else f"{variant}_"
    return ROOT / "results" / f"features_{sufijo}train.csv"


def save(nombre: str, folds: pd.DataFrame, oof: np.ndarray, seconds: float) -> None:
    folds.to_csv(OUT_DIR / f"{nombre}.csv", index=False)
    np.save(OUT_DIR / f"{nombre}_oof.npy", oof.astype(np.float32))
    s = summarize_cv(folds)
    print(f"{nombre:<20} PR-AUC {s.loc['pr_auc', 'mean']:.3f} ± {s.loc['pr_auc', 'std']:.3f}  "
          f"recall {s.loc['recall', 'mean']:.2f}  precisión {s.loc['precision', 'mean']:.2f}  ({seconds:.0f} s)",
          flush=True)


def run_classic(solo, repeats: int) -> None:
    for nombre, cfg in classic_configs().items():
        if (solo and nombre not in solo) or (OUT_DIR / f"{nombre}.csv").exists():
            continue
        path = features_path(cfg["features"])
        if not path.exists():
            print(f"{nombre:<20} se salta: falta {path.name} "
                  f"(python scripts/compute_features.py --variante {cfg['features']})", flush=True)
            continue
        F = pd.read_csv(path)
        X, y = F[select_columns(F, "todas")].to_numpy(), F["label"].to_numpy()
        make = CLASSIC_MODELS[cfg["model"]]
        t0 = time.time()
        folds, oof = repeated_cv(lambda: make(balanced=cfg["balanced"]), X, y, n_splits=5, n_repeats=repeats)
        save(nombre, folds, oof, time.time() - t0)


def run_cnn(solo, repeats: int, epochs: int) -> None:
    from src.models_cnn import CNNClassifier   # import tardío: torch solo se carga si se usa
    X, y = load_split("train")
    cache = {}
    for nombre, cfg in CNN_CONFIGS.items():
        if (solo and nombre not in solo) or (OUT_DIR / f"{nombre}.csv").exists():
            continue
        prep = cfg.get("prep", {})
        key = tuple(sorted(prep.items()))
        if key not in cache:
            cache[key] = preprocess(X, **prep)
        params = {"epochs": epochs, **cfg.get("model", {})}
        t0 = time.time()
        folds, oof = repeated_cv(lambda: CNNClassifier(**params), cache[key], y, n_splits=5, n_repeats=repeats)
        save(nombre, folds, oof, time.time() - t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grupo", choices=["clasicos", "cnn", "todos"], default="todos")
    parser.add_argument("--solo", nargs="*", help="correr solo estas configuraciones")
    parser.add_argument("--repeats-clasicos", type=int, default=5)
    parser.add_argument("--repeats-cnn", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.grupo in ("clasicos", "todos"):
        run_classic(args.solo, args.repeats_clasicos)
    if args.grupo in ("cnn", "todos"):
        run_cnn(args.solo, args.repeats_cnn, args.epochs)


if __name__ == "__main__":
    main()
