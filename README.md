# Detección de exoplanetas en curvas de luz de Kepler

Proyecto INF396 (2026). Comparación entre un clasificador basado en features físicas
(FFT y Box Least Squares + Random Forest / XGBoost) y una CNN 1D, con un protocolo de
evaluación robusto al desbalance de clases, estudio de ablación e interpretabilidad.

**Autores:** Noelia Morales, Matías Lara (Ingeniería Civil Informática, UTFSM)

## Requisitos

- Python 3.10 o superior
- Dependencias en `requirements.txt`

## Instalación

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Ver [data/README.md](data/README.md) para descargar el dataset de Kaggle.

## Estructura

```
data/        datasets (no se suben) e instrucciones de descarga
src/         código reutilizable (carga, preprocesamiento, features, modelos, evaluación)
notebooks/   exploración y experimentos
scripts/     scripts para reproducir los experimentos
results/     tablas y figuras generadas
report/      informes en LaTeX (propuesta, parcial, final)
```

## Ejecución

_(Se completará a medida que avance la implementación.)_

## Estado

Entrega parcial en curso. Ver `report/parcial/`.
