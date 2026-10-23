# Detección de exoplanetas en curvas de luz de Kepler

Proyecto INF396 (2026). Comparación entre un clasificador basado en features físicas
(FFT y Box Least Squares + Random Forest / XGBoost) y una CNN 1D, con un protocolo de
evaluación robusto al desbalance de clases, estudio de ablación e interpretabilidad.

**Autores:** Noelia Morales, Matías Lara (Ingeniería Civil Informática, UTFSM)

## Descripción

Se busca detectar automáticamente estrellas con exoplanetas a partir de sus curvas de luz
(brillo medido por el telescopio Kepler a lo largo del tiempo). Solo el ~0,7% de las
estrellas del dataset tiene planeta, por lo que *accuracy* no sirve como métrica: un modelo
que siempre responde "sin planeta" obtiene ~99%. Usamos PR-AUC como métrica principal,
validación cruzada estratificada repetida e intervalos de confianza por bootstrap.

Dos enfoques comparados:

1. **Features físicas:** 18 features por estrella (8 estadísticos, 4 de FFT, 6 del periodograma
   BLS) y modelos Random Forest y XGBoost.
2. **CNN 1D:** red convolucional sobre la curva preprocesada completa.

## Requisitos

- Python 3.10 o superior
- Dependencias en `requirements.txt` (numpy, pandas, scipy, scikit-learn, xgboost, astropy,
  matplotlib, torch, jupyter). PyTorch se usa en CPU; no se necesita GPU.

## Instalación

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Ver [data/README.md](data/README.md) para descargar el dataset de Kaggle. Los archivos
`exoTrain.csv` y `exoTest.csv` deben quedar dentro de `data/` (no se suben al repositorio).

## Estructura

```
data/        datasets (no se suben) e instrucciones de descarga
src/         código reutilizable
notebooks/   exploración y experimentos (se ejecutan en orden numérico)
scripts/     scripts para reproducir los cálculos pesados
results/     tablas, figuras y features generadas
report/      informes en LaTeX (propuesta, parcial, final)
```

Módulos en `src/`:

| Archivo | Contenido |
|---|---|
| `data.py` | carga de los CSV; etiquetas como 0 (sin planeta) y 1 (con planeta) |
| `preprocessing.py` | detrending con mediana móvil, recorte de picos, normalización |
| `features.py` | features estadísticas, FFT y BLS |
| `evaluation.py` | métricas, validación cruzada repetida, bootstrap |
| `models_classic.py` | Random Forest y XGBoost con ponderación de clases |
| `models_cnn.py` | CNN 1D (PyTorch) con interfaz `fit` / `predict_proba` |

## Ejecución

Desde la raíz del repositorio, con el entorno virtual activado y los CSV en `data/`.

**1. Calcular las features** (opcional: ya están guardadas en `results/features_train.csv` y
`results/features_test.csv`). El BLS es la parte lenta (~6 min con 12 núcleos):

```bash
python scripts/compute_features.py
```

**2. Ejecutar los notebooks en orden**, desde `jupyter notebook`:

| Notebook | Qué hace | Tiempo estimado |
|---|---|---|
| `01_exploracion.ipynb` | tamaño del dataset, desbalance, curvas de ejemplo | segundos |
| `02_preprocesamiento.ipynb` | elección de la ventana, antes y después | ~1 min |
| `03_evaluacion.ipynb` | protocolo de evaluación y modelos de referencia | ~2 min |
| `04_features.ipynb` | FFT y BLS sobre ejemplos, análisis de features | ~1 min |
| `05_baseline.ipynb` | Random Forest y XGBoost, ablación por familia de features | pocos min |
| `06_cnn.ipynb` | CNN 1D y comparación contra el baseline | ~12 min |

Los resultados que generan se guardan en `results/`. Los notebooks ya vienen ejecutados, así
que se pueden leer sin correrlos.

**3. Compilar el informe** (requiere una distribución de LaTeX):

```bash
cd report/parcial
pdflatex parcial_v1.tex && bibtex parcial_v1 && pdflatex parcial_v1.tex && pdflatex parcial_v1.tex
```

## Resultados preliminares

PR-AUC en validación cruzada (el azar da 0,007):

| Modelo | PR-AUC |
|---|---|
| Random Forest, 18 features | 0,39 (CV 5x5) |
| XGBoost, 18 features | 0,37 (CV 5x5) |
| CNN 1D (versión inicial, sin augmentation) | 0,27 (CV 5x1) |

Las features del BLS aportan mucho más que los estadísticos genéricos. Con solo 37 positivos
en entrenamiento y 5 en prueba, las diferencias entre modelos tienen alta incertidumbre; ver
el informe en `report/parcial/` para el análisis completo.

## Estado

Entrega parcial lista (`report/parcial/parcial_v1.pdf`). Pendiente para la entrega final:
data augmentation y ajuste de la CNN, ablación completa, interpretabilidad (saliencia y
Grad-CAM), sensibilidad al supuesto de cadencia, informe final y presentación.
