# Datos

Dataset: **Exoplanet Hunting in Deep Space – Kepler labelled time series data** (CC0)
https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

## Cómo obtenerlo

1. Descargar el dataset desde el enlace de Kaggle (requiere cuenta gratuita).
2. Descomprimir y dejar estos dos archivos en esta carpeta (`data/`):
   - `exoTrain.csv` (5.087 estrellas, 37 con exoplanetas)
   - `exoTest.csv` (570 estrellas, 5 con exoplanetas)

Los CSV están en `.gitignore`: **no se suben al repositorio**.

## Formato

Cada fila es una estrella: columna `LABEL` (1 = sin exoplaneta, 2 = con exoplaneta)
y 3.197 columnas `FLUX.1 ... FLUX.3197` con el flujo luminoso medido.
