# Reglas del proyecto

## Commits

- **Nunca** agregar a Claude como coautor en un commit: ni `Co-Authored-By: Claude ...`
  ni ninguna otra línea de atribución a Claude en mensajes de commit o descripciones de PR.
- Después de hacer cada commit, verificar con `git log -1 --format=%B` que el mensaje no
  contenga "Co-Authored-By" ni "Claude", y **avisar explícitamente al usuario** el resultado
  de esa verificación.
- La verificación es literal: el mensaje tampoco puede nombrar este archivo (`CLAUDE.md`).
  Usar "archivo de instrucciones del proyecto".
- Los commits van a nombre de Noelia (configuración de git de este repo). En un PC nuevo,
  comprobar `git config user.name` / `user.email` dentro del repo: deben ser
  `n0eli4` / `noelia.morales@usm.cl`; si no, configurarlos con `git config` (local, sin `--global`).
- Mensajes de commit en español, estilo "Paso N: ..." para avances del pipeline.
- Antes de hacer push, `git pull`: se trabaja desde dos PCs.

## Mantener este archivo al día

Al terminar cada avance, actualizar las secciones **Estado**, **Hallazgos** y **Pendiente** de
abajo, para que cualquier sesión en cualquier PC pueda continuar sin contexto previo.

---

# Contexto del proyecto

**INF396 (UTFSM, 2026)**, profesor Alejandro Veloz. Autores: Noelia Morales y Matías Lara
(Ingeniería Civil Informática). Tema: detección de exoplanetas en curvas de luz de Kepler,
comparando **features físicas (estadísticos, FFT, BLS) + Random Forest/XGBoost** contra una
**CNN 1D**, con evaluación robusta al desbalance, ablación e interpretabilidad.

- **Entrega final y presentación oral: viernes 20 de noviembre de 2026.**
  - Repo en GitHub (compartido con los docentes) con README: descripción, requisitos,
    instalación e instrucciones para reproducir.
  - Informe en LaTeX en `report/` con el PDF final subido. Debe incluir: título y autores
    (nombre, apellido, carrera), resumen, introducción y datos, solución propuesta,
    implementación (arquitectura, módulos, decisiones de diseño), metodología, métricas y
    resultados, análisis crítico, conclusiones y trabajo futuro, y bibliografía con BibTeX.
    No incluir listados extensos de código.
  - Presentación presencial de 10–15 min: problema, solución, implementación, resultados y
    conclusiones, con figuras.
- **Rúbrica:** claridad del informe y la presentación 20; solución 40; implementación y repo 25;
  experimentación y análisis 15 (se premia el aporte propio).
  **−20 si el informe final no es consistente con el informe de avance.** Todo lo prometido en
  `report/parcial/parcial_v1.tex` debe aparecer en el final, o justificarse por qué cambió.
- Las instrucciones completas del curso están en `intrucciones-proyecto.pdf` (fuera del repo,
  en la carpeta padre del PC de Matías) y en https://aavelozb.github.io/inf396/evals/proyectos-inf396-2026.html

# Configuración en un PC nuevo (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU (más liviano)
pip install -r requirements.txt
```

- Datos: `exoTrain.csv` y `exoTest.csv` de Kaggle
  (https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data, requiere
  login) en `data/`. No están en git (pesan ~290 MB).
- La consola de Windows usa cp1252 y muestra `±` y los tildes como `�`. Para evitarlo:
  `$env:PYTHONIOENCODING="utf-8"` (o `PYTHONIOENCODING=utf-8` en bash).
- Ejecutar un notebook sin abrir Jupyter:
  `cd notebooks; ..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 07_ablacion.ipynb`
- Las advertencias de git "LF will be replaced by CRLF" son inofensivas.

# Máquinas y tiempos de cómputo

| PC | CPU | CV de la CNN (5 folds × 20 épocas) | BLS por variante (train + test) |
|---|---|---|---|
| Casa | 12 hilos | ~7 min (411 s, medido) | ~6 min |
| Laptop de Matías | Ryzen 5 4500U, 6 núcleos, 7 GB RAM | ~20–35 min (estimado) | ~10 min (medido: 536 s + 57 s) |

Probar el código en el laptop con pocas épocas y dejar las corridas largas de la CNN para el PC
de la casa.

# Estructura y convenciones

- `src/`: toda la lógica reutilizable. `notebooks/`: solo invocan `src/`, numerados en orden.
  `scripts/`: cómputo pesado, reanudable y ejecutable por línea de comandos.
  `results/`: figuras y tablas con el prefijo del notebook (`07_...`).
- Código, comentarios y docstrings en español, al estilo de los módulos existentes.
- **PR-AUC es la métrica principal** (el azar da 0,007). Recall, precisión y F1 son secundarias.
- **Misma partición en todo:** `repeated_cv(..., seed=0)`. La CV 5×1 coincide con la primera
  repetición de la 5×5, así que las configuraciones se comparan fold a fold con
  `paired_diff`. Clásicos en 5×5; CNN en 5×1 por costo.
- **El test oficial (5 positivos) solo confirma; nunca se usa para decidir.** El umbral se elige
  con `choose_threshold` sobre las predicciones fuera de fold de la CV del entrenamiento.
- Las opciones nuevas se agregan **apagadas por defecto**, para que los valores por defecto
  reproduzcan el informe de avance (por ejemplo, `CNNClassifier(augment=False, class_weight="balanced", cosine_lr=False)`).
- Ablación: cambiar **una decisión a la vez** respecto de la base.

## Archivos clave

| Archivo | Qué hace |
|---|---|
| `src/preprocessing.py` | `preprocess(X, window=101, n_sigma=4.0)`: mediana móvil, recorte de picos hacia arriba y normalización robusta. `window=None` o `n_sigma=None` omiten el paso correspondiente |
| `src/features.py` | 8 estadísticos, 4 de FFT y 6 de BLS. Supuestos: cadencia de 30 min, períodos de 0,5–30 d, duraciones de 0,05–0,3 d |
| `src/evaluation.py` | `compute_metrics`, `repeated_cv` (devuelve folds y oof), `bootstrap_ci`, `report_test`, `choose_threshold`, `paired_diff` |
| `src/models_classic.py` | `make_random_forest(balanced=)`, `make_xgboost(balanced=)` |
| `src/models_cnn.py` | `CNN1D` (3 bloques conv, global max pool) y `CNNClassifier` (`augment`, `noise_std`, `class_weight` ∈ {balanced, sqrt, none}, `cosine_lr`) |
| `scripts/compute_features.py` | `--variante base\|sin_detrend\|sin_recorte\|ventana_301\|cadencia_15\|cadencia_60\|periodos_15\|todas`. Escribe `results/features_<variante>_{train,test}.csv` |
| `scripts/run_ablation.py` | `--grupo clasicos\|cnn\|todos`, `--solo <nombres>`. Guarda `results/07_ablacion/<nombre>.csv` y `<nombre>_oof.npy`; se salta lo que ya existe |
| `notebooks/07_ablacion.ipynb` | Lee `results/07_ablacion/` y genera `results/07_ablacion_resumen.csv` y `results/07_ablacion.png` |

# Estado (actualizado 2026-11-17)

**Hecho hasta el informe de avance (pasos 1–8, octubre):** carga y exploración,
preprocesamiento, protocolo de evaluación, features (estadísticos, FFT, BLS), baseline RF/XGB
con ablación por familia de features, CNN 1D inicial, informe de avance (`report/parcial/`) y
README.

**Hecho el 2026-11-17:**
- Paso 9: augmentation en la CNN (desplazamiento circular, inversión temporal, ruido
  gaussiano), opciones de peso de clase y tasa coseno; `choose_threshold`.
- Paso 10: `run_ablation.py`, variantes en `compute_features.py`, `paired_diff`, notebook 07.
- Paso 11: las 6 variantes de features calculadas y versionadas
  (`results/features_<variante>_*.csv`, no hace falta recalcularlas). **Ablación clásica
  completa:** 16 configuraciones, con conclusiones escritas en el notebook 07.
- `.venv` instalado en el laptop (torch 2.15 CPU, xgboost 3.2, scikit-learn 1.9.1).
- El notebook 07 se genera con un script auxiliar fuera del repo. Para editarlo, modificar
  directamente el `.ipynb`.

**En curso:**
- PC de la casa: `run_ablation.py --grupo cnn` (8 configuraciones, ~1 h). Después, commit y push
  de `results/07_ablacion/`, ejecutar el notebook 07 y completar las conclusiones de la CNN.

# Hallazgos importantes

- **Baseline (CV 5×5, PR-AUC):** Random Forest 0,394 ± 0,200 (reproducido exacto);
  XGBoost 0,377 ± 0,186 (0,374 en el informe de avance; la pequeña diferencia viene
  probablemente de la versión de xgboost).
- **Por familia de features:** estadísticos ≈ 0,04, FFT ≈ 0,14, BLS ≈ 0,33, todas ≈ 0,39.
  El BLS es lo que más aporta; la FFT suma poco.
- **CNN inicial (CV 5×1):** PR-AUC 0,27 ± 0,16, con el recall más alto (0,56) y baja precisión.
  En el test encuentra los 5 planetas entre las posiciones 3 y 7 de 570.
- **Test oficial:** intervalos bootstrap enormes (por ejemplo, PR-AUC del RF de 0,58 con
  IC [0,16; 1,0]). No hay evidencia de que un modelo sea mejor que otro.
- **Ablación clásica (diferencia pareada contra la base, 25 folds; detalle en
  `results/07_ablacion_resumen.csv` y `results/07_ablacion.png`):**
  - **Lo que más pesa es el preprocesamiento.** Sin detrending, el PR-AUC cae a ~0,17 (Δ −0,21 a
    −0,23; mejora solo en 3/25 folds). Con ventana 301 cae a 0,21–0,24 (Δ −0,14 a −0,19). Sin
    recorte de picos, Δ −0,05 a −0,09 (en el RF mejora solo en 2/25 folds).
  - **La ponderación de clases casi no importa en los árboles:** RF sin peso −0,042 (mejora en
    10/25 folds); XGB sin peso +0,029 (15/25). Ambas diferencias están dentro del ruido.
  - **Sensibilidad a la cadencia:** el PR-AUC va de 0,32 (15 min) a 0,43 (60 min); con
    períodos de 0,5–15 d (y cadencia de 30 min) da 0,41–0,43. La explicación es que la grilla
    está en días: con 15 min, el período máximo cubre casi toda la curva (~2.880 puntos) y el
    BLS ajusta caídas aisladas; limitar el período máximo a ~720 puntos (al menos ~4
    repeticiones) ayuda un poco (+0,02 a +0,05, en 16–18/25 folds). **Exigir que el tránsito se
    repita ayuda** (relacionado con la estrella 28). Estas mejoras se eligieron mirando la misma
    CV, así que **no se cambió la configuración principal**; se reportan como sensibilidad. Si se
    adoptaran, habría que decirlo en el informe.
- **Umbral 0,5 vs umbral elegido en CV:** el RF con 0,5 tiene recall 0,09; con el umbral de la
  CV (~0,25) sube a ~0,32 con precisión ~0,7. Hay que reportar las métricas con el umbral
  elegido en CV, no con 0,5.
- **Rarezas de los datos:** etiquetas originales 2 = planeta y 1 = no (`data.py` las pasa a
  1/0); sin marcas de tiempo (de ahí el supuesto de cadencia); 4 estrellas con MAD = 0 tras
  el detrending (se usa la desviación estándar); más del 99 % de los datos es de la campaña 3
  y los positivos extra vienen de otras campañas, lo que podría introducir sesgo (sin
  verificar).
- **Estrella 28 (con planeta):** el BLS se apoya en una sola caída profunda. Sirve como
  ejemplo para la interpretabilidad.

# Pendiente (por prioridad), entrega el 2026-11-20

1. Terminar la ablación de la CNN (PC de la casa) y escribir sus conclusiones en el notebook 07.
   La parte clásica ya está lista.
2. Elegir la mejor configuración de la CNN, correrla en CV 5×5 (~35 min en el PC de la casa) y
   evaluarla en el test con el umbral elegido en CV.
3. **Interpretabilidad** (notebook 08, prometido en el informe de avance): saliencia y
   Grad-CAM 1D sobre los positivos del test y los falsos positivos con mayor puntaje;
   comprobar si la CNN mira las mismas caídas que el BLS; importancia por permutación en el
   RF; análisis de falsos positivos y negativos; revisar el sesgo por campaña.
4. Ajuste de hiperparámetros: grilla chica para RF/XGB con validación anidada liviana. Si no
   alcanza el tiempo, justificarlo en el informe.
5. Informe final en `report/final/`, partiendo de `parcial_v1.tex` y agregando las secciones
   que exige el curso. Subir el PDF.
6. README: actualizar la tabla de notebooks (07 y 08), los resultados y el estado.
7. Presentación de 10–15 min (lista el jueves 19).

**Se puede recortar, si se justifica en el informe:** la CNN híbrida con features BLS, las
features de consistencia entre tránsitos, la validación anidada completa y la CV 5×5 de todas
las variantes de la CNN.
