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
| Laptop de Matías | Ryzen 5 4500U, 6 núcleos, 7 GB RAM | ~55 min (extrapolado: 4,7 s por época con 570 curvas; un entrenamiento completo, ~14 min) | ~10 min (medido: 536 s + 57 s) |

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
| `src/interpretability.py` | `saliency`, `grad_cam` (ReLU del último bloque conv), `smooth`, `bls_transit_mask` (puntos dentro de los tránsitos del mejor BLS) y `attention_on_transits` (concentración del Grad-CAM en esos tránsitos; ×1 = sin preferencia) |
| `notebooks/08_interpretabilidad.ipynb` | Grad-CAM contra tránsitos BLS en el test, importancia por permutación del RF, análisis de errores con las predicciones fuera de fold y prueba de sesgo por campaña. `CNN_NOMBRE`/`CNN_PARAMS` al inicio: cambiarlos a la mejor CNN de la ablación. La CNN entrenada se guarda en `results/08_<nombre>.pt` (ignorado por git) |
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
- Paso 12: `src/interpretability.py` y notebook 08 (Grad-CAM, importancia por permutación,
  análisis de errores y **fuga de información del dataset**), ejecutado con la CNN base y con
  conclusiones escritas. Figuras: `results/08_gradcam.png` y `results/08_importancia_rf.png`.
- Paso 13 (**M3 y M4 hechas**): modelos del ramo en `src/models_classic.py` (`MODELOS_RAMO`:
  logística L1/L2 con C elegido por CV interna, LDA con encogimiento, KNN k=15, SVM RBF; escalado
  estándar dentro del pipeline), `paired_tests` (t-test corregido de Nadeau y Bengio + Wilcoxon) y
  `holm` en `src/evaluation.py`, y notebook `09_modelos_ramo` con conclusiones. Resultados en
  `results/09_modelos/`, `results/09_comparacion_modelos.csv`, `results/09_contrastes_ablacion.csv`
  y `results/09_modelos.png`.
- `.venv` instalado en el laptop (torch 2.15 CPU, xgboost 3.2, scikit-learn 1.9.1).
- El notebook 07 se genera con un script auxiliar fuera del repo. Para editarlo, modificar
  directamente el `.ipynb`.

**En curso:**
- PC de la casa: `run_ablation.py --grupo cnn` (8 configuraciones, ~1 h). Después, commit y push
  de `results/07_ablacion/`, ejecutar el notebook 07 y completar las conclusiones de la CNN.

# Hallazgos importantes

- **Modelos del ramo (M3, CV 5×5, 18 features):** RF 0,394, KNN 0,378 y XGB 0,377 quedan empatados
  (Δ ≈ −0,02, p > 0,7). Logística L1 0,306, L2 0,287, LDA 0,196 y SVM RBF 0,140. La relación no es
  lineal (interacciones como "período corto **y** muchos tránsitos"). Que un KNN simple empate
  con el RF indica que **lo valioso son las features físicas, no el clasificador**. El lasso
  conserva 15 de 18 features; las de mayor peso combinan FFT y BLS.
- **Lección de preprocesamiento (reportar en el informe):** la transformación por cuantiles a
  distribución normal hundió la logística de 0,29 a 0,10, porque aplasta las colas, y con eventos
  raros la señal está en los valores extremos. Se cambió a la estandarización común. La variante
  descartada quedó registrada como `logistica_l2_cuantiles`.
- **Contraste de hipótesis (M4):** con el t-test corregido de Nadeau y Bengio y Holm (14
  comparaciones), **ninguna diferencia de la ablación es significativa**, ni siquiera quitar el
  detrending (Δ −0,23, p = 0,04 sin ajustar y 0,52 con Holm). Con Wilcoxon (optimista, ignora la
  dependencia entre folds) y Holm sí lo son: quitar el detrending y la ventana 301 en ambos
  modelos, y quitar el recorte solo en el RF. La ponderación de clases, la cadencia, los períodos y
  las diferencias RF/XGB/KNN no son significativos con ninguna prueba. **Mensaje:** los efectos
  grandes tienen una dirección muy consistente, pero con 37 positivos el experimento no tiene
  potencia para confirmarlos con una prueba conservadora. Los efectos chicos no deben presentarse
  como mejoras.

- **⚠ El dataset tiene una fuga de información (hallazgo central para el informe, notebook 08,
  sección 4):** las 5.615 estrellas sin planeta (train y test) tienen la mediana del flujo crudo
  **exactamente en 0**, pero 31 de 37 y 3 de 5 estrellas con planeta no. La regla "mediana ≠ 0 ⇒
  planeta" logra recall 0,84 y precisión 1,0 en el entrenamiento, y 0,60 / 1,0 en el test, mejor que
  cualquiera de nuestros modelos. Por eso las cifras publicadas sobre este dataset (por ejemplo, el
  99,62 % de Priyadarshini y Puri) no son comparables. **Nuestro pipeline es inmune:** el
  preprocesamiento centra todas las curvas, y un RF con solo el "lado de arriba" de la curva
  preprocesada da PR-AUC 0,018 en CV y 0,012 en el test (azar: 0,007). Ruido, tendencia y valores
  repetidos difieren un poco entre clases, pero no separan (AP ≈ 0,01).
- **Interpretabilidad de la CNN base:** el Grad-CAM no se concentra en los tránsitos del BLS
  (×1,0 en planetas, igual que en estrellas sin planeta). En los planetas de señal débil mira los
  tramos ruidosos del final; en los de tránsitos profundos marca algunas caídas. Falsos positivos:
  test 241 (tres caídas aisladas: la red detecta caídas, no su repetición, como se planteó en el
  informe de avance), test 372 (probable binaria eclipsante) y test 486 (oscilación). La lectura es
  cualitativa (resolución de ~16 puntos, 5 planetas).
- **Errores del RF (fuera de fold):** detecta 13 de 37 planetas con el umbral de la CV. Los no
  detectados tienen un período BLS de ~22 d, ~2 tránsitos y contraste 1,6, prácticamente lo mismo que
  una estrella sin planeta típica: el BLS no engancha la repetición. **El cuello de botella es la
  extracción de features, no el clasificador** (coherente con la sensibilidad a la cadencia). Los
  falsos positivos tienen muchas caídas periódicas (~21) y SNR alto, típico de binarias eclipsantes.
- **Importancia por permutación del RF:** `fft_peak_freq`, `bls_period`, `bls_power_ratio` y
  `bls_n_transits` son las más importantes, con mucha varianza entre folds.

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
  y los positivos extra vienen de otras campañas. Eso produce la fuga de la mediana descrita
  arriba (verificada).
- **Estrella 28 (con planeta):** el BLS se apoya en una sola caída profunda. Sirve como
  ejemplo para la interpretabilidad.

# Cómo leer y presentar los resultados (acordado con el usuario el 2026-11-17)

- Los números son **modestos en términos absolutos, pero sólidos**: PR-AUC de ~0,39–0,43 frente a
  0,007 del azar (unas 55 veces mejor) y 0,07 de una regresión logística simple. En el test, la CNN
  base pone los 5 planetas entre los puestos 3 y 7 de 570 estrellas. El RF, con su umbral, detecta 1
  de cada 3 planetas, con ~70 % de precisión.
- **Debilidad que hay que reconocer:** se escapan unos 2 de cada 3 planetas, sobre todo los de período
  largo o señal débil, en los que el BLS no engancha la repetición.
- **Mensaje para el informe y la presentación:** "con una evaluación honesta, el problema es mucho más
  difícil de lo que sugiere la literatura. El mejor modelo es ~55 veces mejor que el azar, los errores
  se explican por la extracción de features y no por el clasificador, y el 99 % publicado no es
  comparable (accuracy inútil con este desbalance, más la fuga de la mediana)". La rúbrica evalúa la
  solución, la implementación y el análisis, no superar un número.

# Plan de mejoras con herramientas del ramo (acordado el 2026-11-17)

El ramo es **INF396 Introducción a la Ciencia de Datos** (https://aavelozb.github.io/inf396/):
preprocesamiento y extracción de características, inferencia (bootstrap, contraste de hipótesis,
bayesiana), CV y regularización (ridge, lasso), regresión logística, LDA, KNN, SVM, redes feedforward,
procesos gaussianos, aprendizaje no supervisado y ML causal. Las mejoras deben usar estas herramientas
y justificarse con ellas en el informe.

**Orden acordado: primero M3 y M4 (rápidas y seguras), después M1. M2 solo si sobra tiempo. M5
cuando estén los resultados de la CNN.**

- **M1. Vista plegada + red feedforward (unidad 8) — la de mayor potencial.** Plegar la curva
  preprocesada en el período del BLS (fase de −0,5 a 0,5, centrada en el tránsito) y promediarla en
  ~200 bins: todas las repeticiones del tránsito quedan apiladas. Usarla como entrada de un MLP
  (o una CNN chica) con el mismo protocolo de CV 5×5. Es la idea de "vista global/local" de
  Shallue y Vanderburg (2018), ya citado. Ataca las dos debilidades encontradas: la CNN no ve la
  repetición, y la repetición es justo lo que la vista plegada hace explícito. Tiempo: ~2–3 h más
  ~10 min de BLS.
- **M2. Features para descartar binarias eclipsantes (extracción de características).** Comparar
  la profundidad de los tránsitos pares contra los impares (en una binaria difieren), buscar un
  eclipse secundario a fase 0,5, medir la variación de profundidad entre tránsitos individuales y
  sacar el segundo y tercer mejor período del BLS. Ataca los falsos positivos (por ejemplo, el
  test 372) y los casos con un período espurio. Tiempo: ~2 h más ~10 min de BLS.
- **M3. Modelos del ramo sobre las 18 features (clasificación).** Regresión logística con L1
  (lasso, que además selecciona features) y L2, LDA, KNN y SVM, con escalado dentro del pipeline
  para que no haya fuga entre folds. Mismo protocolo (CV 5×5, seed 0). Sirve como "comparación con
  otros métodos". Tiempo: ~30 min.
- **M4. Contraste de hipótesis para comparar modelos (inferencia).** Sobre las diferencias
  pareadas por fold: prueba de Wilcoxon de rangos con signo y t-test corregido para CV repetida
  (Nadeau y Bengio, 2003), que corrige la dependencia entre folds que comparten datos de
  entrenamiento. Reemplaza el "está dentro del ruido" dicho a ojo por un p-valor. Tiempo: ~30 min.
- **M5. Ensamble CNN + RF.** Promediar los rankings: cometen errores distintos (la CNN tiene más
  recall; el RF, más precisión). Tiempo: ~20 min con los OOF de la CNN.
- **Trabajo futuro (solo mencionar en el informe):** procesos gaussianos para modelar la
  variabilidad estelar antes de buscar tránsitos (se usan en astronomía, unidad 5).

**Precaución contra el sobreajuste a la CV (obligatoria):** con 37 positivos, probar muchas
variantes y quedarse con la mejor infla los resultados. Por eso: (1) el informe reporta **todo lo
probado**, no solo lo que funcionó; (2) el **test se toca una sola vez al final**, con los modelos
ya decididos; (3) las comparaciones usan siempre la misma partición y diferencias pareadas.

# Pendiente (por prioridad), entrega el 2026-11-20

1. **M1** (vista plegada + red feedforward), en curso en el laptop. M3 y M4 ya están hechas
   (notebook 09).
2. Terminar la ablación de la CNN (PC de la casa) y escribir sus conclusiones en el notebook 07.
   La parte clásica ya está lista.
3. Elegir la mejor configuración de la CNN, correrla en CV 5×5 (~35 min en el PC de la casa), hacer
   M5 y evaluar en el test, una sola vez, todos los modelos finales con el umbral elegido en CV.
4. Interpretabilidad (notebook 08): **hecha con la CNN base.** Cuando esté la ablación de la
   CNN, cambiar `CNN_NOMBRE`/`CNN_PARAMS` a la mejor configuración, volver a ejecutar el
   notebook y revisar si cambian las conclusiones (el análisis de errores ya usará los OOF de
   la CNN). Ese cambio requiere reentrenar la CNN: ~14 min en el laptop o ~4 en el PC de la casa.
5. Informe final en `report/final/`, partiendo de `parcial_v1.tex` y agregando las secciones
   que exige el curso. Se puede empezar ya, dejando marcados los números de la CNN final. Subir el PDF.
6. README: actualizar la tabla de notebooks (07, 08 y los nuevos), los resultados y el estado.
7. Presentación de 10–15 min (lista el jueves 19). Incluir una diapositiva sobre la fuga de
   información.
8. Ajuste de hiperparámetros de RF/XGB con validación anidada liviana: prioridad baja. Si no se
   hace, justificarlo en el informe (pocos positivos, riesgo de sobreajuste a la CV).

**Se puede recortar, si se justifica en el informe:** M2, la CNN híbrida con features BLS, la
validación anidada completa y la CV 5×5 de todas las variantes de la CNN.
