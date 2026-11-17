"""Modelos clásicos sobre las features físicas: Random Forest y XGBoost (principales) y, para comparar,
los modelos vistos en el ramo (regresión logística con L1/L2, LDA, KNN y SVM)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

FEATURE_SETS = {
    "estadisticos": "stat_",
    "fft": "fft_",
    "bls": "bls_",
    "todas": "",
}


def select_columns(features: pd.DataFrame, family: str) -> list[str]:
    """Columnas de features de una familia ('estadisticos', 'fft', 'bls' o 'todas')."""
    prefix = FEATURE_SETS[family]
    return [c for c in features.columns if c != "label" and c.startswith(prefix)]


def make_random_forest(seed: int = 0, balanced: bool = True) -> RandomForestClassifier:
    """Random Forest con ponderación de clases balanceada en cada árbol (balanced=False la quita)."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample" if balanced else None,
        n_jobs=-1,
        random_state=seed,
    )


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost que ajusta scale_pos_weight = (negativos / positivos) con los datos de entrenamiento.

    El peso se calcula en fit() para que dependa de cada fold de entrenamiento y no de todo el dataset.
    balanced=False usa scale_pos_weight = 1 (sin ponderación, para el estudio de ablación).
    """

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, balanced=True, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.balanced = balanced
        self.seed = seed

    def fit(self, X, y):
        pos = max(int(np.sum(y == 1)), 1)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=np.sum(y == 0) / pos if self.balanced else 1.0,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_xgboost(seed: int = 0, balanced: bool = True) -> BalancedXGB:
    return BalancedXGB(balanced=balanced, seed=seed)


MODELS = {"random_forest": make_random_forest, "xgboost": make_xgboost}


# ---------------------------------------------------------------------------------------------------
# Modelos del ramo, para comparar. Todos llevan antes un escalado, DENTRO del pipeline: se ajusta solo con el
# fold de entrenamiento (sin fuga hacia el de validación).
#   'estandar': resta la media y divide por la desviación (por defecto).
#   'cuantiles': transforma cada feature a distribución normal por cuantiles. Se probó primero y empeora
#       mucho (logística de ~0,3 a ~0,1 de PR-AUC): aplasta las colas, y con eventos tan raros la señal está
#       justamente en los valores extremos (SNR y contraste del BLS muy altos). Se deja para documentarlo.


class DecisionScore(BaseEstimator, ClassifierMixin):
    """Adapta un clasificador que solo da decision_function (como SVC) a la interfaz predict_proba.

    Aplica una sigmoide al puntaje: no es una probabilidad calibrada, pero conserva el orden, que es lo
    que miden PR-AUC y el umbral elegido en CV. Evita SVC(probability=True), que hace una CV interna
    lenta e inestable con tan pocos positivos.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.estimator_.decision_function(X)))
        return np.c_[1 - p, p]


def _scaler(escalado: str):
    if escalado == "cuantiles":
        return QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0)
    return StandardScaler()


def make_logistic(penalty: str = "l1", escalado: str = "estandar", seed: int = 0):
    """Regresión logística con regularización L1 (lasso: deja en 0 los coeficientes inútiles) o L2 (ridge).

    La fuerza de la regularización C se elige con una CV interna de 3 folds dentro de cada fold de
    entrenamiento (validación anidada), maximizando la precisión promedio (PR-AUC).
    """
    return make_pipeline(_scaler(escalado), LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 11), cv=3, penalty=penalty, solver="liblinear", scoring="average_precision",
        class_weight="balanced", random_state=seed, max_iter=2000))


def make_lda(escalado: str = "estandar"):
    """Análisis discriminante lineal con encogimiento (shrinkage) de la covarianza, estimado automáticamente."""
    return make_pipeline(_scaler(escalado), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def make_knn(k: int = 15, escalado: str = "estandar"):
    """K vecinos más cercanos ponderados por distancia. El puntaje es la fracción (ponderada) de vecinos con planeta."""
    return make_pipeline(_scaler(escalado), KNeighborsClassifier(n_neighbors=k, weights="distance"))


def make_svm(escalado: str = "estandar", seed: int = 0):
    """SVM con kernel RBF y peso de clase balanceado; el puntaje es la distancia al hiperplano (ver DecisionScore)."""
    return DecisionScore(make_pipeline(_scaler(escalado), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                       class_weight="balanced", random_state=seed)))


MODELOS_RAMO = {
    "logistica_l1": lambda: make_logistic("l1"),
    "logistica_l2": lambda: make_logistic("l2"),
    "lda": make_lda,
    "knn": make_knn,
    "svm": make_svm,
}
