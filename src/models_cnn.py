"""CNN 1D para clasificar curvas de luz, con interfaz tipo scikit-learn (fit / predict_proba)."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

CLIP_MIN, CLIP_MAX = -30.0, 5.0  # recorte de la entrada: evita que estrellas patológicas desestabilicen el entrenamiento


class CNN1D(nn.Module):
    """Tres bloques Conv1D -> BatchNorm -> ReLU -> MaxPool, global max pooling y capa densa.

    El global max pooling pregunta "¿algún filtro encontró su patrón en CUALQUIER parte de la curva?",
    de modo que no importa en qué instante ocurra la caída del tránsito.
    La salida es un logit (la sigmoide se aplica al predecir).
    """

    def __init__(self, channels=(16, 32, 64), kernel_sizes=(7, 5, 5), pool=4, dropout=0.3):
        super().__init__()
        layers, in_ch = [], 1
        for ch, k in zip(channels, kernel_sizes):
            layers += [nn.Conv1d(in_ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(pool)]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x):                       # x: (batch, 1, n_puntos)
        h = self.features(x)                    # (batch, canales, n_puntos reducido)
        h = h.max(dim=2).values                 # global max pooling
        return self.head(h).squeeze(1)          # logit por estrella


def _to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.clip(X, CLIP_MIN, CLIP_MAX), dtype=torch.float32).unsqueeze(1)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """Entrena la CNN 1D con pérdida ponderada por clase (pos_weight = negativos / positivos).

    Número de épocas fijo (sin early stopping): con ~7 positivos por fold, un conjunto de validación
    interno sería demasiado ruidoso, y usar el fold de evaluación para parar sería una fuga de información.
    """

    def __init__(self, epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, dropout=0.3, seed=0, verbose=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        Xt, yt = _to_tensor(X), torch.tensor(y, dtype=torch.float32)
        self.net_ = CNN1D(dropout=self.dropout)
        pos_weight = torch.tensor(float(np.sum(y == 0) / max(np.sum(y == 1), 1)))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(Xt)
        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if self.verbose:
                print(f"época {epoch + 1}/{self.epochs}  pérdida {total / n:.4f}")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.net_.eval()
        Xt = _to_tensor(X)
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(self.net_(Xt[i:i + 256])) for i in range(0, len(Xt), 256)]).numpy()
        return np.c_[1 - p, p]
