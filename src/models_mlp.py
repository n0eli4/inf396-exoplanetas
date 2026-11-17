"""Red neuronal feedforward (perceptrón multicapa) con interfaz tipo scikit-learn, para las vistas plegadas (mejora M1)."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from src.folding import N_GLOBAL
from src.models_cnn import class_pos_weight


class MLP(nn.Module):
    """Capas densas Linear -> ReLU -> Dropout; la salida es un logit."""

    def __init__(self, n_in, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(dropout)]
            n_in = h
        self.net = nn.Sequential(*layers, nn.Linear(n_in, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class FeedforwardClassifier(BaseEstimator, ClassifierMixin):
    """Entrena el MLP con pérdida ponderada por clase (como la CNN) y número fijo de épocas.

    Las entradas se estandarizan con la media y desviación del conjunto de entrenamiento (dentro de fit, así no
    hay fuga entre folds).
    flip_views: augmentation que invierte en el tiempo las vistas plegadas (las primeras n_views columnas) con
        probabilidad 0,5. Un tránsito es simétrico, así que la vista invertida sigue siendo válida. Solo tiene
        sentido si las primeras columnas son la vista global (N_GLOBAL) y la local, en ese orden.
    """

    def __init__(self, hidden=(64, 32), dropout=0.3, epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
                 class_weight="balanced", flip_views=False, n_local=61, seed=0):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.flip_views = flip_views
        self.n_local = n_local
        self.seed = seed

    def _flip(self, xb, gen):
        flip = torch.rand(len(xb), generator=gen) < 0.5
        g, l = N_GLOBAL, N_GLOBAL + self.n_local
        xb = xb.clone()
        xb[flip, :g] = xb[flip, :g].flip(1)
        xb[flip, g:l] = xb[flip, g:l].flip(1)
        return xb

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xt = torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.net_ = MLP(X.shape[1], self.hidden, self.dropout)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(class_pos_weight(y, self.class_weight))))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(Xt)
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = self._flip(Xt[idx], gen) if self.flip_views else Xt[idx]
                opt.zero_grad()
                loss_fn(self.net_(xb), yt[idx]).backward()
                opt.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.net_.eval()
        Xt = torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.net_(Xt)).numpy()
        return np.c_[1 - p, p]
