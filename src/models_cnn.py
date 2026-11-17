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


def augment_batch(xb: torch.Tensor, gen: torch.Generator, noise_std: float = 0.1) -> torch.Tensor:
    """Data augmentation para series de tiempo, aplicada a cada curva del lote por separado.

    - Desplazamiento circular aleatorio: el tránsito puede ocurrir en cualquier instante.
    - Inversión temporal con probabilidad 0,5: un tránsito es (aprox.) simétrico, así que la
      curva invertida sigue siendo una estrella con planeta.
    - Ruido gaussiano de desviación noise_std (en unidades de la desviación robusta, ya normalizada).
    """
    n, _, length = xb.shape
    shifts = torch.randint(0, length, (n,), generator=gen)
    idx = (torch.arange(length).unsqueeze(0) + shifts.unsqueeze(1)) % length   # (n, length)
    xb = torch.gather(xb, 2, idx.unsqueeze(1))
    flip = torch.rand(n, generator=gen) < 0.5
    xb = torch.where(flip[:, None, None], xb.flip(2), xb)
    if noise_std > 0:
        xb = xb + noise_std * torch.randn(xb.shape, generator=gen)
    return xb


def class_pos_weight(y: np.ndarray, mode: str = "balanced") -> float:
    """Peso de la clase positiva en la pérdida.

    'balanced': negativos / positivos (~136); 'sqrt': su raíz (~12), un término medio que penaliza
    menos las falsas alarmas; 'none': sin ponderación (1).
    """
    ratio = np.sum(y == 0) / max(np.sum(y == 1), 1)
    return {"balanced": ratio, "sqrt": np.sqrt(ratio), "none": 1.0}[mode]


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """Entrena la CNN 1D con pérdida ponderada por clase (pos_weight según class_weight).

    Número de épocas fijo (sin early stopping): con ~7 positivos por fold, un conjunto de validación
    interno sería demasiado ruidoso, y usar el fold de evaluación para parar sería una fuga de información.

    Opciones (todas apagadas por defecto, equivalen a la versión inicial del informe de avance):
        augment: aplica augment_batch a cada lote de entrenamiento.
        class_weight: 'balanced', 'sqrt' o 'none' (ver class_pos_weight).
        cosine_lr: tasa de aprendizaje decreciente (coseno) a lo largo de las épocas.
    """

    def __init__(self, epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, dropout=0.3, augment=False,
                 noise_std=0.1, class_weight="balanced", cosine_lr=False, seed=0, verbose=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.augment = augment
        self.noise_std = noise_std
        self.class_weight = class_weight
        self.cosine_lr = cosine_lr
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        Xt, yt = _to_tensor(X), torch.tensor(y, dtype=torch.float32)
        self.net_ = CNN1D(dropout=self.dropout)
        pos_weight = torch.tensor(float(class_pos_weight(y, self.class_weight)))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs) if self.cosine_lr else None
        n = len(Xt)
        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = augment_batch(Xt[idx], gen, self.noise_std) if self.augment else Xt[idx]
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if sched is not None:
                sched.step()
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
