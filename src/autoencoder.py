"""
Autoencoder for spectral manifold learning in GiBS.

Implements the symmetric fully-connected autoencoder described in Section 2.4
of the paper. Four independent models are trained, one for each combination of
response type (absorption, scattering) and material phase (insulating, metallic).

Architecture  (encoder path):
    201 -> 90 -> 60 -> 50 -> 20 -> 2   (tanh activations)
Decoder mirrors back to 201 with a linear output layer.

Loss function (Eq. 5):
    L = lambda1 * (1 - cosine_similarity) + lambda2 * (1/N) * L1
    lambda1 = 0.7,  lambda2 = 0.3
"""

import numpy as np
from typing import Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, backend as K
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def gibs_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray,
                    lambda1: float = 0.7, lambda2: float = 0.3) -> float:
    """
    GiBS combined spectral loss (Eq. 5), evaluated in NumPy.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  shape (N, 201)
    lambda1, lambda2 : float
        Weighting between cosine similarity and L1 terms.

    Returns
    -------
    float
        Mean loss over the batch.
    """
    norm_true = np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-8
    norm_pred = np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8
    cosine_sim = np.sum((y_true / norm_true) * (y_pred / norm_pred), axis=1)
    cosine_loss = 1 - cosine_sim

    l1_loss = np.mean(np.abs(y_true - y_pred), axis=1)

    return float(np.mean(lambda1 * cosine_loss + lambda2 * l1_loss))


# ---------------------------------------------------------------------------
# PyTorch implementation (primary, no extra install needed for most users)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _GiBSLoss(nn.Module):
        def __init__(self, lambda1: float = 0.7, lambda2: float = 0.3):
            super().__init__()
            self.lambda1 = lambda1
            self.lambda2 = lambda2

        def forward(self, y_pred: "torch.Tensor", y_true: "torch.Tensor") -> "torch.Tensor":
            norm_true = y_true.norm(dim=1, keepdim=True).clamp(min=1e-8)
            norm_pred = y_pred.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cos_sim = (y_true / norm_true * y_pred / norm_pred).sum(dim=1)
            cosine_loss = 1 - cos_sim
            l1_loss = (y_true - y_pred).abs().mean(dim=1)
            return (self.lambda1 * cosine_loss + self.lambda2 * l1_loss).mean()

    class SpectralAutoencoder(nn.Module):
        """
        Symmetric fully-connected autoencoder for spectral dimensionality reduction.

        Encoder:  201 -> 90 -> 60 -> 50 -> 20 -> latent_dim  (tanh)
        Decoder:  latent_dim -> 20 -> 50 -> 60 -> 90 -> 201   (tanh / linear)
        """

        def __init__(self, input_dim: int = 201, latent_dim: int = 2):
            super().__init__()
            hidden = [90, 60, 50, 20]

            # Encoder
            enc_layers = []
            prev = input_dim
            for h in hidden:
                enc_layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            enc_layers += [nn.Linear(prev, latent_dim)]
            self.encoder = nn.Sequential(*enc_layers)

            # Decoder (mirrors encoder)
            dec_layers = []
            prev = latent_dim
            for h in reversed(hidden):
                dec_layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            dec_layers += [nn.Linear(prev, input_dim)]   # linear output
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(self.encoder(x))

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(x)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(z)

else:
    class SpectralAutoencoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for SpectralAutoencoder. "
                "Install it with: pip install torch"
            )


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_autoencoder(
    spectra: np.ndarray,
    latent_dim: int = 2,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    lambda1: float = 0.7,
    lambda2: float = 0.3,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Tuple["SpectralAutoencoder", dict]:
    """
    Train a SpectralAutoencoder on a batch of optical spectra.

    Parameters
    ----------
    spectra : np.ndarray  shape (N_samples, 201)
        Normalised spectral responses (absorption or scattering cross-sections).
    latent_dim : int
        Bottleneck size. Paper uses 2 for visualisation.
    epochs : int
        Training epochs (paper uses 20).
    batch_size : int
        Mini-batch size.
    lr : float
        Adam learning rate.
    val_split : float
        Fraction of data held out for validation.
    lambda1, lambda2 : float
        Loss weights (paper values: 0.7, 0.3).
    verbose : bool
        Print epoch losses.
    device : str or None
        'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    model : SpectralAutoencoder
        Trained model.
    history : dict
        {'train_loss': [...], 'val_loss': [...]} per epoch.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. pip install torch")

    spectra = np.asarray(spectra, dtype=np.float32)

    # Normalise each spectrum to unit max to match paper convention
    spec_max = spectra.max(axis=1, keepdims=True).clip(min=1e-8)
    spectra_norm = spectra / spec_max

    n_val = int(len(spectra_norm) * val_split)
    idx = np.random.permutation(len(spectra_norm))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(spectra_norm[train_idx])
    X_val   = torch.tensor(spectra_norm[val_idx])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpectralAutoencoder(input_dim=spectra.shape[1], latent_dim=latent_dim).to(device)
    criterion = _GiBSLoss(lambda1=lambda1, lambda2=lambda2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        train_loss = running / len(X_train)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, X_val.to(device)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

    return model, history


def encode_spectra(
    model: "SpectralAutoencoder",
    spectra: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Project spectra into the 2-D latent space.

    Parameters
    ----------
    model : SpectralAutoencoder
        Trained model.
    spectra : np.ndarray  shape (N, 201)
    device : str or None

    Returns
    -------
    np.ndarray  shape (N, latent_dim)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t = torch.tensor(spectra.astype(np.float32)).to(device)
        z = model.encode(t)
    return z.cpu().numpy()
