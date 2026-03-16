"""
GiBS Demo — No Lumerical required.

This script walks through the full GiBS pipeline using synthetic data
so readers can explore the framework without a Lumerical licence:

  1. Generate a random Fourier-basis GiBS supercell
  2. Visualise the pillar radius map
  3. Generate a synthetic scattering dataset (random + Fourier + Chebyshev)
  4. Train a 2-D spectral autoencoder
  5. Plot the latent-space embedding  (reproduces Fig. 3a style)
  6. Show autoencoder reconstruction quality  (reproduces Fig. 2e style)

Paper figures from the PDF are displayed alongside the generated outputs.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Make src importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from basis import generate_radius_map, apply_fabrication_constraint
from geometry import build_supercell, _random_fourier_coeffs
from dataset import generate_random_dataset
from autoencoder import train_autoencoder, encode_spectra
from visualization import (
    plot_radius_map,
    plot_optical_response,
    plot_latent_space,
    plot_autoencoder_reconstruction,
)

FIGS_DIR = Path(__file__).parent / "figs"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 1. Build a Fourier-basis supercell (paper Fig. 1a coefficients)
# ===========================================================================
print("=" * 60)
print("Step 1 — Build a GiBS supercell (Fourier basis)")
print("=" * 60)

# Coefficients from Fig. 1(c) of the paper
paper_coeffs = np.array([0.0, 0.0, 0.0, 0.15, 0.4, -0.5, -0.1, -0.4, -0.1, 0.6, 0.6])
omega_x = 0.22   # µm (from Fig. 1c)
omega_y = 0.33   # µm

design = build_supercell(
    nx=16, ny=16,
    supercell_period=1.5,   # µm  →  total supercell ≈ 24 µm
    r_base=0.5,
    coeffs=paper_coeffs,
    omega_x=omega_x,
    omega_y=omega_y,
    basis="fourier",
    fab_threshold=0.04,     # 80 nm minimum radius
    z_span=0.4,             # 400 nm PEDOT:PSS film
)
print(f"  Active pillars: {design.n_active_pillars} / {design.nx * design.ny}")
print(f"  Supercell size: {design.supercell_size[0]:.1f} × {design.supercell_size[1]:.1f} µm")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_radius_map(design.X, design.Y, design.R_fab,
                title="GiBS Supercell — Fourier basis (Fig. 1a)",
                ax=axes[0])

paper_fig1 = FIGS_DIR / "fig1_supercell.png"
if paper_fig1.exists():
    img = mpimg.imread(str(paper_fig1))
    axes[1].imshow(img)
    axes[1].axis("off")
    axes[1].set_title("Paper Fig. 1 — reference", fontsize=12)
else:
    axes[1].text(0.5, 0.5, "Place paper Fig. 1\nin figs/fig1_supercell.png",
                 ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
    axes[1].axis("off")

plt.suptitle("GiBS: Parametric Basis Supercell", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step1_supercell.png", dpi=150)
plt.show()
print()


# ===========================================================================
# 1b. Chebyshev-basis supercell
# ===========================================================================
print("Step 1b — Chebyshev-basis supercell")
cheb_coeffs = np.array([
    [1.0,  0.3, -0.2],
    [0.4, -0.1,  0.2],
    [-0.3, 0.15, 0.05],
])
design_cheb = build_supercell(
    nx=16, ny=16,
    supercell_period=0.6,
    r_base=0.25,
    coeffs=cheb_coeffs,
    basis="chebyshev",
    fab_threshold=0.04,
    z_span=0.4,
)
print(f"  Active pillars (Chebyshev): {design_cheb.n_active_pillars} / {design_cheb.nx * design_cheb.ny}")

fig, ax = plt.subplots(figsize=(6, 6))
plot_radius_map(design_cheb.X, design_cheb.Y, design_cheb.R_fab,
                title="GiBS Supercell — Chebyshev basis",
                ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step1b_chebyshev_supercell.png", dpi=150)
plt.show()
print()


# ===========================================================================
# 2. Synthetic optical response (stand-in for FDTD)
# ===========================================================================
print("=" * 60)
print("Step 2 — Synthetic optical response (no Lumerical)")
print("=" * 60)

wavelength = np.linspace(0.4, 1.2, 201)
# Plausible broadband scatterer spectrum
scattering   = 0.55 * np.exp(-0.5 * ((wavelength - 0.65) / 0.22) ** 2) + 0.05
transmission = 1.0 - scattering - 0.08
absorption   = 1.0 - scattering - transmission
scattering   = np.clip(scattering, 0, 1)
transmission = np.clip(transmission, 0, 1)
absorption   = np.clip(absorption, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_optical_response(wavelength, transmission, scattering, absorption,
                      title="Active Metasurface (PEDOT:PSS) — synthetic",
                      ax=axes[0])

paper_fig3b = FIGS_DIR / "fig3b_optical_response.png"
if paper_fig3b.exists():
    img = mpimg.imread(str(paper_fig3b))
    axes[1].imshow(img)
    axes[1].axis("off")
    axes[1].set_title("Paper Fig. 3b — reference", fontsize=12)
else:
    axes[1].text(0.5, 0.5, "Place paper Fig. 3b\nin figs/fig3b_optical_response.png",
                 ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
    axes[1].axis("off")

plt.suptitle("GiBS: Optical Response", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step2_optical_response.png", dpi=150)
plt.show()
print()


# ===========================================================================
# 3. Generate synthetic dataset — random / Fourier / Chebyshev
# ===========================================================================
print("=" * 60)
print("Step 3 — Generate synthetic spectral dataset")
print("=" * 60)

N_PER_CLASS = 400
wav, coeffs_rand, spectra_rand   = generate_random_dataset(N_PER_CLASS, seed=0)
_, coeffs_fourier, spectra_fourier = generate_random_dataset(N_PER_CLASS, seed=1)
_, coeffs_cheb,    spectra_cheb    = generate_random_dataset(N_PER_CLASS, seed=2)

all_spectra = np.concatenate([spectra_rand, spectra_fourier, spectra_cheb], axis=0)
all_labels  = np.concatenate([
    np.zeros(N_PER_CLASS, dtype=int),
    np.ones(N_PER_CLASS, dtype=int),
    np.full(N_PER_CLASS, 2, dtype=int),
])
print(f"  Dataset: {len(all_spectra)} spectra  ({N_PER_CLASS} per class)")
print()


# ===========================================================================
# 4. Train autoencoder
# ===========================================================================
print("=" * 60)
print("Step 4 — Train spectral autoencoder")
print("=" * 60)

try:
    model, history = train_autoencoder(
        all_spectra,
        latent_dim=2,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        val_split=0.2,
        verbose=True,
    )
    print()

    # Plot training curves
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step4_training.png", dpi=150)
    plt.show()

    AE_TRAINED = True
except ImportError as e:
    print(f"  [Skip] PyTorch not available: {e}")
    AE_TRAINED = False


# ===========================================================================
# 5. Latent-space embedding (Fig. 3a style)
# ===========================================================================
if AE_TRAINED:
    print("=" * 60)
    print("Step 5 — Latent-space embedding")
    print("=" * 60)

    z = encode_spectra(model, all_spectra)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_latent_space(
        z, labels=all_labels,
        label_names=["Random", "Fourier (cosine)", "Chebyshev"],
        colors=["red", "blue", "green"],
        title=r"$\sigma_{Sca,\,PEDOT:PSS}$ — latent space",
        ax=axes[0],
    )

    paper_fig3a = FIGS_DIR / "fig3a_latent_space.png"
    if paper_fig3a.exists():
        img = mpimg.imread(str(paper_fig3a))
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Paper Fig. 3a — reference", fontsize=12)
    else:
        axes[1].text(0.5, 0.5, "Place paper Fig. 3a\nin figs/fig3a_latent_space.png",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
        axes[1].axis("off")

    plt.suptitle("GiBS: Latent-Space Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step5_latent_space.png", dpi=150)
    plt.show()
    print()


# ===========================================================================
# 6. Autoencoder reconstruction quality (Fig. 2e style)
# ===========================================================================
if AE_TRAINED:
    print("=" * 60)
    print("Step 6 — Autoencoder reconstruction quality")
    print("=" * 60)

    import torch
    model.eval()
    idx = 42
    sample = torch.tensor(all_spectra[[idx]].astype("float32"))
    with torch.no_grad():
        recon = model(sample).cpu().numpy()[0]
    original = all_spectra[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_autoencoder_reconstruction(
        wav * 1000,   # convert µm → nm for axis label
        original, recon,
        title=r"$\sigma_{Sca,\,PEDOT:PSS}$",
        ax=axes[0],
    )

    paper_fig2e = FIGS_DIR / "fig2e_reconstruction.png"
    if paper_fig2e.exists():
        img = mpimg.imread(str(paper_fig2e))
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Paper Fig. 2e — reference", fontsize=12)
    else:
        axes[1].text(0.5, 0.5, "Place paper Fig. 2e\nin figs/fig2e_reconstruction.png",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
        axes[1].axis("off")

    plt.suptitle("GiBS: Autoencoder Reconstruction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step6_reconstruction.png", dpi=150)
    plt.show()

print("=" * 60)
print("Demo complete.  Figures saved to:", OUTPUT_DIR)
print("=" * 60)
