"""
Visualization utilities for GiBS.

Reproduces and extends the key figures from the GiBS paper:
  - Optical response plots (transmission / scattering / absorption)
  - Latent-space scatter plots (Fig. 3a / Fig. 4)
  - Autoencoder reconstruction comparison (Fig. 2e)
  - Pillar radius map (Fig. 1a style)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Optical response
# ---------------------------------------------------------------------------

def plot_optical_response(
    wavelength: np.ndarray,
    transmission: np.ndarray,
    scattering: np.ndarray,
    absorption: np.ndarray,
    title: str = "Active Metasurface (PEDOT:PSS)",
    lam_lim: Tuple[float, float] = (0.4, 1.2),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot transmission, scattering, and absorption spectra (matches Fig. 2c / 3b-c style).

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis in microns.
    transmission, scattering, absorption : np.ndarray
        Optical efficiency spectra (same length as wavelength).
    title : str
        Axes title.
    lam_lim : tuple
        (min, max) wavelength limits for the x-axis.
    save_path : str or None
        If given, the figure is saved here.
    ax : plt.Axes or None
        Axes to draw on; a new figure is created if None.

    Returns
    -------
    plt.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(wavelength, transmission, "k", linewidth=3, label="Transmission")
    ax.plot(wavelength, scattering,   "b", linewidth=3, label="Scattering")
    ax.plot(wavelength, absorption,   "r", linewidth=3, label="Absorption")

    ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=14)
    ax.set_ylabel("Intensity", fontsize=14)
    ax.set_xlim(*lam_lim)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Latent-space scatter (Fig. 3a / Fig. 4)
# ---------------------------------------------------------------------------

def plot_latent_space(
    latent_coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = r"$\sigma_{Sca,\,PEDOT:PSS}$",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter plot of 2-D latent embeddings coloured by design family.

    Parameters
    ----------
    latent_coords : np.ndarray  shape (N, 2)
        2-D latent space coordinates.
    labels : np.ndarray  shape (N,) or None
        Integer class labels (0, 1, 2, …). If None, all points use one colour.
    label_names : list of str or None
        Display names for each label value (e.g. ['Random', 'Fourier', 'Chebyshev']).
    colors : list of str or None
        Matplotlib colour strings per class. Defaults to ['red','blue','green'].
    title : str
        Axes title.
    save_path : str or None
    ax : plt.Axes or None

    Returns
    -------
    plt.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))

    if colors is None:
        colors = ["red", "blue", "green", "orange", "purple"]

    if labels is None:
        ax.scatter(latent_coords[:, 0], latent_coords[:, 1],
                   s=20, alpha=0.7, color=colors[0])
    else:
        unique = np.unique(labels)
        for i, lab in enumerate(unique):
            mask = labels == lab
            name = label_names[i] if label_names else str(lab)
            ax.scatter(
                latent_coords[mask, 0], latent_coords[mask, 1],
                s=20, alpha=0.7,
                color=colors[i % len(colors)],
                label=name,
            )
        ax.legend(fontsize=11, markerscale=2)

    ax.set_xlabel("Latent Dimension 1", fontsize=13)
    ax.set_ylabel("Latent Dimension 2", fontsize=13)
    ax.set_title(title, fontsize=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.1)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Autoencoder reconstruction comparison (Fig. 2e)
# ---------------------------------------------------------------------------

def plot_autoencoder_reconstruction(
    wavelength: np.ndarray,
    spectrum_true: np.ndarray,
    spectrum_recon: np.ndarray,
    title: str = r"$\sigma_{Sca,\,PEDOT:PSS}$",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare one simulated spectrum with its autoencoder reconstruction.

    Parameters
    ----------
    wavelength : np.ndarray  (N,)
    spectrum_true : np.ndarray  (N,)
        Original simulated spectrum.
    spectrum_recon : np.ndarray  (N,)
        Decoder output.
    title, save_path, ax : see plot_optical_response.

    Returns
    -------
    plt.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(wavelength, spectrum_true,  "r",  linewidth=2.5, label="Simulation")
    ax.plot(wavelength, spectrum_recon, "b--", linewidth=2.5, label="Reconstruction")

    ax.set_xlabel(r"$\lambda$ (nm)", fontsize=13)
    ax.set_ylabel("Efficiency", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.1)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Pillar radius map (Fig. 1a style)
# ---------------------------------------------------------------------------

def plot_radius_map(
    X: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    title: str = "GiBS Supercell",
    colormap: str = "Reds",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualise the 2-D pillar radius distribution as a filled scatter plot.

    Active pillars (R > 0) are drawn as circles scaled to their radius.
    Suppressed pillars are omitted.

    Parameters
    ----------
    X, Y : np.ndarray  shape (nx, ny)
        Pillar coordinate grids (microns).
    R : np.ndarray  shape (nx, ny)
        Fabrication-constrained radius map.
    title : str
    colormap : str
        Matplotlib colormap for pillar colouring.
    save_path : str or None
    ax : plt.Axes or None

    Returns
    -------
    plt.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    mask = R > 0
    x_active = X[mask].ravel()
    y_active = Y[mask].ravel()
    r_active = R[mask].ravel()

    # Scale marker size to be proportional to pillar radius
    # (points² unit — tune the multiplier as needed)
    pixel_scale = 72 / (X.max() - X.min()) * ax.get_figure().get_size_inches()[0] if standalone else 20
    s = (r_active * pixel_scale) ** 2

    sc = ax.scatter(
        x_active, y_active, s=s,
        c=r_active, cmap=colormap,
        alpha=0.85, edgecolors="none",
    )
    if standalone:
        plt.colorbar(sc, ax=ax, label="Radius (μm)")

    ax.set_xlim(X.min() - 0.5, X.max() + 0.5)
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)", fontsize=12)
    ax.set_ylabel("y (μm)", fontsize=12)
    ax.set_title(title, fontsize=13)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.1)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Multi-panel summary figure (paper Fig. 2 style)
# ---------------------------------------------------------------------------

def plot_workflow_summary(
    wavelength: np.ndarray,
    spectrum_true: np.ndarray,
    spectrum_recon: np.ndarray,
    latent_coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Composite 1×2 figure showing (left) autoencoder reconstruction quality
    and (right) latent-space scatter — mirroring Fig. 2 of the paper.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_autoencoder_reconstruction(
        wavelength, spectrum_true, spectrum_recon,
        title=r"$\sigma_{Sca,\,PEDOT:PSS}$ — AE reconstruction",
        ax=axes[0],
    )
    plot_latent_space(
        latent_coords, labels=labels, label_names=label_names,
        title="Latent space embedding",
        ax=axes[1],
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
