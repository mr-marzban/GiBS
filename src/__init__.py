"""
GiBS: Generative Input-side Basis-driven Structures

An inverse-design framework for large-scale nonlocal metasurfaces using
compact parametric basis representations (Fourier/Chebyshev) combined with
autoencoder-based manifold learning.
"""

__version__ = "1.0.0"
__author__ = "Reza Marzban"

from .basis import fourier_basis, chebyshev_basis, generate_radius_map, apply_fabrication_constraint
from .geometry import build_supercell, GiBSDesign
from .autoencoder import SpectralAutoencoder, train_autoencoder
from .visualization import (
    plot_optical_response,
    plot_latent_space,
    plot_radius_map,
    plot_autoencoder_reconstruction,
)

__all__ = [
    "fourier_basis",
    "chebyshev_basis",
    "generate_radius_map",
    "apply_fabrication_constraint",
    "build_supercell",
    "GiBSDesign",
    "SpectralAutoencoder",
    "train_autoencoder",
    "plot_optical_response",
    "plot_latent_space",
    "plot_radius_map",
    "plot_autoencoder_reconstruction",
]
