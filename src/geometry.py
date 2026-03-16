"""
GiBS supercell geometry builder.

Converts a radius map R(x, y) into a structured design description that
can be exported to GDS (for e-beam lithography) or passed to Lumerical FDTD.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

from .basis import generate_radius_map, apply_fabrication_constraint


@dataclass
class GiBSDesign:
    """
    Container for a single GiBS metasurface design.

    Attributes
    ----------
    nx, ny : int
        Grid dimensions (number of pillar sites).
    supercell_period : float
        Lattice pitch in microns.
    r_base : float
        Base radius scaling factor (microns).
    coeffs : array-like
        Basis coefficients (Fourier: 1-D; Chebyshev: 2-D).
    omega_x, omega_y : float
        Spatial frequency parameters (Fourier basis only).
    basis : str
        'fourier' or 'chebyshev'.
    fab_threshold : float
        Minimum printable radius (microns). Pillars below this are removed.
    z_span : float
        Pillar height / film thickness (microns).
    X, Y : np.ndarray or None
        Pillar coordinate grids (set after calling build()).
    R_raw : np.ndarray or None
        Radius map before fabrication constraint (set after build()).
    R_fab : np.ndarray or None
        Radius map after fabrication constraint (set after build()).
    """

    nx: int = 16
    ny: int = 16
    supercell_period: float = 1.5       # microns
    r_base: float = 0.5                 # microns
    coeffs: object = field(default_factory=lambda: np.ones(11))
    omega_x: float = 1.0
    omega_y: float = 1.0
    basis: str = "fourier"
    fab_threshold: float = 0.04         # 80 nm min feature (radius)
    z_span: float = 0.4                 # microns (400 nm PEDOT:PSS)

    # Computed fields (populated by build())
    X: Optional[np.ndarray] = field(default=None, repr=False)
    Y: Optional[np.ndarray] = field(default=None, repr=False)
    R_raw: Optional[np.ndarray] = field(default=None, repr=False)
    R_fab: Optional[np.ndarray] = field(default=None, repr=False)

    def build(self) -> "GiBSDesign":
        """Generate radius maps from the stored coefficients."""
        self.X, self.Y, self.R_raw = generate_radius_map(
            self.nx,
            self.ny,
            self.supercell_period,
            self.r_base,
            self.coeffs,
            omega_x=self.omega_x,
            omega_y=self.omega_y,
            basis=self.basis,
        )
        self.R_fab = apply_fabrication_constraint(self.R_raw, self.fab_threshold)
        return self

    @property
    def supercell_size(self) -> Tuple[float, float]:
        """Total supercell dimensions (Px, Py) in microns."""
        return self.nx * self.supercell_period, self.ny * self.supercell_period

    @property
    def n_active_pillars(self) -> int:
        """Number of pillars surviving the fabrication threshold."""
        if self.R_fab is None:
            raise RuntimeError("Call build() first.")
        return int(np.sum(self.R_fab > 0))

    def active_pillar_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (x, y, r) arrays for pillars that pass the fabrication filter.
        """
        if self.R_fab is None:
            raise RuntimeError("Call build() first.")
        mask = self.R_fab > 0
        return self.X[mask], self.Y[mask], self.R_fab[mask]

    def to_dict(self) -> dict:
        """Serialize design parameters to a flat dictionary (for CSV export)."""
        coeffs = np.asarray(self.coeffs).ravel()
        d = {
            "nx": self.nx,
            "ny": self.ny,
            "supercell_period": self.supercell_period,
            "r_base": self.r_base,
            "omega_x": self.omega_x,
            "omega_y": self.omega_y,
            "basis": self.basis,
            "fab_threshold": self.fab_threshold,
            "z_span": self.z_span,
        }
        for i, c in enumerate(coeffs):
            d[f"An{i}"] = c
        return d


def build_supercell(
    nx: int = 16,
    ny: int = 16,
    supercell_period: float = 1.5,
    r_base: float = 0.5,
    coeffs=None,
    omega_x: float = 1.0,
    omega_y: float = 1.0,
    basis: str = "fourier",
    fab_threshold: float = 0.04,
    z_span: float = 0.4,
) -> GiBSDesign:
    """
    Convenience function: create and build a GiBSDesign in one call.

    Parameters
    ----------
    nx, ny : int
        Number of pillar sites along each axis (paper uses 16×16).
    supercell_period : float
        Unit-cell lattice constant (microns). Total supercell = nx*period.
    r_base : float
        Base radius scaling factor (microns).
    coeffs : array-like or None
        Basis coefficients. If None, uses random Fourier coefficients
        matching the sampling ranges in the paper.
    omega_x, omega_y : float
        Spatial frequency parameters for the Fourier basis.
    basis : str
        'fourier' or 'chebyshev'.
    fab_threshold : float
        Minimum printable radius (microns).
    z_span : float
        Pillar height / film thickness (microns).

    Returns
    -------
    GiBSDesign
        Built design (X, Y, R_raw, R_fab all populated).
    """
    if coeffs is None:
        coeffs = _random_fourier_coeffs()

    design = GiBSDesign(
        nx=nx,
        ny=ny,
        supercell_period=supercell_period,
        r_base=r_base,
        coeffs=coeffs,
        omega_x=omega_x,
        omega_y=omega_y,
        basis=basis,
        fab_threshold=fab_threshold,
        z_span=z_span,
    )
    return design.build()


def _random_fourier_coeffs(n_terms: int = 5, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample random Fourier coefficients in the ranges used in the paper
    (Version2.ipynb sampling bounds).

    An0 = 1 (DC / base)
    An1, An2  in [-0.9, +0.9]
    An3, An4  in [-0.5, +0.5]
    An5, An6  in [-0.5, +0.5]
    An7, An8  in [-0.4, +0.4]
    An9, An10 in [-0.6, +0.6]
    """
    rng = np.random.default_rng(seed)
    bounds = [
        (1.0, 1.0),      # An0  (fixed)
        (-0.9, 0.9),     # An1
        (-0.9, 0.9),     # An2
        (-0.5, 0.5),     # An3
        (-0.5, 0.5),     # An4
        (-0.5, 0.5),     # An5
        (-0.5, 0.5),     # An6
        (-0.4, 0.4),     # An7
        (-0.4, 0.4),     # An8
        (-0.6, 0.6),     # An9
        (-0.6, 0.6),     # An10
    ]
    return np.array([rng.uniform(lo, hi) for lo, hi in bounds])
