"""
Parametric basis functions for GiBS geometry generation.

Implements Fourier and Chebyshev basis expansions that parameterize
the nanopillar radius distribution R(x, y) across a metasurface supercell,
as described in Equations (1)-(3) of the GiBS paper.
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def fourier_basis(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    omega_x: float,
    omega_y: float,
    Px: float,
    Py: float,
) -> np.ndarray:
    """
    Evaluate the 2-D Fourier basis expansion (Eq. 2 in the paper).

    R(x, y) = A0 + sum_k [ A_{2k-1} * sin(k*omega_x * 2pi*x/Px + k*omega_y * 2pi*y/Py)
                          + A_{2k}   * cos(k*omega_x * 2pi*x/Px + k*omega_y * 2pi*y/Py) ]

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate grids (same shape), e.g. from np.meshgrid.
    coeffs : np.ndarray
        1-D array of length 2N+1: [A0, A1, A2, ..., A_{2N-1}, A_{2N}].
        A0 is the DC offset; (A1, A2) are the k=1 sine/cosine amplitudes, etc.
    omega_x, omega_y : float
        Spatial frequency parameters (in microns⁻¹).
    Px, Py : float
        Supercell periodicities in x and y (microns).

    Returns
    -------
    np.ndarray
        Unnormalized radius field, same shape as x/y.
    """
    assert len(coeffs) % 2 == 1, "coeffs must have odd length: [A0, A1, A2, ...]"
    N = (len(coeffs) - 1) // 2

    result = np.full_like(x, float(coeffs[0]), dtype=float)
    for k in range(1, N + 1):
        phase = k * omega_x * (2 * np.pi * x / Px) + k * omega_y * (2 * np.pi * y / Py)
        result += coeffs[2 * k - 1] * np.sin(phase)
        result += coeffs[2 * k]     * np.cos(phase)
    return result


def chebyshev_basis(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """
    Evaluate the 2-D Chebyshev basis expansion (Eq. 3 in the paper).

    phi_{nx, ny}(x, y) = T_nx(u) * T_ny(v)

    where u, v are the rescaled coordinates on [-1, 1].

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate grids (same shape).
    coeffs : np.ndarray
        2-D array of shape (Nx+1, Ny+1) containing coefficients A_{nx, ny}.
    x_min, x_max, y_min, y_max : float
        Domain boundaries for coordinate rescaling.

    Returns
    -------
    np.ndarray
        Unnormalized radius field, same shape as x/y.
    """
    u = (2 * x - (x_max + x_min)) / (x_max - x_min)
    v = (2 * y - (y_max + y_min)) / (y_max - y_min)

    Nx, Ny = np.array(coeffs).shape
    result = np.zeros_like(x, dtype=float)

    Tx = _chebyshev_polys(u, Nx - 1)  # shape (Nx, *x.shape)
    Ty = _chebyshev_polys(v, Ny - 1)  # shape (Ny, *y.shape)

    for nx in range(Nx):
        for ny in range(Ny):
            result += coeffs[nx, ny] * Tx[nx] * Ty[ny]
    return result


def _chebyshev_polys(u: np.ndarray, N: int) -> np.ndarray:
    """Compute Chebyshev polynomials T_0, T_1, ..., T_N evaluated at u."""
    polys = [np.ones_like(u, dtype=float), u.astype(float)]
    for n in range(2, N + 1):
        polys.append(2 * u * polys[-1] - polys[-2])
    return np.array(polys[:N + 1])


# ---------------------------------------------------------------------------
# Radius map generation
# ---------------------------------------------------------------------------

def generate_radius_map(
    nx: int,
    ny: int,
    supercell_period: float,
    r_base: float,
    coeffs,
    omega_x: float = 1.0,
    omega_y: float = 1.0,
    basis: str = "fourier",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2-D grid of pillar radii using the chosen basis expansion.

    Parameters
    ----------
    nx, ny : int
        Number of pillar sites along x and y within the supercell.
    supercell_period : float
        Unit-cell pitch (microns); total supercell is nx*period × ny*period.
    r_base : float
        Base radius (microns) that scales the expansion output.
    coeffs : array-like
        For 'fourier': 1-D array [A0, A1, ..., A_{2N}].
        For 'chebyshev': 2-D array of shape (Nx, Ny).
    omega_x, omega_y : float
        Spatial frequency parameters used only for the Fourier basis.
    basis : {'fourier', 'chebyshev'}
        Which basis family to use.

    Returns
    -------
    X, Y : np.ndarray  (shape nx × ny)
        Pillar coordinate grids (microns).
    R : np.ndarray  (shape nx × ny)
        Radius at each grid point (microns), before fabrication thresholding.
    """
    Px = nx * supercell_period
    Py = ny * supercell_period

    xs = np.linspace(0, Px, nx, endpoint=False) + supercell_period / 2
    ys = np.linspace(0, Py, ny, endpoint=False) + supercell_period / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    coeffs = np.asarray(coeffs, dtype=float)

    if basis == "fourier":
        raw = fourier_basis(X, Y, coeffs, omega_x, omega_y, Px, Py)
    elif basis == "chebyshev":
        raw = chebyshev_basis(X, Y, coeffs, xs.min(), xs.max(), ys.min(), ys.max())
    else:
        raise ValueError(f"Unknown basis '{basis}'. Choose 'fourier' or 'chebyshev'.")

    R = r_base * raw
    return X, Y, R


# ---------------------------------------------------------------------------
# Fabrication constraint (Eq. 4 in the paper)
# ---------------------------------------------------------------------------

def apply_fabrication_constraint(
    R: np.ndarray,
    threshold: float,
    r_low: float = 0.0,
) -> np.ndarray:
    """
    Apply the threshold-based fabrication constraint F(alpha) (Eq. 4).

    Pillars whose radius falls below `threshold` are replaced by `r_low`
    (default 0 = pillar removed). Radii above the threshold are kept as-is.

    Parameters
    ----------
    R : np.ndarray
        Raw radius map from generate_radius_map.
    threshold : float
        Minimum printable feature radius (microns).
    r_low : float
        Value assigned to sub-threshold sites (default: 0 = no pillar).

    Returns
    -------
    np.ndarray
        Fabrication-constrained radius map.
    """
    R_fab = R.copy()
    R_fab[R_fab < threshold] = r_low
    return R_fab
