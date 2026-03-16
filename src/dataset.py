"""
Dataset utilities for GiBS.

Handles saving and loading design parameters and optical responses in the
CSV format used in Version2.ipynb (one CSV per structure, per response type).
Also provides helpers to assemble batches for autoencoder training.
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .geometry import GiBSDesign, _random_fourier_coeffs


# ---------------------------------------------------------------------------
# Save / load a single design
# ---------------------------------------------------------------------------

def save_design_csv(design: GiBSDesign, item: int, root: str = ".") -> None:
    """
    Save design parameters to a CSV file matching the Version2.ipynb format.

    File is written to:
        {root}/Design_parameters/Design_{item}.csv

    Parameters
    ----------
    design : GiBSDesign
    item : int
        Structure index used as the filename suffix.
    root : str
        Root directory for the dataset.
    """
    out_dir = Path(root) / "Design_parameters"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = design.to_dict()
    # Build the Parameter/Value table used in the notebook
    rows = [(str(k), float(v) if not isinstance(v, str) else v)
            for k, v in d.items()]
    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    df.to_csv(out_dir / f"Design_{item}.csv", index=False)


def load_design_csv(item: int, root: str = ".") -> dict:
    """
    Load design parameters saved by save_design_csv.

    Returns a plain dict {parameter_name: value}.
    """
    path = Path(root) / "Design_parameters" / f"Design_{item}.csv"
    params = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)   # skip header
        for row in reader:
            if len(row) == 2:
                key, val = row
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params


# ---------------------------------------------------------------------------
# Save / load spectral responses
# ---------------------------------------------------------------------------

def save_response_csv(
    wavelength: np.ndarray,
    values: np.ndarray,
    item: int,
    response_type: str,   # 'Scattering', 'Absorption', 'Zeroth_order'
    root: str = ".",
    phase: int = 1,       # 1 = insulating, 2 = metallic (matches notebook dirs)
) -> None:
    """
    Save a spectral response to CSV.

    File is written to:
        {root}/Response_{phase}/{response_type}/{response_type}_structure_{item}.csv
    """
    suffix = "" if phase == 1 else "_2"
    out_dir = Path(root) / f"Response{suffix}" / response_type
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"Lambda": wavelength, response_type: values})
    df.to_csv(out_dir / f"{response_type}_structure_{item}.csv", index=False)


def load_response_csv(
    item: int,
    response_type: str,
    root: str = ".",
    phase: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a spectral response saved by save_response_csv.

    Returns (wavelength, values) arrays.
    """
    suffix = "" if phase == 1 else "_2"
    path = (
        Path(root)
        / f"Response{suffix}"
        / response_type
        / f"{response_type}_structure_{item}.csv"
    )
    df = pd.read_csv(path)
    return df["Lambda"].values, df[response_type].values


# ---------------------------------------------------------------------------
# Batch assembly for autoencoder training
# ---------------------------------------------------------------------------

def load_spectra_batch(
    items: List[int],
    response_type: str = "Scattering",
    root: str = ".",
    phase: int = 1,
    n_wavelengths: int = 201,
) -> np.ndarray:
    """
    Load a batch of spectral responses and stack into a 2-D array.

    Parameters
    ----------
    items : List[int]
        Structure indices to load.
    response_type : str
        One of 'Scattering', 'Absorption', 'Zeroth_order'.
    root : str
        Dataset root directory.
    phase : int
        1 = insulating, 2 = metallic.
    n_wavelengths : int
        Expected number of wavelength points (spectra will be resampled
        to this length if they differ).

    Returns
    -------
    np.ndarray  shape (len(items), n_wavelengths)
    """
    spectra = []
    for i in items:
        try:
            lam, vals = load_response_csv(i, response_type, root=root, phase=phase)
            if len(vals) != n_wavelengths:
                # Resample to uniform grid
                lam_new = np.linspace(lam.min(), lam.max(), n_wavelengths)
                vals = np.interp(lam_new, lam, vals)
            spectra.append(vals.astype(np.float32))
        except FileNotFoundError:
            pass   # skip missing files

    if not spectra:
        raise ValueError(f"No valid spectra found for items {items[:5]}... in {root!r}")
    return np.stack(spectra)


# ---------------------------------------------------------------------------
# Random dataset generation (without Lumerical — for demo/testing)
# ---------------------------------------------------------------------------

def generate_random_dataset(
    n_designs: int = 100,
    n_wavelengths: int = 201,
    lam_min: float = 0.4,
    lam_max: float = 1.2,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic GiBS spectra without Lumerical (for demos).

    Each design is a random Fourier-basis supercell; its "spectrum" is a
    smooth synthetic curve that mimics broadband PEDOT:PSS scattering.

    Parameters
    ----------
    n_designs : int
    n_wavelengths : int
    lam_min, lam_max : float
        Wavelength range (microns).
    seed : int or None

    Returns
    -------
    wavelength : np.ndarray  (n_wavelengths,)
    coeffs_matrix : np.ndarray  (n_designs, 11)  — design parameters
    spectra : np.ndarray  (n_designs, n_wavelengths) — scattering-like spectra
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(lam_min, lam_max, n_wavelengths)

    coeffs_list = []
    spectra = []

    for _ in range(n_designs):
        c = _random_fourier_coeffs(seed=int(rng.integers(0, 2**31)))
        coeffs_list.append(c)

        # Synthetic smooth spectrum: Gaussian envelope + harmonic ripple
        lam_peak = rng.uniform(0.5, 0.9)
        width    = rng.uniform(0.1, 0.35)
        amp      = rng.uniform(0.3, 0.85)
        ripple   = rng.uniform(0.02, 0.08) * np.sin(
            2 * np.pi * (wavelength - lam_min) / rng.uniform(0.3, 0.7)
        )
        base = amp * np.exp(-0.5 * ((wavelength - lam_peak) / width) ** 2) + ripple
        base = np.clip(base, 0, 1)
        spectra.append(base.astype(np.float32))

    return wavelength, np.array(coeffs_list), np.array(spectra)
