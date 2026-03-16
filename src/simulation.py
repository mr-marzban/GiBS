"""
Lumerical FDTD simulation interface for GiBS.

Wraps the lumapi calls from Version2.ipynb into clean, reusable functions
that drive the FDTD solver, extract optical cross-sections, and compute
angle-integrated scattering / absorption efficiencies.

Requires a licensed Lumerical installation with lumapi on the Python path.
The path to lumapi.py can be set via the environment variable LUMAPI_PATH,
or passed explicitly to connect().

Example
-------
>>> sim = LumericalFDTD.connect(state=0)
>>> design = build_supercell(...)
>>> result = sim.run_single(design, incident_theta=0)
>>> print(result.scattering)
"""

import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .geometry import GiBSDesign


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    Optical cross-section spectra for a single design / angle / phase.

    Attributes
    ----------
    wavelength : np.ndarray  (N_lambda,)
        Wavelength in microns.
    transmission : np.ndarray
        Zeroth-order (ballistic) transmission efficiency.
    scattering : np.ndarray
        Total scattered (non-zeroth-order) efficiency.
    absorption : np.ndarray
        Absorbed fraction (1 - T_total - R_total).
    state : int
        Material phase: 0 = insulating PEDOT:PSS, 1 = metallic.
    incident_theta : float
        Illumination angle in degrees.
    """
    wavelength: np.ndarray
    transmission: np.ndarray
    scattering: np.ndarray
    absorption: np.ndarray
    state: int = 0
    incident_theta: float = 0.0


# ---------------------------------------------------------------------------
# FDTD interface
# ---------------------------------------------------------------------------

class LumericalFDTD:
    """
    Thin wrapper around lumapi.FDTD for GiBS simulations.

    Parameters
    ----------
    fsp_file : str or Path
        Path to the Lumerical .fsp project file for the chosen material phase.
    state : int
        0 = PEDOT:PSS insulating phase,  1 = metallic phase.
    """

    # Default .fsp filenames expected alongside this module (or in cwd)
    _FSP_FILES = {0: "Lumrcwa_0.fsp", 1: "Lumrcwa_1.fsp"}

    def __init__(self, rcwa, state: int):
        self._rcwa = rcwa
        self.state = state

    @classmethod
    def connect(
        cls,
        state: int = 0,
        fsp_file: Optional[str] = None,
        lumapi_path: Optional[str] = None,
    ) -> "LumericalFDTD":
        """
        Open a Lumerical FDTD session.

        Parameters
        ----------
        state : int
            Material phase: 0 (insulating) or 1 (metallic).
        fsp_file : str or None
            Path to .fsp file. Defaults to 'Lumrcwa_{state}.fsp' in cwd.
        lumapi_path : str or None
            Path to lumapi.py. Falls back to LUMAPI_PATH env variable.

        Returns
        -------
        LumericalFDTD instance.
        """
        lumapi = _load_lumapi(lumapi_path)

        if fsp_file is None:
            fsp_file = cls._FSP_FILES[state]
        fsp_file = str(Path(fsp_file).resolve())

        rcwa = lumapi.FDTD(filename=fsp_file)
        return cls(rcwa, state)

    def close(self):
        """Save and close the Lumerical session."""
        self._rcwa.save()
        self._rcwa.close()

    # ------------------------------------------------------------------
    # Single-angle run
    # ------------------------------------------------------------------

    def run_single(
        self,
        design: GiBSDesign,
        incident_theta: float = 0.0,
        gds_dir: str = "GDS_files",
        item: int = 0,
    ) -> SimResult:
        """
        Configure the FDTD solver with `design` parameters and run at
        one illumination angle `incident_theta`.

        Parameters
        ----------
        design : GiBSDesign
            Built GiBSDesign (coefficients already set).
        incident_theta : float
            Incidence angle in degrees.
        gds_dir : str
            Folder where GDS layout files are written.
        item : int
            Structure index used to name GDS/CSV output files.

        Returns
        -------
        SimResult
        """
        params = design.to_dict()
        coeffs = np.asarray(design.coeffs).ravel()

        Px = design.nx * design.supercell_period * 1e-6   # metres
        Py = design.ny * design.supercell_period * 1e-6

        rcwa = self._rcwa
        rcwa.switchtolayout()

        rcwa.setnamed("wavy_surf", "period x",       Px)
        rcwa.setnamed("wavy_surf", "period y",       Py)
        rcwa.setnamed("wavy_surf", "nx",             float(design.nx))
        rcwa.setnamed("wavy_surf", "ny",             float(design.ny))
        rcwa.setnamed("wavy_surf", "z span",         design.z_span * 1e-6)
        rcwa.setnamed("wavy_surf", "radius",         design.r_base * 1e-6)
        rcwa.setnamed("wavy_surf", "ax",             design.supercell_period * 1e-6)
        rcwa.setnamed("wavy_surf", "ay",             design.supercell_period * 1e-6)
        rcwa.setnamed("wavy_surf", "wx",             params.get("omega_x", 1.0) * 1e-6)
        rcwa.setnamed("wavy_surf", "wy",             params.get("omega_y", 1.0) * 1e-6)
        rcwa.setnamed("wavy_surf", "gds_fname",      f"{gds_dir}/Structure_{item}.gds")

        for i, c in enumerate(coeffs[:11]):
            rcwa.setnamed("wavy_surf", f"An{i}", float(c))

        rcwa.setnamed("RCWA", "x span", Px)
        rcwa.setnamed("RCWA", "y span", Py)
        rcwa.setnamed("RCWA", "angle theta", float(incident_theta))

        time.sleep(0.5)
        rcwa.run()

        total_energy   = rcwa.getresult("RCWA", "total_energy")
        grating_orders = rcwa.getresult("RCWA", "grating_orders")

        T0, R0, T_nm, R_nm = _extract_orders(grating_orders, total_energy, incident_theta)

        denom = np.cos(np.deg2rad(incident_theta))

        wavelength   = total_energy["lambda"][:, 0] * 1e6
        transmission = (T0 / denom)[:, 0]
        scattering   = ((R0 + T_nm + R_nm) / denom)[:, 0]
        absorption   = (1 - (T0 + R0 + T_nm + R_nm) / denom)[:, 0]

        return SimResult(
            wavelength=wavelength,
            transmission=transmission,
            scattering=scattering,
            absorption=absorption,
            state=self.state,
            incident_theta=incident_theta,
        )

    # ------------------------------------------------------------------
    # Angle-integrated run (matches paper's R_and_T_integral approach)
    # ------------------------------------------------------------------

    def run_integrated(
        self,
        design: GiBSDesign,
        n_angles: int = 16,
        theta_max: float = 90.0,
        gds_dir: str = "GDS_files",
        item: int = 0,
    ) -> SimResult:
        """
        Compute angle-integrated optical efficiencies by summing cos(theta)-
        weighted single-angle runs (Lambertian weighting).

        Parameters
        ----------
        design : GiBSDesign
        n_angles : int
            Number of discrete angles from 0 to theta_max.
        theta_max : float
            Maximum illumination angle (degrees).
        gds_dir, item : str, int
            Passed to run_single().

        Returns
        -------
        SimResult with angle-integrated transmission / scattering / absorption.
        """
        delta = theta_max / n_angles
        T0_sum = R0_sum = T_nm_sum = R_nm_sum = 0.0
        denom = 0.0
        wavelength = None

        rcwa = self._rcwa
        params = design.to_dict()
        coeffs = np.asarray(design.coeffs).ravel()
        Px = design.nx * design.supercell_period * 1e-6
        Py = design.ny * design.supercell_period * 1e-6

        rcwa.switchtolayout()
        rcwa.setnamed("wavy_surf", "period x",  Px)
        rcwa.setnamed("wavy_surf", "period y",  Py)
        rcwa.setnamed("wavy_surf", "nx",         float(design.nx))
        rcwa.setnamed("wavy_surf", "ny",         float(design.ny))
        rcwa.setnamed("wavy_surf", "z span",     design.z_span * 1e-6)
        rcwa.setnamed("wavy_surf", "radius",     design.r_base * 1e-6)
        rcwa.setnamed("wavy_surf", "ax",         design.supercell_period * 1e-6)
        rcwa.setnamed("wavy_surf", "ay",         design.supercell_period * 1e-6)
        rcwa.setnamed("wavy_surf", "wx",         params.get("omega_x", 1.0) * 1e-6)
        rcwa.setnamed("wavy_surf", "wy",         params.get("omega_y", 1.0) * 1e-6)
        rcwa.setnamed("wavy_surf", "gds_fname",  f"{gds_dir}/Structure_{item}.gds")
        for i, c in enumerate(coeffs[:11]):
            rcwa.setnamed("wavy_surf", f"An{i}", float(c))
        rcwa.setnamed("RCWA", "x span", Px)
        rcwa.setnamed("RCWA", "y span", Py)

        for i in range(n_angles):
            theta = i * delta
            cos_t = np.cos(np.deg2rad(theta))

            rcwa.switchtolayout()
            rcwa.setnamed("RCWA", "angle theta", float(theta))
            time.sleep(0.5)
            rcwa.run()

            total_energy   = rcwa.getresult("RCWA", "total_energy")
            grating_orders = rcwa.getresult("RCWA", "grating_orders")

            T0, R0, T_nm, R_nm = _extract_orders(grating_orders, total_energy, theta)

            T0_sum  += cos_t * T0
            R0_sum  += cos_t * R0
            T_nm_sum += cos_t * T_nm
            R_nm_sum += cos_t * R_nm
            denom   += cos_t

            if wavelength is None:
                wavelength = total_energy["lambda"][:, 0] * 1e6

        transmission = (T0_sum / denom)[:, 0]
        scattering   = ((R0_sum + T_nm_sum + R_nm_sum) / denom)[:, 0]
        absorption   = (1 - (T0_sum + R0_sum + T_nm_sum + R_nm_sum) / denom)[:, 0]

        return SimResult(
            wavelength=wavelength,
            transmission=transmission,
            scattering=scattering,
            absorption=absorption,
            state=self.state,
            incident_theta=-1,   # -1 signals angle-integrated
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_orders(
    grating_orders: dict,
    total_energy: dict,
    incident_theta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract zeroth-order and total-scattered T/R from Lumerical grating results.

    Returns (T0, R0, T_nm, R_nm) each of shape (N_lambda, 1).
    """
    Ts = grating_orders["Ts_grating"]
    cos_t = np.cos(np.deg2rad(incident_theta))

    # Find zeroth-order index (maximum of first wavelength row)
    max_idx = np.unravel_index(np.argmax(Ts[0, 0, :, :]), Ts[0, 0, :, :].shape)
    mi, mj = max_idx

    T0 = cos_t * 0.5 * (
        grating_orders["Ts_grating"][:, :, mi, mj]
        + grating_orders["Tp_grating"][:, :, mi, mj]
    )
    R0 = cos_t * 0.5 * (
        grating_orders["Rs_grating"][:, :, mi, mj]
        + grating_orders["Rp_grating"][:, :, mi, mj]
    )
    T_total = 0.5 * (total_energy["Ts"] + total_energy["Tp"])
    R_total = 0.5 * (total_energy["Rs"] + total_energy["Rp"])
    T_nm = cos_t * (T_total - T0)
    R_nm = cos_t * (R_total - R0)

    return T0, R0, T_nm, R_nm


def _load_lumapi(lumapi_path: Optional[str] = None):
    """Dynamically import lumapi from a user-specified or env-variable path."""
    import importlib.util, sys

    if lumapi_path is None:
        lumapi_path = os.environ.get("LUMAPI_PATH", r"C:\Program Files\Lumerical\v232\api\python\lumapi.py")

    spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
    if spec is None:
        raise FileNotFoundError(
            f"Could not find lumapi.py at '{lumapi_path}'.\n"
            "Set the LUMAPI_PATH environment variable or pass lumapi_path= explicitly."
        )
    lumapi = importlib.util.module_from_spec(spec)
    sys.modules["lumapi"] = lumapi
    spec.loader.exec_module(lumapi)
    return lumapi
