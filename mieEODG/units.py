"""
Unit conversion utilities for mieEODG.

Spectral grid convention
------------------------
All internal computations use wavenumber in cm⁻¹.  These helpers convert
between wavenumber and wavelength (µm or nm) and between particle radii
in cm (internal) and µm (user-facing).

Wavenumber–wavelength conversions are self-inverse (the same formula
maps wn→λ and λ→wn), relying on 1e4/x (µm) and 1e7/x (nm).

Grid helpers return a **decreasing-wavenumber** (increasing-wavelength)
1-D array consistent with the plotting convention used throughout the
package.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Scalar / array conversions
# ---------------------------------------------------------------------------

def wn_from_um(wl_um: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength in µm to wavenumber in cm⁻¹.

    Parameters
    ----------
    wl_um : float or array-like
        Wavelength [µm].

    Returns
    -------
    float or np.ndarray
        Wavenumber [cm⁻¹].
    """
    return 1.0e4 / np.asarray(wl_um, dtype=np.float64)


def wn_from_nm(wl_nm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength in nm to wavenumber in cm⁻¹.

    Parameters
    ----------
    wl_nm : float or array-like
        Wavelength [nm].

    Returns
    -------
    float or np.ndarray
        Wavenumber [cm⁻¹].
    """
    return 1.0e7 / np.asarray(wl_nm, dtype=np.float64)


def wl_um_from_wn(wn: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavenumber in cm⁻¹ to wavelength in µm.

    Parameters
    ----------
    wn : float or array-like
        Wavenumber [cm⁻¹].

    Returns
    -------
    float or np.ndarray
        Wavelength [µm].
    """
    return 1.0e4 / np.asarray(wn, dtype=np.float64)


def wl_nm_from_wn(wn: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavenumber in cm⁻¹ to wavelength in nm.

    Parameters
    ----------
    wn : float or array-like
        Wavenumber [cm⁻¹].

    Returns
    -------
    float or np.ndarray
        Wavelength [nm].
    """
    return 1.0e7 / np.asarray(wn, dtype=np.float64)


def r_cm_from_um(r_um: float | np.ndarray) -> float | np.ndarray:
    """
    Convert particle radius from µm to cm.

    Parameters
    ----------
    r_um : float or array-like
        Radius [µm].

    Returns
    -------
    float or np.ndarray
        Radius [cm].
    """
    return np.asarray(r_um, dtype=np.float64) * 1.0e-4


def r_um_from_cm(r_cm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert particle radius from cm to µm.

    Parameters
    ----------
    r_cm : float or array-like
        Radius [cm].

    Returns
    -------
    float or np.ndarray
        Radius [µm].
    """
    return np.asarray(r_cm, dtype=np.float64) * 1.0e4


# ---------------------------------------------------------------------------
# Grid builders  (return decreasing-wavenumber arrays)
# ---------------------------------------------------------------------------

def wn_grid_from_um(
    wl_min_um: float,
    wl_max_um: float,
    step_um: float,
) -> np.ndarray:
    """
    Build a wavenumber grid from a wavelength range specified in µm.

    The output grid runs from ``wn_from_um(wl_max_um)`` (low wavenumber)
    to ``wn_from_um(wl_min_um)`` (high wavenumber), sampled on a uniform
    **wavelength** grid with spacing *step_um*, then converted and sorted
    in ascending wavenumber order.

    Parameters
    ----------
    wl_min_um : float
        Shortest wavelength [µm] (highest wavenumber end).
    wl_max_um : float
        Longest wavelength [µm] (lowest wavenumber end).
    step_um : float
        Wavelength step [µm].

    Returns
    -------
    np.ndarray
        Wavenumber grid [cm⁻¹], sorted ascending.

    Examples
    --------
    >>> wn = wn_grid_from_um(8.0, 14.0, 0.01)
    >>> wn.min(), wn.max()
    (714.29..., 1250.0)
    """
    wl = np.arange(wl_min_um, wl_max_um + step_um / 2.0, step_um)
    wn = wn_from_um(wl)
    return np.sort(wn)


def wn_grid_from_nm(
    wl_min_nm: float,
    wl_max_nm: float,
    step_nm: float,
) -> np.ndarray:
    """
    Build a wavenumber grid from a wavelength range specified in nm.

    Parameters
    ----------
    wl_min_nm : float
        Shortest wavelength [nm] (highest wavenumber end).
    wl_max_nm : float
        Longest wavelength [nm] (lowest wavenumber end).
    step_nm : float
        Wavelength step [nm].

    Returns
    -------
    np.ndarray
        Wavenumber grid [cm⁻¹], sorted ascending.
    """
    wl = np.arange(wl_min_nm, wl_max_nm + step_nm / 2.0, step_nm)
    wn = wn_from_nm(wl)
    return np.sort(wn)
