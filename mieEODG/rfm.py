"""
RFM (Reference Forward Model) interface utilities.

Provides export of AerosolXSC results to the HITRAN cross-section (.xsc) file
format for use with RFM aerosol molecules, supporting both direct km⁻¹ path
(molecule 100) and synthetic multi-tabulation approach (custom molecules with
pre-scaled cm²/molec per (T,p) node).

File format
-----------
Each tabulation consists of one 100-character header line followed by data
records of exactly 100 characters (10 values × 10 chars, ``'(10E10.3)'``).
Multiple tabulations may be concatenated in a single file.

HITRAN header fields (total 100 chars):

    Molecule name  20 char  right-justified
    ν_min          10 char  float  [cm⁻¹]
    ν_max          10 char  float  [cm⁻¹]
    N               7 char  int    number of spectral points
    T               7 char  float  [K]
    P               6 char  float  [Torr]
    σ_max          10 char  float  peak value
    Resolution      5 char  float  [cm⁻¹]
    Common name    15 char  string
    Reserved        4 char  spaces
    Broadener       3 char  'AIR'
    Reference       3 char  int (0)

RFM workflow — Single aerosol species (molecule 100, generic mode)
-------------------------------------------------------------------
τ = k_ext(ν) [km⁻¹]  ×  ATM_profile [dimensionless]  ×  path [km]

RFM workflow — Multiple aerosol species (custom molecule, multitab mode)
------------------------------------------------------------------------
τ = k_eff(ν, T, p) [cm²/molec]  ×  n_air(T,p) [molec/cm³]  ×  VMR [ppmv]  ×  path [cm]

where k_eff is pre-scaled so that ideal gas law conversion cancels:
k_eff = k_ext [km⁻¹]  ×  k_B·T / (p [Pa] × 10⁻⁷)

Multi-species capability
------------------------
RFM molecule 100 (`aerosol`) is reserved for direct km⁻¹ path (1 species).
Custom molecules (X109, X110) can use the synthetic multi-tabulation approach
to represent up to 2 additional aerosol species per run (total: 3 species).

References
----------
HITRAN cross-section format:
    https://hitran.org/docs/cross-sections-definitions/
RFM *XSC section:
    https://eodg.atm.ox.ac.uk/RFM/sum/xscsec.html
RFM *ATM section:
    https://eodg.atm.ox.ac.uk/RFM/sum/atmfil.html
Aerosol XSC interface (rfmwrapper):
    Synthetic multi-tabulation approach for multiple aerosol species
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants for multi-tabulation pre-scaling
# ---------------------------------------------------------------------------

K_B = 1.380649e-23  # Boltzmann constant [J/K]
N_A = 6.02214076e23  # Avogadro's number [mol⁻¹]
R   = 8.31446261815  # Gas constant [J/(mol·K)]

# Standard (T, p) grid for atmospheric radiative transfer
STANDARD_T_GRID_K = np.array([180, 200, 220, 240, 260, 280, 300])       # K
STANDARD_P_GRID_MBAR = np.array([1, 5, 10, 30, 100, 300, 500, 800, 1013])  # mbar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_uniform(wn: np.ndarray, tol: float = 1e-3) -> bool:
    """Return True if the wavenumber grid is uniform within ``tol`` cm⁻¹."""
    diffs = np.diff(wn)
    return bool(np.all(np.abs(diffs - diffs[0]) < tol))


def _resample_uniform(wn: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample ``profile`` onto a uniform wavenumber grid.

    Uses the same span and number of points as the original grid.

    Parameters
    ----------
    wn : np.ndarray
        Original (possibly non-uniform) wavenumber grid [cm⁻¹], ascending.
    profile : np.ndarray
        Spectral values on ``wn``.

    Returns
    -------
    wn_uniform : np.ndarray
        Uniform wavenumber grid.
    profile_uniform : np.ndarray
        Profile linearly interpolated onto ``wn_uniform``.
    """
    wn_uniform = np.linspace(wn[0], wn[-1], len(wn))
    profile_uniform = np.interp(wn_uniform, wn, profile)
    return wn_uniform, profile_uniform


def _scale_to_effective_cm2molec(
    k_ext_km_inv: np.ndarray,
    T_K: float,
    p_mbar: float,
) -> np.ndarray:
    """
    Pre-scale extinction from km⁻¹ to effective cm²/molec for RFM multi-tabulation.

    At each (T, p) node, pre-scale k_ext so that RFM's ideal gas law conversion
    cancels and the correct km⁻¹ extinction is recovered.

    Parameters
    ----------
    k_ext_km_inv : np.ndarray
        Extinction cross-section [km⁻¹], shape (n_wn,)
    T_K : float
        Temperature [K]
    p_mbar : float
        Pressure [mbar]

    Returns
    -------
    np.ndarray
        Pre-scaled effective cross-section [cm²/molec], shape (n_wn,)

    Notes
    -----
    RFM formula for user-defined molecules:
        τ = k_eff [cm²/molec] × n_air(T,p) [molec/cm³] × VMR [ppmv] × path [cm]

    Setting VMR = 1 ppmv and equating to:
        τ = k_ext [km⁻¹] × scaling × path [km]

    solves to:
        k_eff(T, p) = k_ext [km⁻¹] × k_B·T / (p [Pa] × 10⁻⁷)
    """
    p_pa = p_mbar * 100.0  # mbar → Pa
    scale_factor = K_B * T_K / (p_pa * 1e-7)
    return k_ext_km_inv * scale_factor


def _write_tabulation(
    fh,
    profile:     np.ndarray,
    wn:          np.ndarray,
    mol_name:    str,
    common_name: str,
    T_K:         float,
    P_torr:      float,
) -> None:
    """Write one HITRAN-format tabulation (header + data) to an open file."""
    n_pts   = len(wn)
    wn_min  = float(wn[0])
    wn_max  = float(wn[-1])
    delta_wn = (wn_max - wn_min) / max(n_pts - 1, 1)
    sigma_max = float(np.max(np.abs(profile)))

    # --- Header line (100 chars) ---
    # Fields: mol(20) wmin(10) wmax(10) N(7) T(7) P(6) smax(10) res(5) name(15) res(4) broad(3) ref(3)
    mol_field    = f"{mol_name:>20s}"
    wmin_field   = f"{wn_min:10.4f}"
    wmax_field   = f"{wn_max:10.4f}"
    n_field      = f"{n_pts:7d}"
    t_field      = f"{T_K:7.2f}"
    p_field      = f"{P_torr:6.1f}"  # F6.1 per HITRAN spec
    smax_field   = f"{sigma_max:10.3E}"
    res_field    = f"{delta_wn:5.2f}"
    name_field   = f"{common_name:<15s}"
    reserved     = "    "
    broad_field  = "AIR"
    ref_field    = "  0"

    header = (mol_field + wmin_field + wmax_field + n_field
              + t_field + p_field + smax_field + res_field + name_field
              + reserved + broad_field + ref_field)
    fh.write(header + "\n")

    # --- Data block: 10 values × 10 chars per line ---
    for i in range(0, n_pts, 10):
        chunk = profile[i:i + 10]
        line  = "".join(f"{v:10.3E}" for v in chunk)
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_rfm_xsc(
    xsc,
    path: str | Path,
    *,
    mode:       str = 'generic',
    quantity:   str = 'MEC',
    T_K:        float | None = None,
    p_mbar:     float | None = None,
    T_grid:     np.ndarray | None = None,
    p_grid:     np.ndarray | None = None,
) -> None:
    """
    Export an :class:`~mieEODG.aerosol.AerosolXSC` result to a HITRAN-format
    ``.xsc`` file for use with RFM, supporting both direct km⁻¹ path
    (molecule 100) and synthetic multi-tabulation approach (custom molecules).

    Parameters
    ----------
    xsc : AerosolXSC
        A computed :class:`~mieEODG.aerosol.AerosolXSC` instance
        (:meth:`~mieEODG.aerosol.AerosolXSC.compute` must have been called).
    path : str or Path
        Output file path.

    Keyword Arguments
    -----------------
    mode : {'generic', 'multitab'}
        Export mode (default ``'generic'``):

        ``'generic'``
            Single-block km⁻¹ format for RFM molecule 100 (`aerosol`).
            Output file name is forced to ``aerosol.xsc``.  The ``.atm``
            profile field must be named ``AEROSOL`` and contain dimensionless
            scaling factors (1.0 = reference extinction).

        ``'multitab'``
            Multi-block format with (T, p) grid, pre-scaled to effective
            cm²/molec per node.  Output file name taken from ``path``
            parameter (e.g., ``ash.xsc``).  The ``.atm`` profile field
            must be named after the molecule (same as filename without
            extension) and contain dimensionless scaling factors
            (1.0 = reference loading).

    quantity : {'MEC', 'MSC', 'MAC'}
        Which mass coefficient to export.  Default ``'MEC'``.

    T_K : float or None
        Reference temperature [K] for generic mode.  Defaults to
        ``xsc.ri.T_K`` if available, otherwise 296 K.  Ignored in
        multitab mode (uses T_grid instead).

    p_mbar : float or None
        Reference pressure [mbar] for generic mode (converted to Torr for
        HITRAN header).  Default 1013 mbar (≈ 760 Torr, sea level).
        Ignored in multitab mode.

    T_grid : np.ndarray or None
        Temperature nodes [K] for multitab mode.  Default: standard
        atmospheric grid [180, 200, 220, 240, 260, 280, 300] K.
        Ignored in generic mode.

    p_grid : np.ndarray or None
        Pressure nodes [mbar] for multitab mode.  Default: standard
        atmospheric grid [1, 5, 10, 30, 100, 300, 500, 800, 1013] mbar.
        Ignored in generic mode.

    Raises
    ------
    RuntimeError
        If :meth:`~mieEODG.aerosol.AerosolXSC.compute` has not been called.
    ValueError
        If ``mode`` is not 'generic' or 'multitab', or if ``quantity``
        is not 'MEC', 'MSC', or 'MAC'.

    Notes
    -----
    **Generic mode (molecule 100)**:
    Stores k_ext in km⁻¹ directly.  RFM's computation:

        τ = k_ext(ν) [km⁻¹]  ×  ATM_profile [dimensionless]  ×  path [km]

    **Multi-tabulation mode (custom molecule)**:
    Stores k_eff pre-scaled to cm²/molec at each (T, p) node, such that
    RFM's ideal gas law conversion cancels:

        τ = k_eff(ν, T, p) [cm²/molec]  ×  n_air(T,p) [molec/cm³]
            ×  VMR [ppmv]  ×  path [cm]

    recovers the correct km⁻¹ extinction at every atmospheric layer.

    **Wavenumber grid**: RFM requires a uniform grid.  If ``xsc.wn`` is
    non-uniform, the profile is automatically resampled and an INFO message
    is logged.

    **Matching .atm profile** (generic mode)::

        # In your .atm file, under *AEROSOL:
        # altitude [km]   scaling factor (dimensionless)
        0.0    1.0        # reference extinction
        1.0    0.5        # half reference
        ...

    **Matching .atm profile** (multitab mode, e.g., ``ash.xsc``)::

        # In your .atm file, under ASH:
        # altitude [km]   scaling factor (dimensionless)
        0.0    1.0        # reference loading
        1.0    0.5        # half reference
        ...
    """
    # --- Validate ---
    if quantity not in ('MEC', 'MSC', 'MAC'):
        raise ValueError(
            f"quantity must be 'MEC', 'MSC', or 'MAC'; got '{quantity}'."
        )
    if mode not in ('generic', 'multitab'):
        raise ValueError(
            f"mode must be 'generic' or 'multitab'; got '{mode}'."
        )

    # --- Extract spectral profile ---
    profile_raw: np.ndarray = getattr(xsc, quantity)   # raises RuntimeError if not computed
    wn = xsc.wn.copy()

    # Ensure ascending wavenumber order
    if wn[0] > wn[-1]:
        wn        = wn[::-1]
        profile_raw = profile_raw[::-1]

    # Resample to uniform grid if needed
    if not _check_uniform(wn):
        logger.info(
            "Wavenumber grid is non-uniform — resampling to uniform grid "
            "(%d pts, %.4f–%.4f cm⁻¹) for RFM .xsc export.",
            len(wn), wn[0], wn[-1],
        )
        wn, profile_raw = _resample_uniform(wn, profile_raw)

    # --- Metadata ---
    substance = (xsc.ri.substance or '').strip()
    common_name = substance[:15] if substance else 'Aerosol'

    # --- Generic mode (single block, km⁻¹ direct) ---
    if mode == 'generic':
        T_out = T_K if T_K is not None else (xsc.ri.T_K if xsc.ri.T_K is not None else 296.0)
        p_out_mbar = p_mbar if p_mbar is not None else 1013.0
        p_out_torr = p_out_mbar * 0.75006  # mbar → Torr

        logger.info(
            "Writing RFM .xsc (generic)  —  %s  %s  "
            "km⁻¹ direct  ν=[%.1f, %.1f] cm⁻¹  N=%d  T=%.1f K  P=%.1f mbar",
            quantity, substance or 'Aerosol', wn[0], wn[-1], len(wn), T_out, p_out_mbar,
        )

        # Force output path to aerosol.xsc
        output_path = Path(path).parent / 'aerosol.xsc'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as fh:
            _write_tabulation(
                fh,
                profile      = profile_raw,  # km⁻¹ direct
                wn           = wn,
                mol_name     = 'aerosol',
                common_name  = common_name,
                T_K          = T_out,
                P_torr       = p_out_torr,
            )
        logger.info("Saved → %s", output_path)

    # --- Multi-tabulation mode (multiple (T, p) blocks, pre-scaled cm²/molec) ---
    else:  # mode == 'multitab'
        # Default grids
        _T_grid = T_grid if T_grid is not None else STANDARD_T_GRID_K.copy()
        _p_grid = p_grid if p_grid is not None else STANDARD_P_GRID_MBAR.copy()

        # Extract molecule name from path
        path_obj = Path(path)
        mol_name = path_obj.stem.lower()  # filename without extension
        mol_name_display = mol_name[:20]

        logger.info(
            "Writing RFM .xsc (multitab)  —  %s  %s  "
            "molecule=%s  ν=[%.1f, %.1f] cm⁻¹  N=%d  "
            "(T, p) grid: %d × %d nodes = %d blocks",
            quantity, substance or 'Aerosol', mol_name,
            wn[0], wn[-1], len(wn),
            len(_T_grid), len(_p_grid), len(_T_grid) * len(_p_grid),
        )

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with path_obj.open('w') as fh:
            for T in _T_grid:
                for p in _p_grid:
                    # Pre-scale to effective cm²/molec
                    profile_scaled = _scale_to_effective_cm2molec(
                        profile_raw,  # Input assumed to be km⁻¹ equivalent
                        T,
                        p,
                    )
                    p_torr = p * 0.75006  # mbar → Torr

                    _write_tabulation(
                        fh,
                        profile      = profile_scaled,  # cm²/molec (pre-scaled)
                        wn           = wn,
                        mol_name     = mol_name_display,
                        common_name  = common_name,
                        T_K          = T,
                        P_torr       = p_torr,
                    )

        logger.info("Saved → %s  (%d blocks)", path_obj, len(_T_grid) * len(_p_grid))
