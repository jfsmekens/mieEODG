"""
Shared configuration and utilities for mieEODG demo scripts.

Consolidates spectral ranges, size distribution presets, output handling,
and plotting utilities to reduce duplication across demos.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.colors as mcolors

# ============================================================================
# Spectral range presets
# ============================================================================

@dataclass
class SpectralRange:
    """Define a spectral range with typical wavenumber bounds and default step."""
    wn_min: float       # cm⁻¹
    wn_max: float       # cm⁻¹
    wn_step: float      # cm⁻¹ (default spectral resolution)
    description: str

# ============================================================================
# GLOBAL DEFAULTS  (do not edit)
# ============================================================================

# Built from the canonical presets in mieEODG.constants (single source of truth).
from mieEODG.constants import SPECTRAL_RANGE_PRESETS as _PRESETS
SPECTRAL_RANGES = {name: SpectralRange(**spec) for name, spec in _PRESETS.items()}

# ============================================================================
# OVERRIDE DEFAULTS  (set to non-None to override for all demo scripts)
# ============================================================================
# These settings apply globally to all demo scripts. Set to None to use the
# defaults defined in SPECTRAL_RANGES, SIZE_DISTRIBUTION_PRESETS, etc.
# Individual demo scripts can also override these at the call site.
#
# Example: To use mid-IR for all demos without editing each one, set:
#   OVERRIDE_SPECTRAL_RANGE = 'mid_ir'
#
# Example: To use coarser spectral resolution for faster testing:
#   OVERRIDE_WN_STEP = 10.0

OVERRIDE_SPECTRAL_RANGE: str | None = None
"""Spectral range preset: 'thermal_ir', 'mid_ir', 'solar', 'visible' (None = use default)."""

OVERRIDE_WN_STEP: float | None = None
"""Wavenumber step [cm⁻¹]. If set, overrides the preset default (None = use default)."""

OVERRIDE_SIZE_PRESET: str | None = None
"""Size distribution preset: 'fine', 'coarse', 'full', 'aerosol' (None = use default)."""

OVERRIDE_N_NODES: int | None = None
"""Number of LUT nodes. If set, overrides the preset default (None = use default)."""

OVERRIDE_BACKEND: str | None = None
"""Mie backend: 'python', 'fortran'. If set, overrides demo defaults (None = use default)."""

OVERRIDE_OUTPUT_DIR: Path | None = None
"""Output directory. If set, all demos write here instead of demo_output/ (None = use default)."""


def make_wavenumber_grid(
    spectral_range: str | None = None,
    wn_step: float | None = None,
) -> np.ndarray:
    """
    Build a descending wavenumber grid for a spectral range.

    Respects global OVERRIDE_* settings if no explicit arguments provided.

    Parameters
    ----------
    spectral_range : str or None
        Key from SPECTRAL_RANGES (e.g., 'thermal_ir', 'mid_ir').
        If None, uses OVERRIDE_SPECTRAL_RANGE or defaults to 'thermal_ir'.
    wn_step : float or None
        Spectral resolution [cm⁻¹]. If None, uses OVERRIDE_WN_STEP or preset default.

    Returns
    -------
    np.ndarray
        Wavenumber array in descending order (high to low), shape (n_wn,).
    """
    # Apply overrides if arguments not explicitly provided
    if spectral_range is None:
        spectral_range = OVERRIDE_SPECTRAL_RANGE or 'thermal_ir'

    if spectral_range not in SPECTRAL_RANGES:
        raise ValueError(
            f"Unknown spectral_range '{spectral_range}'. "
            f"Must be one of: {list(SPECTRAL_RANGES.keys())}"
        )

    cfg = SPECTRAL_RANGES[spectral_range]
    if wn_step is None:
        wn_step = OVERRIDE_WN_STEP or cfg.wn_step

    # Build uniformly-spaced wavenumber grid in descending order
    wn = np.arange(cfg.wn_max, cfg.wn_min - wn_step / 2, -wn_step)
    return wn


# ============================================================================
# Size distribution presets
# ============================================================================

@dataclass
class SizeDistributionPreset:
    """Define a size distribution sweep range."""
    r0_min_um: float
    r0_max_um: float
    n_nodes: int = 10        # LUT nodes (log-spaced)
    description: str = ""

SIZE_DISTRIBUTION_PRESETS = {
    'fine': SizeDistributionPreset(
        r0_min_um=0.01,
        r0_max_um=1.0,
        n_nodes=8,
        description='Fine mode (10 nm – 1 µm)'
    ),
    'coarse': SizeDistributionPreset(
        r0_min_um=0.5,
        r0_max_um=10.0,
        n_nodes=12,
        description='Coarse mode (0.5 – 10 µm)'
    ),
    'full': SizeDistributionPreset(
        r0_min_um=0.1,
        r0_max_um=10.0,
        n_nodes=12,
        description='Full range (100 nm – 10 µm)'
    ),
    'aerosol': SizeDistributionPreset(
        r0_min_um=0.1,
        r0_max_um=2.0,
        n_nodes=10,
        description='Aerosol-relevant (100 nm – 2 µm)'
    ),
}


def get_r0_grid(
    size_preset: str | None = None,
    r0_min_um: float | None = None,
    r0_max_um: float | None = None,
    n_nodes: int | None = None,
) -> np.ndarray:
    """
    Build a log-spaced radius grid for size distributions.

    Respects global OVERRIDE_* settings if no explicit arguments provided.

    Parameters
    ----------
    size_preset : str or None
        Key from SIZE_DISTRIBUTION_PRESETS (e.g., 'fine', 'coarse', 'full').
        If None, uses OVERRIDE_SIZE_PRESET or defaults to 'aerosol'.
    r0_min_um : float or None
        Minimum radius [µm]. If None, uses preset or OVERRIDE.
    r0_max_um : float or None
        Maximum radius [µm]. If None, uses preset or OVERRIDE.
    n_nodes : int or None
        Number of nodes. If None, uses preset or OVERRIDE_N_NODES.

    Returns
    -------
    np.ndarray
        Radius grid [µm] in log-spaced order, shape (n_nodes,).
    """
    # Apply overrides if arguments not explicitly provided
    if size_preset is None:
        size_preset = OVERRIDE_SIZE_PRESET or 'aerosol'

    if size_preset not in SIZE_DISTRIBUTION_PRESETS:
        raise ValueError(
            f"Unknown size_preset '{size_preset}'. "
            f"Must be one of: {list(SIZE_DISTRIBUTION_PRESETS.keys())}"
        )

    preset = SIZE_DISTRIBUTION_PRESETS[size_preset]
    r0_min = r0_min_um if r0_min_um is not None else preset.r0_min_um
    r0_max = r0_max_um if r0_max_um is not None else preset.r0_max_um
    n = n_nodes if n_nodes is not None else (OVERRIDE_N_NODES or preset.n_nodes)

    return np.logspace(np.log10(r0_min), np.log10(r0_max), n)


# ============================================================================
# Output directory management
# ============================================================================

# Central output directory for all demo outputs
OUTPUT_BASE = Path(__file__).parent / 'demo_output'
OUTPUT_BASE.mkdir(exist_ok=True)


def get_output_path(demo_name: str, filename: str) -> Path:
    """
    Get the output path for a demo figure or file.

    Respects OVERRIDE_OUTPUT_DIR if set. Creates demo-specific subdirectory if it doesn't exist.

    Parameters
    ----------
    demo_name : str
        Name of the demo (e.g., 'single', 'lut', 'multidim_lut').
    filename : str
        Filename to save (e.g., 'spectrum.png').

    Returns
    -------
    Path
        Full path to output file.

    Examples
    --------
    >>> path = get_output_path('single', 'mec_spectrum.png')
    >>> path
    PosixPath('/...tests/demo_output/single/mec_spectrum.png')
    """
    base_dir = OVERRIDE_OUTPUT_DIR or OUTPUT_BASE
    demo_dir = Path(base_dir) / demo_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir / filename


# ============================================================================
# Plotting utilities
# ============================================================================

def make_r0_colormap(
    r0_min_um: float,
    r0_max_um: float,
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """
    Create a logarithmic colormap for radius visualization.

    Parameters
    ----------
    r0_min_um : float
        Minimum radius [µm].
    r0_max_um : float
        Maximum radius [µm].

    Returns
    -------
    cmap : mcolors.Colormap
        Colormap (from mieEODG.plotting if available, else viridis).
    norm : mcolors.LogNorm
        Logarithmic normalization for r0 range.
    """
    norm = mcolors.LogNorm(vmin=r0_min_um, vmax=r0_max_um)
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis')
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('viridis')
    return cmap, norm


# ============================================================================
# Predefined size distributions (common parameterizations)
# ============================================================================

LOGNORMAL_PRESETS = {
    'fine': {'s': 1.6, 'description': 'Fine mode (typical accumulation)'},
    'coarse': {'s': 2.0, 'description': 'Coarse mode (typical dust)'},
    'narrow': {'s': 1.3, 'description': 'Narrow distribution'},
    'broad': {'s': 2.5, 'description': 'Broad distribution'},
}

GAUSSIAN_PRESETS = {
    'narrow': {'sigma_um': 0.2, 'description': 'Narrow Gaussian'},
    'moderate': {'sigma_um': 0.5, 'description': 'Moderate spread'},
    'broad': {'sigma_um': 1.0, 'description': 'Broad spread'},
}


# ============================================================================
# Demo descriptions (for README or documentation)
# ============================================================================

DEMO_MANIFEST = {
    'single': {
        'title': 'Single Particle Mie Calculation',
        'description': 'Computes MEC, MSC, MAC for a single particle radius.',
        'outputs': ['mec_msc_mac_single.png'],
        'duration': '~5 sec',
    },
    'size_dist': {
        'title': 'Size Distribution Comparison',
        'description': 'Shows 3 distribution types (LogNormal, Gaussian, ModifiedGamma) and their effects on optical properties.',
        'outputs': ['lognormal_spectrum.png', 'gaussian_spectrum.png', 'modifiedgamma_spectrum.png'],
        'duration': '~10 sec',
    },
    'lut': {
        'title': 'Look-Up Table (LUT) Generation & Validation',
        'description': 'Builds a LUT across a r0 grid, validates PCHIP interpolation accuracy, and performs multi-species survey.',
        'outputs': ['interpolation_validation.png', 'multispecies_survey.png'],
        'duration': '~30 sec',
    },
    'multidim_lut': {
        'title': 'Multi-Dimensional LUT',
        'description': 'Constructs and visualizes N-D LUTs with sweeps across r0, composition, and temperature.',
        'outputs': ['multidim_lut_sweep.png'],
        'duration': '~60 sec',
    },
    'prata': {
        'title': 'Prata et al. (2019) Volcanic Ash RI',
        'description': 'Generates synthetic refractive indices from bulk oxide composition using linear spectral regression.',
        'outputs': ['prata_composition_series.png', 'prata_andesite_spectra.png'],
        'duration': '~15 sec',
    },
}
