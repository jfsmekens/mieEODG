"""
mieEODG
=======
Mie scattering routines for spherical particles, ported from the EODG IDL
package (https://eodg.atm.ox.ac.uk/MIE/).

Public API
----------
AerosolXSC          : Primary object — Mie cross-sections and mass coefficients
                      for a given refractive index + size distribution pair.
mieSingle           : Mie efficiencies for a single particle (array of sizes).
mieSizeDist         : Bulk optical properties integrated over a size distribution.
mieDerivs           : Analytical derivatives for a single particle.
mieSterivsLn        : Analytical derivatives integrated over a log-normal
                      distribution.
quadrature          : Quadrature abscissae and weights on [-1, 1].
shiftQuadrature     : Shift quadrature interval from [-1, 1] to [a, b].
legendreExpand      : Expand a function as a Legendre series.
legendreReconstruct : Reconstruct a function from Legendre coefficients.
genSphFuncs         : Generalised spherical functions for phase matrix expansion.
scatPhaseMatrixExp  : Expand phase matrix in generalised spherical functions.
LogNormalDist       : Log-normal particle size distribution class.
ModifiedGammaDist   : Modified-gamma particle size distribution class.
createDist          : Factory for size distribution objects.
ARIADatabase        : Interface to the bundled ARIA refractive index database.
RefrIndex           : Refractive index data container.
readRI              : Read a single .ri file.
"""

from .quadrature import quadrature, shiftQuadrature
from .size_dist import LogNormalDist, ModifiedGammaDist, GaussianDist, createDist
from .refindex import ARIADatabase, RefrIndex, readRI, ash_refrindex, write_ri
from .composition import BulkComposition, nbot_from_composition
from .aerosol import AerosolXSC
from .lut import AerosolLUT
from .units import (
    wn_from_um, wn_from_nm,
    wl_um_from_wn, wl_nm_from_wn,
    r_cm_from_um, r_um_from_cm,
    wn_grid_from_um, wn_grid_from_nm,
)
from .plotting import plot_spectrum, plot_lut_sweep, plot_lut, plot_lut_slice, plot_lut_heatmap, plot_size_dist, plot_phase_function, plot_phase_function_heatmap
from .rfm import write_rfm_xsc
from .constants import (
    DENSITY_WATER, DENSITY_ICE, DENSITY_ASH, DENSITY_DUST,
    DENSITY_SEA_SALT, DENSITY_BLACK_CARBON, DENSITY_CALCITE, DENSITY_QUARTZ,
    DENSITY_TABLE, DISPLAY_NAMES,
    h2so4_density, hno3_density, ice_density, sio2_density, aria_density,
    SUBSTANCE_DENSITY, ARIA_DEFAULTS,
    plot_cmaps, plot_colors, get_cmap, get_color,
)

# Modules implemented progressively — imported here once available
try:
    from .mie_single import mieSingle
    from .mie_size_dist import mieSizeDist
    from .mie_derivs import mieDerivs
    from .mie_derivs_ln import mieSterivsLn
    from .legendre import legendreExpand, legendreReconstruct
    from .sph_funcs import genSphFuncs, scatPhaseMatrixExp
except ImportError:
    pass

__all__ = [
    # Primary objects
    "AerosolXSC",
    "AerosolLUT",
    # Low-level Mie routines
    "mieSingle",
    "mieSizeDist",
    "mieDerivs",
    "mieSterivsLn",
    # Quadrature / Legendre / phase matrix
    "quadrature",
    "shiftQuadrature",
    "legendreExpand",
    "legendreReconstruct",
    "genSphFuncs",
    "scatPhaseMatrixExp",
    # Size distributions
    "LogNormalDist",
    "ModifiedGammaDist",
    "createDist",
    # Refractive index
    "ARIADatabase",
    "RefrIndex",
    "readRI",
    "ash_refrindex",
    "write_ri",
    # Composition
    "BulkComposition",
    "nbot_from_composition",
    # Unit conversions
    "wn_from_um",
    "wn_from_nm",
    "wl_um_from_wn",
    "wl_nm_from_wn",
    "r_cm_from_um",
    "r_um_from_cm",
    "wn_grid_from_um",
    "wn_grid_from_nm",
    # Plotting
    "plot_spectrum",
    "plot_lut_sweep",
    "plot_lut",
    "plot_lut_slice",
    "plot_lut_heatmap",
    "plot_size_dist",
    "plot_phase_function",
    "plot_phase_function_heatmap",
    # RFM export
    "write_rfm_xsc",
    # Constants
    "DENSITY_WATER",
    "DENSITY_ICE",
    "DENSITY_ASH",
    "DENSITY_DUST",
    "DENSITY_SEA_SALT",
    "DENSITY_BLACK_CARBON",
    "DENSITY_CALCITE",
    "DENSITY_QUARTZ",
    "DENSITY_TABLE",
    "h2so4_density",
]
