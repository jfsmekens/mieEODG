"""
Demonstration: Particle size distribution vs computed spectral coefficients.

Shows the relationship between particle size distribution and optical properties
for three distribution types (LogNormal, Gaussian, ModifiedGamma).

Layout: 3 rows × 2 columns
  Rows — distribution types (LogNormal, Gaussian, ModifiedGamma)
  Left column — size distribution PDF (dN/d ln r)
  Right column — spectral coefficients (MEC, MSC, MAC)

Colour convention: Each row uses a dedicated colour for consistency.

Edit the USER CONFIGURATION section below to customise the run.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mieEODG import (
    AerosolXSC, ARIADatabase,
    LogNormalDist, GaussianDist,
    plot_size_dist, ARIA_DEFAULTS,
)
from mieEODG.size_dist import ModifiedGammaDist
from mieEODG.plotting import _add_top_axis

from demo_config import make_wavenumber_grid, get_output_path

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Refractive index
RI_PATH = ARIA_DEFAULTS['volcanic_ash']

# Spectral grid (thermal IR, 1 cm⁻¹ resolution)
wavenumber = make_wavenumber_grid('thermal_ir', wn_step=1.0)

# Size distribution parameters
R0_UM  = 1.0          # characteristic radius (µm)
S_LOG  = 1.5          # log-normal spread
N      = 1.0          # number density (arbitrary units)

# Backend
BACKEND = 'python'

# ===========================================================================
# Configuration
# ===========================================================================

DIST_CONFIGS = [
    ('LogNormal', LogNormalDist.from_um(R0_UM, S_LOG, n=N)),
    ('Gaussian',  GaussianDist.from_um(R0_UM, R0_UM * np.log(S_LOG), n=N)),
    ('ModGamma',  ModifiedGammaDist.from_um(R0_UM * 0.7, 0.15, 0.1, 20.0, n=N)),
]

QUANTITIES = ['MEC', 'MSC', 'MAC']

# ===========================================================================
# Computation
# ===========================================================================

db = ARIADatabase()
ri = db.read(RI_PATH, wavenumber=wavenumber, out_of_range='clip')

print(f"Species: {ri.substance} ({ri.category})")
print(f"Spectral range: {wavenumber[-1]:.0f}–{wavenumber[0]:.0f} cm⁻¹ ({len(wavenumber)} points)")
print(f"Size distributions: {len(DIST_CONFIGS)} types")
print()

results = {}
for dist_name, dist in DIST_CONFIGS:
    print(f"Computing {dist_name}...")
    xsc = AerosolXSC(ri=ri, dist=dist, wn=wavenumber, backend=BACKEND)
    xsc.compute(progress=False)
    results[dist_name] = {
        'dist': dist,
        'MEC': xsc.MEC,
        'MSC': xsc.MSC,
        'MAC': xsc.MAC,
    }
    print(f"  ✓ Density: {xsc.density:.4f} g/cm³")

# ===========================================================================
# Determine global radius range for all distributions
# ===========================================================================

r_ranges = [data['dist'].bounds() for data in results.values()]
r_min_cm = min(r[0] for r in r_ranges)
r_max_cm = max(r[1] for r in r_ranges)
r_min_um = r_min_cm * 1e4
r_max_um = r_max_cm * 1e4

# ===========================================================================
# Figure (3 rows × 2 columns, with width_ratios=[1, 3])
# ===========================================================================

fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 10),
    gridspec_kw={'width_ratios': [1, 3]},
    constrained_layout=True,
)

prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for row, (dist_name, _) in enumerate(DIST_CONFIGS):
    color = prop_cycle[row % len(prop_cycle)]
    data = results[dist_name]

    # --- Left panel: size distribution ---
    ax_dist = axes[row, 0]

    # Compute PDF
    r_cm = data['dist']
    r_um = np.logspace(np.log10(r_min_um), np.log10(r_max_um), 300)
    r_cm_array = r_um * 1e-4
    n_r = r_cm.pdf(r_cm_array)
    dndlnr = n_r * r_cm_array
    dndlnr = dndlnr / dndlnr.max()  # normalize

    # Histogram style
    ax_dist.bar(r_um, dndlnr, width=np.diff(r_um).mean() * 0.8,
                color=color, label=dist_name, alpha=0.7, edgecolor='none')
    ax_dist.set_xscale('log')
    ax_dist.set_xlim(r_min_um, r_max_um)
    ax_dist.set_ylim(bottom=0, top=1.1)
    ax_dist.set_xlabel('Radius (µm)', fontsize=9)
    ax_dist.set_ylabel('dN/d ln r (norm.)', fontsize=9)
    ax_dist.legend(fontsize=9, loc='upper right')
    ax_dist.grid(True, alpha=0.3, lw=0.5)

    # --- Right panel: spectral coefficients (MEC, MSC, MAC stacked) ---
    ax_spec = axes[row, 1]

    for i, qty in enumerate(QUANTITIES):
        alpha = 1.0 - 0.2 * i  # Fade slightly for visual hierarchy
        ax_spec.plot(
            wavenumber,
            data[qty],
            color=color,
            label=qty,
            lw=0.9,
            alpha=alpha,
        )

    ax_spec.set_xlim(wavenumber[0], wavenumber[-1])
    ax_spec.set_ylim(bottom=0)
    ax_spec.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax_spec.set_ylabel(r'Coeff. (cm$^2$ g$^{-1}$)', fontsize=9)
    ax_spec.legend(fontsize=8, loc='upper right')
    ax_spec.grid(True, alpha=0.3, lw=0.5)

    # Add wavelength axis on top
    _add_top_axis(ax_spec, units='um')

    if row == 0:
        ax_spec.set_title('Spectral mass coefficients', fontsize=10)

substance_str = ri.substance or ri.category
fig.suptitle(
    f'{substance_str} — Size distributions and spectral coefficients\n'
    f'r₀ = {R0_UM} µm, s = {S_LOG}, λ range = {1e4/wavenumber[0]:.1f}–{1e4/wavenumber[-1]:.1f} µm',
    fontsize=12,
)

out_file = get_output_path('size_dist', 'distributions_and_spectra.png')
fig.savefig(out_file, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_file}")
plt.show()
