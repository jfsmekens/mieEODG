"""
Demonstration: Single particle size — spectral mass coefficients.

Computes MEC, MSC, and MAC for a single particle radius using AerosolXSC.
Shows how optical properties vary with wavelength for a fixed particle size.

A narrow LogNormal distribution (s ≈ 1.01) approximates a monodisperse
ensemble of particles at the specified radius.

Edit the USER CONFIGURATION section below to customise the run.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mieEODG import (
    AerosolXSC, ARIADatabase,
    LogNormalDist, ARIA_DEFAULTS,
    get_cmap,
)
from mieEODG.plotting import _add_top_axis

from demo_config import make_wavenumber_grid, get_output_path

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Refractive index
RI_PATH = ARIA_DEFAULTS['volcanic_ash']

# Spectral grid (thermal IR, 1 cm⁻¹ resolution)
wavenumber = make_wavenumber_grid('thermal_ir', wn_step=1.0)

# Particle size (µm) — monodisperse approximation via narrow LogNormal
R0_UM = 1.0

# Backend
BACKEND = 'python'

# ===========================================================================
# Computation
# ===========================================================================

print(f"Single particle size demonstration")
print("=" * 60)
print(f"Particle radius: {R0_UM} µm")
print(f"Spectral range: {wavenumber[-1]:.0f}–{wavenumber[0]:.0f} cm⁻¹ ({len(wavenumber)} points)")

# Load refractive index
db = ARIADatabase()
ri = db.read(RI_PATH, wavenumber=wavenumber, out_of_range='clip')
print(f"Species: {ri.substance} ({ri.category})")

# Create monodisperse approximation (very narrow distribution)
dist = LogNormalDist.from_um(R0_UM, s=1.01, n=1.0)
print(f"Distribution: LogNormal (s=1.01 ≈ monodisperse)")

# Compute optical properties
xsc = AerosolXSC(ri=ri, dist=dist, wn=wavenumber, backend=BACKEND)
xsc.compute(progress=False)

print(f"Density: {xsc.density:.4f} g/cm³")
print(f"Asymmetry parameter g: {xsc.g}")
print()

# ===========================================================================
# Figure (single panel)
# ===========================================================================

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

# Get color from category-specific colormap
cmap = get_cmap(ri.category or '')
color = cmap(0.5)  # mid-range value for visual balance

# Plot MEC, MSC, MAC with varying alpha for visual hierarchy
alphas = [1.0, 0.7, 0.4]  # MEC brightest, MAC faintest
ax.plot(wavenumber, xsc.MEC, color=color, label='MEC', lw=1.2, alpha=alphas[0])
ax.plot(wavenumber, xsc.MSC, color=color, label='MSC', lw=1.2, alpha=alphas[1])
ax.plot(wavenumber, xsc.MAC, color=color, label='MAC', lw=1.2, alpha=alphas[2])

ax.set_xlim(wavenumber[0], wavenumber[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
ax.set_ylabel(r'Mass coefficient (cm$^2$ g$^{-1}$)', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, lw=0.5)

# Add wavelength axis on top
_add_top_axis(ax, units='um')

substance_str = ri.substance or ri.category
fig.suptitle(
    f'{substance_str} — Single particle size r₀ = {R0_UM} µm',
    fontsize=12,
)

out_file = get_output_path('single', 'mec_msc_mac.png')
fig.savefig(out_file, dpi=150, bbox_inches='tight')
print(f"Figure saved → {out_file}")
plt.show()
