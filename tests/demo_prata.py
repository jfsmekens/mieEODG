"""
Demonstration: Prata et al. (2019) volcanic ash RI parameterization.

Generates complex refractive index from bulk oxide composition using the
linear spectral regression model of Prata et al. (2019, JGR Atmospheres).

Part 1 — Simple SiO₂-based parameterization
Part 2 — Full composition with NBO/T calculator
Part 3 — Comparison across composition range

Edit the USER CONFIGURATION section below to customise the run.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

from mieEODG import (
    ash_refrindex, write_ri,
    BulkComposition, nbot_from_composition,
    AerosolXSC, LogNormalDist,
)
from mieEODG.plotting import _add_top_axis

from demo_config import make_wavenumber_grid, get_output_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Spectral grid: thermal IR (8–15 µm) with 1 cm⁻¹ resolution
wavenumber = make_wavenumber_grid('thermal_ir', wn_step=1.0)

# ---------------------------------------------------------------------------
# Compositions for Part 2 & 3
# ---------------------------------------------------------------------------
# Basalt-to-rhyolite compositional series (Reed et al. 2018 training data)

COMPOSITIONS = {
    'Basalt\n(SiO₂=48%)': BulkComposition(
        sio2=48.0, tio2=1.5, al2o3=15.0, fe2o3=10.0, mno=0.2,
        mgo=10.0, cao=10.0, na2o=2.5, k2o=0.5, p2o5=0.2
    ),
    'Andesite\n(SiO₂=62%)': BulkComposition(
        sio2=62.0, tio2=0.8, al2o3=16.0, fe2o3=6.0, mno=0.1,
        mgo=3.5, cao=6.0, na2o=3.0, k2o=1.5, p2o5=0.1
    ),
    'Dacite\n(SiO₂=68%)': BulkComposition(
        sio2=68.0, tio2=0.5, al2o3=16.0, fe2o3=3.5, mno=0.05,
        mgo=1.5, cao=3.0, na2o=3.5, k2o=2.5, p2o5=0.1
    ),
    'Rhyolite\n(SiO₂=75%)': BulkComposition(
        sio2=75.0, tio2=0.2, al2o3=13.5, fe2o3=1.5, mno=0.02,
        mgo=0.5, cao=1.5, na2o=3.5, k2o=4.0, p2o5=0.05
    ),
}

ROCK_TYPES = {
    'Basalt': 'basalt',
    'Andesite': 'andesite',
    'Dacite': 'dacite',
    'Rhyolite': 'rhyolite',
}

# Output is managed by demo_config.get_output_path()

# ===========================================================================
# PART 1 — Simple SiO₂-based parameterization
# ===========================================================================

print("\n" + "=" * 70)
print("PART 1: Simple SiO₂-based parameterization (no composition needed)")
print("=" * 70)

sio2_val = 62.0
ri_sio2 = ash_refrindex(wavenumber, sio2=sio2_val)
print(f"Generated RI for SiO₂ = {sio2_val} wt%")
print(f"  n range: {ri_sio2.n.min():.4f} – {ri_sio2.n.max():.4f}")
print(f"  k range: {ri_sio2.k.min():.6f} – {ri_sio2.k.max():.6f}")

# ===========================================================================
# PART 2 — Full composition with NBO/T calculator
# ===========================================================================

print("\n" + "=" * 70)
print("PART 2: Full composition with NBO/T-based parameterization")
print("=" * 70)

results_nbot = {}
for comp_name, comp_obj in COMPOSITIONS.items():
    rock_type = list(ROCK_TYPES.values())[
        list(ROCK_TYPES.keys()).index(comp_name.split('\n')[0])
    ]
    nbot = nbot_from_composition(comp_obj, rock_type=rock_type)
    ri = ash_refrindex(wavenumber, comp=comp_obj, rock_type=rock_type)
    results_nbot[comp_name] = {
        'comp': comp_obj,
        'nbot': nbot,
        'ri': ri,
    }
    print(f"{comp_name}: NBO/T = {nbot:.4f}")

# ===========================================================================
# PART 3 — Side-by-side comparison figure
# ===========================================================================

print("\n" + "=" * 70)
print("PART 3: Generating comparison figure")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

colors = ['#8B4513', '#CD853F', '#DEB887', '#F5DEB3']  # Brown to light tan

for idx, ((comp_name, data), color) in enumerate(
    zip(results_nbot.items(), colors)
):
    ax = axes[idx]
    ri = data['ri']
    nbot = data['nbot']

    # Plot n and k
    ax_n = ax
    ax_k = ax.twinx()

    line_n = ax_n.plot(wavenumber, ri.n, color=color, lw=1.5, label='n')
    line_k = ax_k.plot(wavenumber, ri.k, color=color, lw=1.5, alpha=0.5,
                       linestyle='--', label='k')

    ax_n.set_xlim(WAVN_MAX, WAVN_MIN)
    ax_n.set_ylim(0.8, 2.8)
    ax_k.set_ylim(0, 1.5)

    ax_n.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax_n.set_ylabel('n (real part)', fontsize=9, color=color)
    ax_k.set_ylabel('k (extinction)', fontsize=9, color=color, alpha=0.6)

    ax_n.tick_params(axis='y', labelcolor=color)
    ax_k.tick_params(axis='y', labelcolor=color)

    ax_n.set_title(f'{comp_name}  NBO/T = {nbot:.3f}', fontsize=10)
    ax_n.grid(True, alpha=0.3, lw=0.5)

    # Add wavelength axis
    _add_top_axis(ax_n, units='um')

fig.suptitle(
    'Prata et al. (2019) volcanic ash RI parameterization — composition series',
    fontsize=12,
)

out_fig1 = get_output_path('prata', 'prata_composition_series.png')
fig.savefig(out_fig1, dpi=150, bbox_inches='tight')
print(f"✓ Figure saved → {out_fig1}")

# ===========================================================================
# BONUS: Compute optical properties for one composition
# ===========================================================================

print("\n" + "=" * 70)
print("BONUS: Compute optical properties (MEC/MSC/MAC) for andesite")
print("=" * 70)

comp_andesite = COMPOSITIONS['Andesite\n(SiO₂=62%)']
ri_andesite = ash_refrindex(wavenumber, comp=comp_andesite, rock_type='andesite')

# Use with AerosolXSC
dist = LogNormalDist.from_um(1.0, s=1.5, n=1.0)
xsc = AerosolXSC(ri=ri_andesite, dist=dist, wn=wavenumber)
xsc.compute(progress=False)

print(f"MEC peak: {xsc.MEC.max():.2f} cm²/g")
print(f"MSC peak: {xsc.MSC.max():.2f} cm²/g")
print(f"MAC peak: {xsc.MAC.max():.2f} cm²/g")

# Plot
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(wavenumber, xsc.MEC, label='MEC', lw=1.2, alpha=0.9)
ax.plot(wavenumber, xsc.MSC, label='MSC', lw=1.2, alpha=0.7)
ax.plot(wavenumber, xsc.MAC, label='MAC', lw=1.2, alpha=0.4)

ax.set_xlim(WAVN_MAX, WAVN_MIN)
ax.set_ylim(bottom=0)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
ax.set_ylabel(r'Mass coefficient (cm$^2$ g$^{-1}$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, lw=0.5)
ax.set_title('Andesite (SiO₂=62%) — r₀ = 1.0 µm, s = 1.5', fontsize=12)

_add_top_axis(ax, units='um')

out_fig2 = get_output_path('prata', 'prata_andesite_spectra.png')
fig.savefig(out_fig2, dpi=150, bbox_inches='tight')
print(f"✓ Figure saved → {out_fig2}")

# Display figures
plt.show()

# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 70)
print("All demos completed successfully!")
print(f"Output: {get_output_path('prata', '').parent}")
print("=" * 70)
