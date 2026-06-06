"""
Demonstration: MultiDimLUT generation and N-dimensional interpolation.

This script shows how to:
1. Define a multi-dimensional sweep (r0, composition, temperature)
2. Compute a LUT from ARIA database with single-source consistency
3. Interpolate and plot results with variability across sweep dimensions

The plot shows the full range of one dimension while fixing others at midpoint,
with multiple subplots for each sweep axis.

Edit the USER CONFIGURATION section below to customise the run.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mieEODG.lut_multi import LUTAxis, MultiDimLUT
from mieEODG.plotting import plot_lut_slice

from demo_config import make_wavenumber_grid, get_output_path

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Spectral grid (thermal IR, configurable resolution)
WAVN_STEP = 1.0       # cm⁻¹  (decrease for finer resolution, costs more memory)

# ---------------------------------------------------------------------------
# Species and source
# ---------------------------------------------------------------------------
# Start with H₂SO₄ (Myhre 2003) — most complete dataset with 2-D coverage
SUBSTANCE = 'Sulphuric acid'
SOURCE    = 'Myhre 2003'

# ---------------------------------------------------------------------------
# Sweep dimensions (choose 2-D or 3-D)
# ---------------------------------------------------------------------------
# 2-D: r0 × composition (simplest, fast)
# 3-D: r0 × composition × temperature (full coverage, slower)
SWEEP_2D = False

# If 3-D, number of temperature points (reduces to speed up)
# Full range: 14 points (203–298 K); subset: 6 points
N_TEMPERATURE = 10

# ---------------------------------------------------------------------------
# Size grid (r0)
# ---------------------------------------------------------------------------
R0_MIN_UM = 0.1       # µm
R0_MAX_UM = 2.0       # µm
N_R0      = 12        # LUT nodes (log-spaced)

# ---------------------------------------------------------------------------
# Composition (wt%) — subset of available range [38–72] for Myhre 2003
# Valid range: [38, 48, 58, 65, 72] wt%
# ---------------------------------------------------------------------------
COMPOSITION_PTS = np.array([38, 48, 58, 65, 72])

# ---------------------------------------------------------------------------
# Quantity to plot
# ---------------------------------------------------------------------------
QUANTITY = 'MEC'  # 'MEC', 'MSC', or 'MAC'

# ---------------------------------------------------------------------------
# Backend for Mie computations
# ---------------------------------------------------------------------------
# 'python'  — pure Python/NumPy (always available, slower)
# 'fortran' — compiled Fortran extension (~40× faster, requires build)
BACKEND = 'fortran'

# Cache and recompute
RECOMPUTE = False

# ===========================================================================
# DERIVED QUANTITIES  (do not edit below unless you know what you're doing)
# ===========================================================================

print(f'=== MultiDimLUT Demo: {SUBSTANCE} ({SOURCE}) ===\n')

wavenumber = make_wavenumber_grid('thermal_ir', wn_step=WAVN_STEP)
r0_lut_um = np.logspace(np.log10(R0_MIN_UM), np.log10(R0_MAX_UM), N_R0)

print(f'Spectral grid: {wavenumber[-1]:.0f}–{wavenumber[0]:.0f} cm⁻¹ @ {WAVN_STEP} cm⁻¹ ({len(wavenumber)} points)')
print(f'Radius grid: {R0_MIN_UM}–{R0_MAX_UM} µm, {N_R0} nodes (log-spaced)')
print(f'Composition: {COMPOSITION_PTS} wt%')

# ===========================================================================
# BUILD LUT
# ===========================================================================

# Define sweep axes
axes = [
    LUTAxis('r0', r0_lut_um, 'µm', log_spacing=True, fixed_for_ri_load=True),
    LUTAxis('composition', COMPOSITION_PTS, 'wt%', fixed_for_ri_load=False),
]

if not SWEEP_2D:
    # 3-D sweep: add temperature axis
    temperature_pts = np.linspace(203, 298, N_TEMPERATURE)
    axes.append(
        LUTAxis('temperature_K', temperature_pts, 'K', fixed_for_ri_load=False)
    )
    lut_type = f'3-D (r0 × composition × temperature, {N_TEMPERATURE} T pts)'
else:
    lut_type = '2-D (r0 × composition)'

print(f'\nLUT type: {lut_type}')

# Create LUT
lut = MultiDimLUT.from_aria_database(
    substance=SUBSTANCE,
    source=SOURCE,
    axes=axes,
    wavenumber=wavenumber,
    quantity=QUANTITY,
)

print(f'LUT shape: {lut.shape}')
print(f'Data size: {lut.shape[0] * lut.shape[1] * (lut.shape[2] if not SWEEP_2D else 1) * len(wavenumber) / 1e6:.1f} MB')

# Compute
print(f'\nComputing {QUANTITY} for all grid points…')
print(f'Backend: {BACKEND}')
lut.compute(progress=True, backend=BACKEND)

print(f'Data array populated: {lut.data.shape}')
valid_pts = np.sum(~np.isnan(lut.data))
total_pts = np.prod(lut.data.shape)
print(f'Valid data points: {valid_pts}/{total_pts} ({100*valid_pts/total_pts:.1f}%)')

# ===========================================================================
# EXTRACT VARIABILITY ACROSS EACH DIMENSION
# ===========================================================================

# Colormap for the substance
cmap = get_cmap(lut.ri_constraint.substance)

# Midpoint values for fixed dimensions
r0_mid = r0_lut_um[N_R0 // 2]
comp_mid_idx = len(COMPOSITION_PTS) // 2
comp_mid = COMPOSITION_PTS[comp_mid_idx]

if not SWEEP_2D:
    temp_mid = temperature_pts[N_TEMPERATURE // 2]
else:
    temp_mid = None

print(f'\nInterpolation midpoints:')
print(f'  r0 mid: {r0_mid:.3f} µm')
print(f'  composition mid: {comp_mid:.1f} wt%')
if temp_mid is not None:
    print(f'  temperature mid: {temp_mid:.0f} K')

# ===========================================================================
# PLOTTING — Using plot_lut_slice() for composable, reusable panels
# ===========================================================================

print(f'\nPlotting…')

# Set up multi-panel layout
n_subplots = 2 if SWEEP_2D else 3
fig, axes_plot = plt.subplots(1, n_subplots, figsize=(16, 5), constrained_layout=True)

# Ensure axes_plot is always a list (even for single subplot)
if n_subplots == 1:
    axes_plot = [axes_plot]

wn_min, wn_max = wavenumber.min(), wavenumber.max()

# Panel 1: Sweep r0, fix composition & temperature
plt.sca(axes_plot[0])
fixed_ax1 = {'composition': comp_mid}
if temp_mid is not None:
    fixed_ax1['temperature_K'] = temp_mid

plot_lut_slice(
    lut,
    sweep_axis='r0',
    fixed_axes=fixed_ax1,
    quantity=QUANTITY,
    n_points=50,
    fig=fig,
    show=False,
)

# Panel 2: Sweep composition, fix r0 & temperature
plt.sca(axes_plot[1])
fixed_ax2 = {'r0': r0_mid}
if temp_mid is not None:
    fixed_ax2['temperature_K'] = temp_mid

plot_lut_slice(
    lut,
    sweep_axis='composition',
    fixed_axes=fixed_ax2,
    quantity=QUANTITY,
    n_points=50,
    fig=fig,
    show=False,
)

# Panel 3: Sweep temperature (if 3-D)
if not SWEEP_2D:
    plt.sca(axes_plot[2])
    plot_lut_slice(
        lut,
        sweep_axis='temperature_K',
        fixed_axes={'r0': r0_mid, 'composition': comp_mid},
        quantity=QUANTITY,
        n_points=50,
        fig=fig,
        show=False,
    )

# Overall title
fig.suptitle(
    f'{SUBSTANCE} ({SOURCE})  —  {lut_type}\n'
    f'{QUANTITY}  variability across sweep dimensions  '
    f'({wn_max:.0f}–{wn_min:.0f} cm$^{{-1}}$)',
    fontsize=11,
)

out_fig = get_output_path('multidim_lut', f'{SUBSTANCE.replace(" ", "_")}_variability.png')
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Figure saved → {out_fig}')

plt.show()

# ===========================================================================
# SUMMARY STATISTICS
# ===========================================================================

print(f'\n=== Summary ===')
print(f'LUT shape: {lut.shape}')
print(f'Source: {lut.ri_constraint.source}')
print(f'Sweep axes: {lut.ri_constraint.sweep_axes}')
print(f'Quantity: {lut.quantity}')
print(f'\n{QUANTITY} statistics:')
valid_data = lut.data[~np.isnan(lut.data)]
if len(valid_data) > 0:
    print(f'  min: {np.min(valid_data):.6f} cm²/g')
    print(f'  max: {np.max(valid_data):.6f} cm²/g')
    print(f'  mean: {np.mean(valid_data):.6f} cm²/g')
    print(f'  std: {np.std(valid_data):.6f} cm²/g')

print(f'\nDone!')
