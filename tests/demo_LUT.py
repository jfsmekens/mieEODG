"""
Demonstration: AerosolLUT generation and PCHIP interpolation accuracy.

Part 1 — Interpolation validation
    Build a coarse LUT (N_LUT nodes) for a single species, compute ground-truth
    AerosolXSC results at N_DENSE intermediate r0 values, and compare.
    Output: MEC/MSC/MAC spectra (LUT nodes + interpolation vs ground truth)
            + relative error heatmaps.

Part 2 — Multi-species MEC survey
    Build (or reload from cache) one LUT per species and plot MEC across a dense
    r0 sweep for each, coloured by r0.

Edit the USER CONFIGURATION section below to customise the run.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from tqdm.auto import tqdm

from mieEODG import AerosolXSC, AerosolLUT, ARIADatabase, LogNormalDist, get_cmap, ARIA_DEFAULTS
from mieEODG.plotting import _add_top_axis, plot_lut

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# ---------------------------------------------------------------------------
# Spectral grid
# ---------------------------------------------------------------------------
WAVN_MIN  = 650.0     # cm⁻¹
WAVN_MAX  = 1350.0    # cm⁻¹
WAVN_STEP = 1.0       # cm⁻¹  (decrease for finer resolution, costs more memory)

# ---------------------------------------------------------------------------
# Size grid
# ---------------------------------------------------------------------------
S         = 1.5       # log-normal spread parameter
R0_MIN_UM = 0.1       # µm — lower bound of r0 sweep
R0_MAX_UM = 10.0      # µm — upper bound of r0 sweep
N_LUT     = 10        # number of LUT nodes (log-spaced over [R0_MIN_UM, R0_MAX_UM])
N_DENSE   = 50        # number of ground-truth evaluations for Part 1 validation

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
# 'python'  — pure Python/NumPy implementation (always available)
# 'fortran' — compiled Fortran extension (faster; requires build)
BACKEND = 'fortran'

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
# Set RECOMPUTE = True to force a fresh computation and overwrite cached LUTs.
# Useful when you suspect a cached LUT is corrupt or was built with bad inputs.
RECOMPUTE = False

# ---------------------------------------------------------------------------
# Part 1 — species used for the interpolation validation test
# ---------------------------------------------------------------------------
PART1_SPECIES = 'volcanic_ash'   # any key from ARIA_DEFAULTS

# ---------------------------------------------------------------------------
# Part 2 — species to include in the multi-species MEC figure
# Each entry: ARIA_DEFAULTS key  →  display label
# The figure layout adapts automatically (ceil(N/3) rows × 3 columns).
# ---------------------------------------------------------------------------
PART2_SPECIES = {
    'volcanic_ash':        'Volcanic Ash (Etna)',
    'acids/sulphuric':      r'H$_2$SO$_4$ (72 wt%, 298 K)',
    'acids/nitric':        r'HNO$_3$ (50 wt%, 293 K)',
    'sand_and_dust':       'Mineral Dust (Algeria)',
    'water_and_ice/water': 'Liquid Water (298 K)',
    'water_and_ice/ice':   'Ice (266 K)',
}

# ===========================================================================
# DERIVED QUANTITIES  (do not edit below unless you know what you're doing)
# ===========================================================================

wavenumber = np.arange(WAVN_MIN, WAVN_MAX + WAVN_STEP / 2, WAVN_STEP)

R0_LUT_UM = np.logspace(np.log10(R0_MIN_UM), np.log10(R0_MAX_UM), N_LUT)

# Dense ground-truth grid: offset by half a LUT interval to avoid node coincidence
_log_step_half = (np.log10(R0_MAX_UM) - np.log10(R0_MIN_UM)) / (2 * (N_LUT - 1))
R0_DENSE_UM = np.logspace(
    np.log10(R0_MIN_UM) + _log_step_half,
    np.log10(R0_MAX_UM) - _log_step_half,
    N_DENSE,
)

# Dense plot grid for Part 2 (pure LUT queries — cheap)
R0_PLOT_UM = np.logspace(np.log10(R0_MIN_UM), np.log10(R0_MAX_UM), 200)

NORM_LUT   = mcolors.LogNorm(vmin=R0_LUT_UM[0],   vmax=R0_LUT_UM[-1])
NORM_DENSE = mcolors.LogNorm(vmin=R0_DENSE_UM[0], vmax=R0_DENSE_UM[-1])
NORM_PLOT  = mcolors.LogNorm(vmin=R0_MIN_UM,       vmax=R0_MAX_UM)

OUTPUT_DIR = Path(__file__).parent

db = ARIADatabase()

# ===========================================================================
# PART 1 — Interpolation validation
# ===========================================================================

print(f'=== Part 1: interpolation validation  [{PART1_SPECIES}] ===')

ri = db.read(ARIA_DEFAULTS[PART1_SPECIES], wavenumber=wavenumber, out_of_range='clip')

# --- Build LUT (cached) ------------------------------------------------------
lut = AerosolLUT.load_or_compute(
    ri, r0_um=R0_LUT_UM, s=S, wn=wavenumber,
    progress=True, backend=BACKEND, recompute=RECOMPUTE,
)
print(lut)

# --- Ground truth ------------------------------------------------------------
print('\nComputing ground truth …')
MEC_true = np.empty((N_DENSE, len(wavenumber)))
MSC_true = np.empty_like(MEC_true)
MAC_true = np.empty_like(MEC_true)

for i, r0_um in enumerate(tqdm(R0_DENSE_UM, desc='Ground truth', unit='r0')):
    xsc = AerosolXSC(ri=ri, dist=LogNormalDist.from_um(r0_um, S, n=1.0), wn=wavenumber, backend=BACKEND)
    xsc.compute(progress=False)
    MEC_true[i] = xsc.result.MEC
    MSC_true[i] = xsc.result.MSC
    MAC_true[i] = xsc.result.MAC

# --- LUT queries at the same dense grid --------------------------------------
print('\nQuerying LUT …')
MEC_lut = np.empty_like(MEC_true)
MSC_lut = np.empty_like(MEC_true)
MAC_lut = np.empty_like(MEC_true)

for i, r0_um in enumerate(R0_DENSE_UM):
    fwd = lut.forward(r0_um)
    MEC_lut[i] = fwd['MEC']
    MSC_lut[i] = fwd['MSC']
    MAC_lut[i] = fwd['MAC']

# --- Relative error ----------------------------------------------------------
def rel_err(approx: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(truth) > 1e-12, np.abs(truth), np.nan)
    return 100.0 * (approx - truth) / denom

MEC_err = rel_err(MEC_lut, MEC_true)
MSC_err = rel_err(MSC_lut, MSC_true)
MAC_err = rel_err(MAC_lut, MAC_true)

print()
for name, err in [('MEC', MEC_err), ('MSC', MSC_err), ('MAC', MAC_err)]:
    finite = err[np.isfinite(err)]
    print(f'  {name}  max |err| = {np.nanmax(np.abs(err)):.2f}%   '
          f'mean |err| = {np.nanmean(np.abs(finite)):.3f}%')

# --- Figure ------------------------------------------------------------------
cmap_p1      = get_cmap(ri.category)
colors_lut   = [cmap_p1(NORM_LUT(r))   for r in R0_LUT_UM]
colors_dense = [cmap_p1(NORM_DENSE(r)) for r in R0_DENSE_UM]

PANELS_P1 = [
    ('MEC', MEC_true, MEC_lut, MEC_err, r'MEC  (cm$^2$ g$^{-1}$)'),
    ('MSC', MSC_true, MSC_lut, MSC_err, r'MSC  (cm$^2$ g$^{-1}$)'),
    ('MAC', MAC_true, MAC_lut, MAC_err, r'MAC  (cm$^2$ g$^{-1}$)'),
]

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for col in range(3):
    axes1[0, col].sharex(axes1[0, 0])

_dark       = cmap_p1(NORM_LUT(R0_LUT_UM[-1]))
_proxy_node  = plt.Line2D([0], [0], color=_dark, lw=1.4,       label='LUT nodes')
_proxy_interp = plt.Line2D([0], [0], color=_dark, lw=0.8,       label='LUT interp.')
_proxy_truth  = plt.Line2D([0], [0], color=_dark, lw=0.8, ls='--', label='Ground truth')

for col, (key, true_data, lut_data, err_data, ylabel) in enumerate(PANELS_P1):
    ax = axes1[0, col]

    for spec, color in zip([lut.forward(r)[key] for r in R0_LUT_UM], colors_lut):
        ax.plot(wavenumber, spec, color=color, lw=1.4, alpha=0.95)
    for i, color in enumerate(colors_dense):
        ax.plot(wavenumber, lut_data[i],  color=color, lw=0.7, alpha=0.7)
        ax.plot(wavenumber, true_data[i], color=color, lw=0.7, alpha=0.7, ls='--')

    ax.set_xlim(WAVN_MAX, WAVN_MIN)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(key, fontsize=10)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    _add_top_axis(ax)
    if col == 0:
        ax.legend(handles=[_proxy_node, _proxy_interp, _proxy_truth],
                  fontsize=8, loc='upper right')

    ax2 = axes1[1, col]
    err_max = np.nanpercentile(np.abs(err_data), 99)
    im = ax2.imshow(
        err_data,
        aspect='auto', origin='lower',
        extent=[WAVN_MIN, WAVN_MAX,
                np.log10(R0_DENSE_UM[0]), np.log10(R0_DENSE_UM[-1])],
        cmap='RdBu_r', vmin=-err_max, vmax=err_max,
    )
    ax2.invert_xaxis()
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax2.set_ylabel('log$_{10}$($r_0$ / µm)', fontsize=9)
    ax2.set_title(f'{key} relative error (%)', fontsize=10)
    fig1.colorbar(im, ax=ax2, pad=0.01, fraction=0.04, label='%')

sm1 = mcm.ScalarMappable(cmap=cmap_p1, norm=NORM_LUT)
sm1.set_array([])
cbar1 = fig1.colorbar(sm1, ax=axes1[0, -1], pad=0.015, fraction=0.04, aspect=20)
cbar1.set_label('$r_0$ (µm)', fontsize=9)
cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2g}'))

substance_str = ri.substance or ri.category
fig1.suptitle(
    f'AerosolLUT validation — {substance_str}  '
    f'(s={S},  r0={R0_MIN_UM}–{R0_MAX_UM} µm,  '
    f'N_LUT={N_LUT},  {WAVN_MIN:.0f}–{WAVN_MAX:.0f} cm⁻¹ @ {WAVN_STEP} cm⁻¹,  '
    f'backend={BACKEND})',
    fontsize=10,
)

out_p1 = OUTPUT_DIR / f'demo_LUT_p1_{PART1_SPECIES.replace("/", "_")}.png'
fig1.savefig(out_p1, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {out_p1}')

# ===========================================================================
# PART 2 — Multi-species MEC survey
# ===========================================================================

print(f'\n=== Part 2: multi-species MEC  [{len(PART2_SPECIES)} species] ===')

luts_p2: dict[str, AerosolLUT] = {}
for cat, label in PART2_SPECIES.items():
    ri_p2      = db.read(ARIA_DEFAULTS[cat], wavenumber=wavenumber, out_of_range='clip')
    luts_p2[cat] = AerosolLUT.load_or_compute(
        ri_p2, r0_um=R0_LUT_UM, s=S, wn=wavenumber,
        progress=True, backend=BACKEND, recompute=RECOMPUTE,
    )
    print(f'  {label:40s}  ready')

suptitle_p2 = (
    f'Mass extinction coefficient — LUT interpolation  '
    f'(s={S},  r0={R0_MIN_UM}–{R0_MAX_UM} µm,  '
    f'{WAVN_MIN:.0f}–{WAVN_MAX:.0f} cm⁻¹ @ {WAVN_STEP} cm⁻¹,  backend={BACKEND})'
)

# Remap from PART2_SPECIES display labels to LUT objects for plot_lut
luts_display = {label: luts_p2[cat] for cat, label in PART2_SPECIES.items()}

out_p2 = OUTPUT_DIR / 'demo_LUT_p2.png'
plot_lut(
    luts_display,
    quantity='MEC',
    r0_um=R0_PLOT_UM,
    suptitle=suptitle_p2,
    show=False,
    save_path=out_p2,
)
print(f'Figure saved → {out_p2}')

plt.show()
