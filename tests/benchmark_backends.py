"""
Speed and accuracy benchmark: mieEODG Python backend vs Fortran backend.

Configuration
-------------
  H2SO4 (72 wt%, 298 K) — Lund Myhre et al. (2003)
  50 log-spaced median radii (0.1–10 µm),  s = 1.5
  Spectral grid: 650–1350 cm⁻¹ at 0.5 cm⁻¹  (1401 points)
  100 Gauss quadrature points per size

Backends
--------
1. mieEODG Python
   _mie_kernel_batch vectorises over all quadrature radii at each wavenumber.
   One NumPy call per (size, wavenumber).

2. mieEODG Fortran
   mie_batch() — direct Fortran 90 port of the Python kernel, compiled via
   f2py.  Takes the full radius array at once; no Python inner loops.

Output
------
  Timing + accuracy table printed to stdout.
  <script_dir>/benchmark_backends.png  — wall-clock bar chart + spectral overlay.
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_FILE = Path(__file__).parent / 'benchmark_backends.png'

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

ARIA_PATH = (
    'Acids/Sulphuric/70%_to_79%/'
    '72%,_298_K_(Lund_Myhre_et_al._2003)/'
    'H2SO4_72_298K_Myhre_2003.ri'
)
S       = 1.5
N_SIZES = 50
N_PTS   = 100
R0_UM   = np.logspace(np.log10(0.1), np.log10(10.0), N_SIZES)
R0_CM   = R0_UM * 1e-4

WAVN_MIN, WAVN_MAX, WAVN_STEP = 650.0, 1350.0, 0.5
wavenumber = np.arange(WAVN_MIN, WAVN_MAX + WAVN_STEP / 2, WAVN_STEP)
N_NU = len(wavenumber)

# ---------------------------------------------------------------------------
# Load refractive index (shared across both runs)
# ---------------------------------------------------------------------------
from mieEODG import ARIADatabase, LogNormalDist, h2so4_density, get_cmap

db      = ARIADatabase()
ri      = db.read(ARIA_PATH, wavenumber=wavenumber, out_of_range='clip')
Cm_in   = ri.Cm
DENSITY = h2so4_density(0.72)

print(f'Grid      : {N_NU} wavenumbers  ×  {N_SIZES} sizes')
print(f'Substance : {ri.substance}')
print(f'Density   : {DENSITY:.3f} g/cm³')
print(f'N_PTS     : {N_PTS}')
print()

# ---------------------------------------------------------------------------
# Backend 1: mieEODG Python
# ---------------------------------------------------------------------------

def run_python_batch() -> np.ndarray:
    from mieEODG import AerosolXSC
    MEC_all = np.empty((N_SIZES, N_NU))
    for i, r0_cm in tqdm(enumerate(R0_CM), total=N_SIZES,
                         desc='Python batch', unit='size'):
        xsc = AerosolXSC(
            ri      = ri,
            dist    = LogNormalDist(r0=r0_cm, s=S, n=1.0),
            wn      = wavenumber,
            density = DENSITY,
            n_pts   = N_PTS,
        )
        xsc.compute(progress=False)
        MEC_all[i] = xsc.result.MEC
    return MEC_all


# ---------------------------------------------------------------------------
# Backend 2: mieEODG Fortran
# ---------------------------------------------------------------------------

def run_fortran_batch() -> np.ndarray:
    from mieEODG.fortran._mie_kernel_f90 import mie_batch as _mie_batch_f90
    from mieEODG.quadrature import quadrature, shiftQuadrature

    MEC_all = np.empty((N_SIZES, N_NU))
    for i, r0_cm in tqdm(enumerate(R0_CM), total=N_SIZES,
                         desc='Fortran batch', unit='size'):
        dist = LogNormalDist(r0=r0_cm, s=S, n=1.0)
        Rl, Ru = dist.bounds()
        mu_01, wt_01 = quadrature('T', N_PTS)
        R_nodes, wts = shiftQuadrature(mu_01, wt_01, Rl, Ru)
        pdf_vals = dist.pdf(R_nodes)

        V_avg = float(np.dot(wts, (4.0/3.0)*np.pi * R_nodes**3 * pdf_vals))
        B_ext = np.empty(N_NU)
        for j, (nu, m) in enumerate(zip(wavenumber, Cm_in)):
            x_arr = np.ascontiguousarray(2.0 * np.pi * R_nodes * nu)
            qext, _, _ = _mie_batch_f90(x_arr, float(m.real), float(-m.imag))
            B_ext[j] = np.dot(wts, qext * np.pi * R_nodes**2 * pdf_vals)
        MEC_all[i] = B_ext / (DENSITY * V_avg)

    return MEC_all


# ---------------------------------------------------------------------------
# Run and time
# ---------------------------------------------------------------------------

timings: dict[str, float] = {}
results: dict[str, np.ndarray] = {}

for name, fn in [('Python batch', run_python_batch),
                 ('Fortran batch', run_fortran_batch)]:
    t0 = time.perf_counter()
    results[name] = fn()
    timings[name] = time.perf_counter() - t0
    print(f'{name}: {timings[name]:.1f} s')

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
t_py = timings['Python batch']
t_fo = timings['Fortran batch']

diff_rel = np.max(
    np.abs(results['Fortran batch'] - results['Python batch']) /
    np.abs(results['Python batch'])
)

print()
print(f'{"Backend":<20}  {"Wall time":>12}  {"vs Python":>10}  {"vs Fortran":>10}')
print('-' * 60)
for name, t in timings.items():
    print(f'{name:<20}  {t:>9.1f} s     {t/t_py:>8.1f}x  {t/t_fo:>9.1f}x')
print()
print(f'Fortran vs Python  max relative diff: {diff_rel:.2e}')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_bar, ax_spec) = plt.subplots(
    1, 2, figsize=(12, 4.5),
    gridspec_kw={'width_ratios': [1, 2]},
    constrained_layout=True,
)

# --- Bar chart ---
names  = list(timings.keys())
colors = ['#4c72b0', '#55a868']
vals   = [timings[n] for n in names]
bars   = ax_bar.bar(names, vals, color=colors, edgecolor='k', linewidth=0.6, width=0.5)
ax_bar.set_ylabel('Wall clock time (s)', fontsize=10)
ax_bar.set_title(f'{N_SIZES} sizes × {N_NU} wavenumbers\n({N_PTS} quadrature pts/size)',
                 fontsize=10)
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f} s'))
ax_bar.grid(axis='y', lw=0.4, alpha=0.5)
for bar, val in zip(bars, vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f'{val:.1f} s', ha='center', va='bottom', fontsize=9)

# --- Spectral overlay: Python (solid) vs Fortran (dashed) ---
cmap = get_cmap(ri.category)
norm = mcolors.LogNorm(vmin=R0_UM[0], vmax=R0_UM[-1])
idx_show = [0, N_SIZES//4, N_SIZES//2, 3*N_SIZES//4, N_SIZES-1]
for idx in idx_show:
    color = cmap(norm(R0_UM[idx]))
    ax_spec.plot(wavenumber, results['Python batch'][idx],
                 color=color, lw=1.2, ls='-',  alpha=0.9)
    ax_spec.plot(wavenumber, results['Fortran batch'][idx],
                 color=color, lw=0.8, ls='--', alpha=0.7)

ax_spec.set_xlim(WAVN_MIN, WAVN_MAX)
ax_spec.set_ylim(bottom=0)
ax_spec.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
ax_spec.set_ylabel('MEC (cm$^2$ g$^{-1}$)', fontsize=10)
ax_spec.set_title('MEC — Python (—) vs Fortran (- -)', fontsize=10)
ax_spec.grid(lw=0.3, alpha=0.4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # cmap is a Colormap object
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_spec, fraction=0.03, pad=0.01)
cbar.set_label('$r_0$ (µm)', fontsize=9)
cbar.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

from matplotlib.lines import Line2D
ax_spec.legend(handles=[
    Line2D([0],[0], color='k', lw=1.2, ls='-',  label='Python batch'),
    Line2D([0],[0], color='k', lw=0.8, ls='--', label='Fortran batch'),
], fontsize=8)

fig.suptitle(
    f'Backend benchmark — H$_2$SO$_4$ 72 wt%  |  '
    f'{N_SIZES} sizes × {N_NU} wavenumbers  |  '
    f'{N_PTS} quadrature pts/size',
    fontsize=11,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {OUTPUT_FILE}')
plt.show()
