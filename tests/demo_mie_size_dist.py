"""
Demonstration: LogNormal vs Gaussian size distribution comparison for volcanic ash.

Uses Etna volcanic ash (Deguine et al., 2020) as the reference aerosol.
A log-spaced size sweep from 0.1 to 10 µm is computed for both distribution
types with fixed spread s = 1.5.  For the Gaussian, sigma = r0 * ln(s),
giving comparable width to the log-normal near r0.

Layout: 2 rows × 3 columns
  Row 0 — LogNormal
  Row 1 — Gaussian
  Columns — MEC, MSC, MAC

Color convention: light (small r0) → dark (large r0) using the grayC_r colormap.

Output
------
  Saves  tests/demo_size_dist.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from tqdm.auto import tqdm

from mieEODG import (
    AerosolXSC, ARIADatabase,
    LogNormalDist, GaussianDist,
    get_cmap, ARIA_DEFAULTS,
)
from mieEODG.plotting import _add_top_axis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RI_PATH = ARIA_DEFAULTS['volcanic_ash']

WAVN_MIN  = 650.0
WAVN_MAX  = 1350.0
WAVN_STEP = 0.5
wavenumber = np.arange(WAVN_MIN, WAVN_MAX + WAVN_STEP / 2, WAVN_STEP)

S     = 1.5
N_R0  = 10
R0_UM = np.logspace(np.log10(0.1), np.log10(10.0), N_R0)   # µm

NORM = mcolors.LogNorm(vmin=R0_UM[0], vmax=R0_UM[-1])

OUTPUT_FILE = Path(__file__).parent / 'demo_size_dist.png'

PANELS = [
    ('MEC', r'MEC  (cm$^2$ g$^{-1}$)',  'Mass extinction coefficient'),
    ('MSC', r'MSC  (cm$^2$ g$^{-1}$)',  'Mass scattering coefficient'),
    ('MAC', r'MAC  (cm$^2$ g$^{-1}$)',  'Mass absorption coefficient'),
]

DIST_CONFIGS = [
    ('log_normal', 'Log-Normal', lambda r: LogNormalDist.from_um(r, S, n=1.0)),
    ('gaussian',   'Gaussian',   lambda r: GaussianDist.from_um(r, r * np.log(S), n=1.0)),
]

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

db = ARIADatabase()
ri = db.read(RI_PATH, wavenumber=wavenumber, out_of_range='clip')

cmap   = get_cmap(ri.category)
colors = [cmap(NORM(r)) for r in R0_UM]

results = {}
for dist_key, dist_label, dist_factory in DIST_CONFIGS:
    MEC_all = np.empty((N_R0, len(wavenumber)))
    MSC_all = np.empty_like(MEC_all)
    MAC_all = np.empty_like(MEC_all)

    with tqdm(total=N_R0, desc=f'{dist_label}', unit='r0') as pbar:
        for i, r0_um in enumerate(R0_UM):
            pbar.set_description(f'{dist_label}  r₀={r0_um:.2g} µm')
            xsc = AerosolXSC(ri=ri, dist=dist_factory(r0_um), wn=wavenumber)
            xsc.compute(progress=False)
            MEC_all[i] = xsc.result.MEC
            MSC_all[i] = xsc.result.MSC
            MAC_all[i] = xsc.result.MAC
            pbar.update(1)

    results[dist_key] = dict(
        label   = dist_label,
        MEC     = MEC_all,
        MSC     = MSC_all,
        MAC     = MAC_all,
        density = xsc.density,
    )

# ---------------------------------------------------------------------------
# Figure  (2 rows × 3 columns)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 3,
    figsize=(14, 8),
    sharex=True, sharey='col',
    constrained_layout=True,
)

for row, (dist_key, dist_label, _) in enumerate(DIST_CONFIGS):
    data = results[dist_key]

    for col, (key, ylabel, title) in enumerate(PANELS):
        ax = axes[row, col]
        M  = data[key]

        for i, color in enumerate(colors):
            ax.plot(wavenumber, M[i], color=color, lw=0.9, alpha=0.9)

        ax.set_xlim(WAVN_MAX, WAVN_MIN)
        ax.set_ylim(bottom=0)

        if row == 0:
            ax.set_title(title, fontsize=10)
            _add_top_axis(ax)
        if col == 0:
            density_str = f'ρ = {data["density"]:.3f} g/cm³'
            ax.set_ylabel(f'{dist_label}\n{density_str}\n\n{ylabel}', fontsize=9)
        else:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == len(DIST_CONFIGS) - 1:
            ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)

# Shared colorbar on the right of each row
sm = mcm.ScalarMappable(cmap=cmap, norm=NORM)
sm.set_array([])
for row in range(2):
    cbar = fig.colorbar(sm, ax=axes[row, -1], pad=0.015, fraction=0.05, aspect=20)
    cbar.set_label('$r_0$ (µm)', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2g}'))

substance_str = ri.substance or ri.category
fig.suptitle(
    f'Spectral mass coefficients — {substance_str}\n'
    r'LogNormal vs Gaussian size sweep  ($s = 1.5$,  $r_0 = 0.1$–$10\,\mu$m)',
    fontsize=12,
)

fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {OUTPUT_FILE}')
plt.show()
