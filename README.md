# mieEODG

Mie scattering routines for spherical aerosol particles, ported from the
[EODG IDL package](https://eodg.atm.ox.ac.uk/MIE/) (University of Oxford,
Atmospheric, Oceanic and Planetary Physics).

Provides mass extinction, scattering, and absorption coefficients (MEC, MSC,
MAC) for a given complex refractive index and size distribution, with a
precomputed look-up table (LUT) system for fast retrieval use.

---

## Features

- **Mie kernel** — Python/NumPy and compiled Fortran backends (40× faster);
  automatic build via custom PEP 517 backend for all Python versions
- **Size distributions** — LogNormal, ModifiedGamma, Gaussian, Monodisperse
  (exact single particle), and Tabulated (measured/custom PDFs) with easy
  µm-to-cm conversion via `from_um()` constructors; `savePDF()` / `loadPDF()`
  for CSV round-tripping of a distribution
- **ARIA refractive index database** — 450 `.ri` files shipped with the
  package, covering volcanic ash, sulphuric acid, nitric acid, liquid water,
  ice, mineral dust, sea salt, metals, minerals, and more.  No download or
  configuration needed — see [ARIA database](#aria-database)
- **Prata et al. (2019) volcanic ash RI parameterization** — generate complex
  refractive indices from bulk oxide composition using linear spectral
  regression (NBO/T or SiO₂); includes automatic Fe²⁺/Fe³⁺ split via
  Middlemost (1989) rock-type ratios
- **Composition-aware density** — auto-resolved from ARIA metadata
  (H₂SO₄ EOS via Lund Myhre 2003, ice T-dependent, SiO₂-based ash/rock model)
- **Single-particle Mie** — `AerosolXSC` for individual calculations with optional
  phase function and Legendre coefficient export
- **Size-integrated Mie with LUT caching** — `AerosolLUT` for precomputed grids
  over r₀ with PCHIP interpolators; persistent `.nc` cache keyed by SHA-256
  hash of inputs; optional phase function caching for fast DISORT queries
- **N-dimensional LUTs** — `MultiDimLUT` for sweeps across radius, composition,
  and temperature with single-source consistency enforcement and automatic ARIA
  database integration via fuzzy matching; stores all 13 optical quantities
  (Q/C/B ext·sca·abs, MEC/MSC/MAC, g) per grid point and interpolates via a
  cached vector-valued `RegularGridInterpolator`
- **Phase function** — size-integrated scattering phase matrix [F11, F33, F12,
  F34] at user-supplied angles, plus Legendre expansion coefficients (`pmom`)
  in DISORT convention; LRU in-memory cache with optional disk storage
- **RFM export** — write HITRAN-format `.xsc` files for the Reference Forward
  Model aerosol molecule (index 100); dual-mode export (generic km⁻¹ or
  pre-scaled cm²/molec); supports multi-tabulation from N-D LUTs
- **Scientific colormaps** — per-ARIA-category colormaps via
  [cmcrameri](https://www.fabiocrameri.ch/colourmaps/)
- **Graphical interface** — `MieViewer`, a Tkinter desktop GUI to browse the
  ARIA database, compute single-particle and phase-function results, and build
  and query 1-D / N-D LUTs interactively (launched with the `mie-viewer`
  console script)

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| NumPy | any recent | |
| SciPy | any recent | |
| xarray | optional | required for LUT save/load |
| tqdm | optional | progress bars |
| matplotlib | optional | demo scripts, plotting, GUI |
| cmcrameri | optional | scientific colormaps |
| netCDF4 | optional | compressed LUT files (recommended) |
| tkinter | optional | `MieViewer` GUI (bundled with most Python builds) |

---

## Installation

### Build requirements

mieEODG compiles a Fortran kernel at install time. You need a **Fortran
compiler** on your `PATH`:

| Platform | Compiler |
|---|---|
| macOS | `brew install gcc` (provides `gfortran`) |
| Linux | `sudo apt install gfortran` (or your distribution's equivalent) |
| conda, any platform | `conda install -c conda-forge gfortran` |

The other build tools (NumPy's f2py, `meson`, `meson-python`, `ninja`) are
declared in `pyproject.toml` and fetched automatically by pip. No compiler,
no problem: the install still succeeds and falls back to the pure
Python/NumPy kernel, which gives identical results ~4-40× more slowly.
**The fallback is silent**, so check which backend you got (below).

### Install from GitHub

```bash
pip install "git+https://github.com/jfsmekens/mieEODG.git"          # core
pip install "mieEODG[full] @ git+https://github.com/jfsmekens/mieEODG.git"   # + LUT files, plots, progress bars
```

There are no releases on PyPI; rerun with `--upgrade` to pick up fixes. The
optional extras are `lut` (xarray, netCDF4), `plot` (matplotlib, cmcrameri),
`progress` (tqdm), `full` (all of these) and `dev` (pytest).

### Development install

```bash
git clone https://github.com/jfsmekens/mieEODG.git
cd mieEODG
pip install -e ".[full,dev]"
```

### Check the Fortran backend

```python
from mieEODG.fortran._mie_kernel_f90 import mie_batch   # ImportError → Python fallback
```

If the import succeeds, pass `backend='fortran'` to `AerosolXSC` or
`AerosolLUT.load_or_compute()` to use the compiled extension. If it fails,
install a compiler and reinstall, or build it in place in a development
checkout:

```bash
cd mieEODG/fortran
python -m numpy.f2py -c mie_kernel.f90 -m _mie_kernel_f90
```

The compiled `.so` / `.pyd` is specific to the platform and Python version
(e.g. `_mie_kernel_f90.cpython-312-darwin.so`), so each environment builds its
own; no pre-built wheels are distributed. See [BUILD_FORTRAN.md](BUILD_FORTRAN.md)
for details of the build backend.

---

## Quick start

```python
import numpy as np
from mieEODG import (ARIADatabase, AerosolXSC, AerosolLUT, LogNormalDist,
                     ARIA_DEFAULTS)

# Spectral grid
wn = np.arange(650, 1350.5, 0.5)   # cm⁻¹

# Load refractive index from the ARIA database.  Entries are addressed by
# their path within the archive; ARIA_DEFAULTS maps short keys to those paths.
db = ARIADatabase()
ri = db.read(ARIA_DEFAULTS['volcanic_ash'], wavenumber=wn, out_of_range='clip')

# Single computation — one size distribution
xsc = AerosolXSC(
    ri   = ri,
    dist = LogNormalDist(r0=1e-4, s=1.5, n=1.0),   # r0 in cm
    wn   = wn,
)
xsc.compute()
print(xsc.MEC)   # mass extinction coefficient [cm²/g], shape (N_wn,)
print(xsc.g)     # asymmetry parameter, shape (N_wn,)

# Phase function and Legendre coefficients (for DISORT)
theta = np.linspace(0, 180, 181)   # scattering angles [degrees]
xsc.compute(theta_pts=theta, n_legendre=64)
print(xsc.phase_matrix)   # [F11, F33, F12, F34], shape (N_wn, 181, 4)
print(xsc.pmom)           # Legendre coefficients, shape (N_wn, 64), pmom[0]=1

# LUT — precompute over a r0 grid, cache to ~/.mieEODG/lut_cache/
r0_grid = np.logspace(-1, 1, 20)   # µm
lut = AerosolLUT.load_or_compute(ri, r0_um=r0_grid, s=1.5, wn=wn)

# Fast forward at any r0 within the grid bounds
spec = lut.forward(r0_um=1.2)      # {'MEC': ..., 'MSC': ..., 'MAC': ...}

# Inspect the cache
AerosolLUT.list_cache()
```

---

## Graphical interface (MieViewer)

`MieViewer` is an optional Tkinter desktop GUI for interactive exploration —
no scripting required. It lets you browse the ARIA database, load refractive
indices, compute single-particle and phase-function results, and build and
query 1-D and N-D LUTs, all with live plots, a progress bar, and a log console.

Launch it via the console script installed with the package:

```bash
mie-viewer
```

or from Python:

```python
from mieEODG import launch
launch()
```

The GUI requires `tkinter` (bundled with most Python builds) and `matplotlib`.

---

## ARIA database

ARIA (Aerosol Refractive Index Archive) contains measured complex refractive
indices for a wide range of aerosol species.  All 450 entries ship inside the
package, so `ARIADatabase()` works immediately after install with nothing to
download or configure.

To use a different copy — a newer archive, or a shared one — override it.
Resolution order, highest priority first:

| # | Location | Notes |
|---|---|---|
| 1 | `ARIADatabase(path=...)` | explicit argument |
| 2 | `$MIEODG_ARIA_ROOT` | environment override; set-but-missing is an error |
| 3 | `aria_dir` in `~/.config/mieEODG/config.ini` | written by `python -m mieEODG configure --aria-dir` |
| 4 | `mieEODG/data/ARIA` | **bundled — the default** |
| 5 | `~/ARIA` | copy kept outside the package |
| 6 | `$RFM_DIR/ARIA`, `~/RFM/ARIA` | legacy; logged once at INFO |

Check which one is in use:

```bash
python -m mieEODG configure
```

Layer 6 exists only for installs that unpacked ARIA inside the RFM tree,
because ARIA and RFM come from the same group and are often placed side by
side.  **RFM does not read ARIA** — it consumes the `.xsc` cross-sections
mieEODG writes — so the two are independent.

Use `ARIADatabase` to browse and load entries:

```python
db = ARIADatabase()
db.list()                           # print full category tree
db.list(category='Volcanic_Ash')    # list entries for a specific category

# Load by short canonical key (see ARIA_DEFAULTS in constants.py)
from mieEODG import ARIA_DEFAULTS
print(ARIA_DEFAULTS)                # {'volcanic_ash': '...', 'acids': '...', ...}

ri = db.read(ARIA_DEFAULTS['volcanic_ash'], wavenumber=wn, out_of_range='clip')
print(ri.substance)    # 'Etna ash'
print(ri.category)     # 'volcanic_ash'
print(ri.T_K)          # temperature [K] parsed from header, or None
print(ri.composition)  # e.g. {'SiO2': 0.58} for volcanic ash
print(ri.reference)    # the measurement study this material comes from
```

> **Publishing?** Cite the study in `ri.reference` for every material you use.
> See [License](#license).


---

## Backends

| Backend | How to select | Speed | Availability |
|---|---|---|---|
| Python | `backend='python'` (default) | baseline | always |
| Fortran | `backend='fortran'` | ~40× faster | requires build (see above) |

```python
# AerosolXSC
xsc = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='fortran')

# AerosolLUT
lut = AerosolLUT.load_or_compute(ri, r0_um=r0_grid, s=1.5, wn=wn,
                                  backend='fortran')
```

---

## LUT caching

LUTs are cached to the package folder `mieEODG/cache/lut/` by default (falling
back to `~/.mieEODG/lut_cache/` if the package folder is not writable).  The
cache key is a SHA-256 hash of the defining parameters (RI source path and
n,k data, distribution type, s, r0 grid, wavenumber grid, n_pts, density, and
phase settings).  Any change in inputs automatically produces a new cache
entry; stale entries are never silently reused.  N-D `MultiDimLUT` caches are
stored alongside as `ndlut_<hash>.nc`.

```python
# Force a recompute (overwrite the cached file)
lut = AerosolLUT.load_or_compute(..., recompute=True)

# Use a project-local cache instead of the user default
lut = AerosolLUT.load_or_compute(..., cache_dir='./lut_cache')

# List everything in the cache
AerosolLUT.list_cache()
```

---

## Demo scripts

Located in `tests/`:

| Script | Description |
|---|---|
| `demo_single.py` | Single particle Mie calculation — spectral MEC/MSC/MAC |
| `demo_size_dist.py` | Size distributions vs spectral coefficients — LogNormal, Gaussian, ModifiedGamma with side-by-side PDF + spectra |
| `demo_LUT.py` | LUT validation (Part 1: interpolation accuracy) + multi-species MEC survey (Part 2) |
| `demo_multidim_lut.py` | N-dimensional LUT generation and interpolation — r₀ × composition × temperature sweeps with ARIA database integration |
| `demo_prata.py` | Prata et al. (2019) volcanic ash RI parameterization — SiO₂-based and NBO/T-based composition series |
| `test_backends.py` (pytest) | Automated backend performance and accuracy tests — Python vs Fortran comparison |

**Shared configuration:** All demo scripts import `demo_config.py`, which provides
centralized control over spectral ranges, size grids, and output directories.
To customize runs globally without editing individual scripts, set the
`OVERRIDE_*` constants in `demo_config.py` (defaults to `None`, using preset
values).

Outputs are saved to `tests/demo_output/{demo_name}/` for easy organization.

---

## Project structure

```
mieEODG/
├── mieEODG/
│   ├── __init__.py         # public API
│   ├── aerosol.py          # AerosolXSC — single-particle Mie computation
│   ├── lut.py              # AerosolLUT — 1-D LUT with PCHIP interpolation
│   ├── lut_multi.py        # MultiDimLUT — N-D LUT with ARIA integration
│   ├── refindex.py         # ARIADatabase, RefrIndex, volcanic ash parameterization
│   ├── rfm.py              # RFM .xsc export (HITRAN format)
│   ├── size_dist.py        # LogNormalDist, ModifiedGammaDist, GaussianDist, MonodisperseDist, TabulatedDist
│   ├── functions.py        # savePDF / loadPDF — CSV round-trip for size distributions
│   ├── mie_viewer.py       # MieViewer — Tkinter GUI (mie-viewer console script)
│   ├── constants.py        # densities, colormaps, ARIA_DEFAULTS, SPECTRAL_RANGE_PRESETS
│   ├── mie_single.py       # mieSingle — single-particle Mie efficiencies
│   ├── mie_size_dist.py    # mieSizeDist — size-integrated bulk properties
│   ├── quadrature.py       # Gauss quadrature utilities
│   ├── plotting.py         # plot_spectrum, plot_lut, plot_lut_slice, plot_lut_heatmap, plot_phase_function, plot_phase_function_heatmap, plot_size_dist
│   ├── units.py            # spectral unit conversions (wavenumber ↔ wavelength)
│   ├── fortran/
│   │   ├── mie_kernel.f90  # Fortran 90 Mie kernel source
│   │   └── *.so / *.pyd    # compiled extension (build locally, not committed)
│   └── data/
│       ├── ARIA/           # bundled refractive index database (450 entries)
│       ├── Prata2019_coeff.csv  # volcanic ash RI parameterization coefficients
│       ├── lut_design_v1.md     # N-D LUT architecture reference
│       ├── aria_audit/          # ARIA database audit results
│       └── aria_metadata_registry.json  # metadata for N-D LUT construction
├── tests/
│   ├── demo_config.py      # centralized configuration for all demos
│   ├── demo_single.py      # single Mie calculation demo
│   ├── demo_size_dist.py   # size distribution vs spectra demo
│   ├── demo_LUT.py         # LUT validation + multi-species survey
│   ├── demo_multidim_lut.py # N-D LUT generation and interpolation
│   ├── demo_prata.py       # volcanic ash RI parameterization demo
│   ├── test_backends.py    # automated backend tests (pytest)
│   └── demo_output/        # demo figure outputs (git-ignored)
├── build.py                # custom PEP 517 backend for Fortran compilation
├── pyproject.toml          # build system config (meson + setuptools)
└── BUILD_FORTRAN.md        # detailed Fortran build system documentation
```

---

## Provenance

mieEODG is a Python and Fortran 90 port of the Mie scattering routines of the
Earth Observation Data Group (EODG), Atmospheric, Oceanic and Planetary
Physics, University of Oxford: <https://eodg.atm.ox.ac.uk/MIE/>. The original
IDL and Fortran routines were written by Don Grainger, Gareth Thomas, Andy
Smith, Greg McGarragh and colleagues (1990-2015). They are not included in
this repository; download them from EODG
([eodg_mie.tar.gz](https://eodg.atm.ox.ac.uk/MIE/eodg_mie.tar.gz)). Module
docstrings cite the original routine each function was ported from, e.g.
`IDLsource/mie_single.pro` refers to `mie_single.pro` in that distribution.

The bundled refractive index data are a copy of ARIA, the Aerosol Refractive
Index Archive (<https://eodg.atm.ox.ac.uk/ARIA/>), compiled at EODG by Don
Grainger, Dan Peters, Lieven Clarisse and Hervé Herbin. ARIA is a compilation
of measurements published by others; every source is listed in
[`mieEODG/data/ARIA/REFERENCES.md`](mieEODG/data/ARIA/REFERENCES.md).

mieEODG is not an official EODG product.

---

## Citing

If you publish results computed with mieEODG, cite:

1. **The EODG Mie code** this package is ported from:
   <https://eodg.atm.ox.ac.uk/MIE/>. If you use the derivative routines
   (`mie_derivs`, `mie_derivs_ln`), cite also
   > Grainger, R. G., Lucas, J., Thomas, G. E., and Ewen, G. B. L. (2004).
   > Calculation of Mie derivatives. *Applied Optics*, 43(28), 5386-5393.
   > doi:[10.1364/AO.43.005386](https://doi.org/10.1364/AO.43.005386)

2. **The measurement study behind every refractive index you use.** ARIA asks
   for the source publications to be acknowledged. Each `.ri` file records its
   source in the `#REFERENCE` and `#DOI` header fields, the parsed object
   exposes it, and [`REFERENCES.md`](mieEODG/data/ARIA/REFERENCES.md) lists
   all 90:

   ```python
   ri = db.read(ARIA_DEFAULTS['volcanic_ash'], wavenumber=wn)
   print(ri.reference)
   # A. Deguine, D. Petitprez, L. Clarisse, S. Gudmundsson, V. Outes,
   # G. Villarosa, and H. Herbin, "Complex refractive index of volcanic ash
   # aerosol in the infrared, visible, and ultraviolet," Appl. Opt. 59,
   # 884-895 (2020).
   ```

   A few entries were taken by ARIA from the HITRAN aerosol compilation;
   for those, cite also Massie, S. T. and Hervig, M. (2013), HITRAN 2012
   refractive indices, *JQSRT* 130, 373-380,
   doi:[10.1016/j.jqsrt.2013.06.022](https://doi.org/10.1016/j.jqsrt.2013.06.022).
   Acknowledging the ARIA archive and its compilers is appropriate in
   addition, but does not replace the primary citation.

3. **The volcanic ash parameterization**, if you use it
   (`ash_refrindex`, `composition.py`):
   > Prata, G. S., Ventress, L. J., Carboni, E., Mather, T. A., Grainger, R. G.,
   > and Pyle, D. M. (2019). A new parameterization of volcanic ash complex
   > refractive index based on NBO/T and SiO₂ content. *Journal of Geophysical
   > Research: Atmospheres*, 124(3), 1779-1797.
   > doi:[10.1029/2018JD028679](https://doi.org/10.1029/2018JD028679)

   with the Fe²⁺/Fe³⁺ split from Middlemost, E. A. K. (1989), Iron oxidation
   ratios, norms and the classification of volcanic rocks, *Chemical Geology*
   77, 19-26, doi:[10.1016/0009-2541(89)90011-9](https://doi.org/10.1016/0009-2541(89)90011-9).

The `.xsc` files written for RFM use HITRAN's cross-section format; RFM itself
is cited as Dudhia, A. (2017), *JQSRT* 186, 243-253,
doi:[10.1016/j.jqsrt.2016.06.018](https://doi.org/10.1016/j.jqsrt.2016.06.018).

---

## License

The mieEODG code is released under the [MIT License](LICENSE).

The licence covers the code of this package. The bundled ARIA data files
(`mieEODG/data/ARIA/`) remain the work of their original authors and of the
ARIA compilers; they are included for convenience, and their use is subject to
the citation requests above. `mieEODG/data/Prata2019_coeff.csv` reproduces the
regression coefficients of Prata et al. (2019).
