# mieEODG

Mie scattering routines for spherical aerosol particles, ported from the
[EODG IDL package](https://eodg.atm.ox.ac.uk/MIE/) (University of Oxford,
Atmospheric, Oceanic and Planetary Physics).

Provides mass extinction, scattering, and absorption coefficients (MEC, MSC,
MAC) for a given complex refractive index and size distribution, with a
precomputed look-up table (LUT) system for fast retrieval use.

---

## Features

- **Mie kernel** — Python/NumPy and compiled Fortran backends (40× faster)
- **Size distributions** — LogNormal, ModifiedGamma, Gaussian
- **ARIA refractive index database** — bundled `.ri` files for volcanic ash,
  sulphuric acid, nitric acid, liquid water, ice, mineral dust, sea salt,
  metals, minerals, and more
- **Composition-aware density** — auto-resolved from ARIA metadata
  (H₂SO₄ EOS, ice T-dependent, SiO₂-based ash/rock model)
- **AerosolLUT** — precomputed (MEC, MSC, MAC) grids over a r0 sweep with
  PCHIP interpolators; persistent `.nc` cache keyed by SHA-256 hash of inputs
- **Scientific colormaps** — per-ARIA-category colormaps via
  [cmcrameri](https://www.fabiocrameri.ch/colourmaps/)

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| NumPy | any recent | |
| SciPy | any recent | |
| xarray | optional | required for LUT save/load |
| tqdm | optional | progress bars |
| matplotlib | optional | demo scripts |
| cmcrameri | optional | scientific colormaps |
| netCDF4 | optional | compressed LUT files (recommended) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mieEODG.git
cd mieEODG
```

### 2. Create a virtual environment

**conda (recommended):**
```bash
conda create -n mie python=3.11
conda activate mie
conda install numpy scipy xarray tqdm matplotlib
pip install cmcrameri netCDF4
```

**venv:**
```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install numpy scipy xarray tqdm matplotlib cmcrameri netCDF4
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

### 4. Build the Fortran backend (optional but recommended)

The Fortran backend provides a ~40× speedup for the Mie kernel by replacing
the Python/NumPy inner loop with a compiled Fortran 90 routine via f2py.

**Requirements:** a Fortran compiler (`gfortran` on Linux/macOS, Intel `ifort`,
or MSVC with MinGW on Windows) and NumPy (for f2py).

```bash
cd mieEODG/fortran
python -m numpy.f2py -c mie_kernel.f90 -m _mie_kernel_f90
cd ../..
```

Verify the build:
```python
from mieEODG.fortran._mie_kernel_f90 import mie_batch
```

If the import succeeds, the Fortran backend is available.  Pass
`backend='fortran'` to `AerosolXSC` or `AerosolLUT.compute` /
`AerosolLUT.load_or_compute` to use it.

> **Note:** The compiled `.so` / `.pyd` file is platform-specific and is not
> committed to the repository.  Each user must build it locally.

---

## Quick start

```python
import numpy as np
from mieEODG import ARIADatabase, AerosolXSC, AerosolLUT, LogNormalDist

# Spectral grid
wn = np.arange(650, 1350.5, 0.5)   # cm⁻¹

# Load refractive index from the bundled ARIA database
db = ARIADatabase()
ri = db.read('volcanic_ash', wavenumber=wn, out_of_range='clip')

# Single computation — one size distribution
xsc = AerosolXSC(
    ri   = ri,
    dist = LogNormalDist(r0=1e-4, s=1.5, n=1.0),   # r0 in cm
    wn   = wn,
)
xsc.compute()
print(xsc.MEC)   # mass extinction coefficient [cm²/g], shape (N_wn,)

# LUT — precompute over a r0 grid, cache to ~/.mieEODG/lut_cache/
r0_grid = np.logspace(-1, 1, 20)   # µm
lut = AerosolLUT.load_or_compute(ri, r0_um=r0_grid, s=1.5, wn=wn)

# Fast forward at any r0 within the grid bounds
spec = lut.forward(r0_um=1.2)      # {'MEC': ..., 'MSC': ..., 'MAC': ...}

# Inspect the cache
AerosolLUT.list_cache()
```

---

## ARIA database

The bundled ARIA (Aerosol Refractive Index Archive) database contains
measured complex refractive indices for a wide range of aerosol species.
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
```

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

LUTs are cached to `~/.mieEODG/lut_cache/` by default.  The cache key is a
SHA-256 hash of the defining parameters (RI source path, distribution type,
s, r0 grid, wavenumber grid, n_pts).  Any change in inputs automatically
produces a new cache entry; stale entries are never silently reused.

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
| `demo_mie_size_dist.py` | Volcanic ash — LogNormal vs Gaussian size sweep, 2×3 panel figure |
| `demo_LUT.py` | LUT validation (Part 1) + multi-species MEC survey (Part 2) |
| `benchmark_backends.py` | Python vs Fortran speed and accuracy comparison |

All demo scripts have a `USER CONFIGURATION` section at the top — edit
spectral range, size grid, backend, and species without touching the
computation code.

---

## Project structure

```
mieEODG/
├── mieEODG/
│   ├── aerosol.py          # AerosolXSC — primary computation object
│   ├── lut.py              # AerosolLUT — LUT generation, caching, forward model
│   ├── refindex.py         # ARIADatabase, RefrIndex
│   ├── size_dist.py        # LogNormalDist, ModifiedGammaDist, GaussianDist
│   ├── constants.py        # densities, colormaps, ARIA_DEFAULTS
│   ├── mie_single.py       # mieSingle — single-particle Mie efficiencies
│   ├── mie_size_dist.py    # mieSizeDist — size-integrated bulk properties
│   ├── quadrature.py       # Gauss quadrature utilities
│   ├── fortran/
│   │   ├── mie_kernel.f90  # Fortran 90 Mie kernel source
│   │   └── *.so / *.pyd    # compiled extension (build locally, not committed)
│   └── data/
│       └── ARIA/           # bundled refractive index database
│           ├── Acids/
│           ├── Volcanic_Ash/
│           ├── Water_and_Ice/
│           └── ...
├── tests/
│   ├── demo_mie_size_dist.py
│   ├── demo_LUT.py
│   └── benchmark_backends.py
└── pyproject.toml
```

---

## Provenance

Ported from the EODG IDL Mie scattering package:
[https://eodg.atm.ox.ac.uk/MIE/](https://eodg.atm.ox.ac.uk/MIE/)

The ARIA refractive index database is distributed with permission from the
Oxford EODG group:
[https://eodg.atm.ox.ac.uk/ARIA/](https://eodg.atm.ox.ac.uk/ARIA/)

---

## License

The Python implementation in this package is original work by the authors.

The underlying Mie scattering algorithms are ported from the EODG IDL Mie
package developed at the Atmospheric, Oceanic and Planetary Physics group,
University of Oxford:
[https://eodg.atm.ox.ac.uk/MIE/](https://eodg.atm.ox.ac.uk/MIE/)

The bundled ARIA (Aerosol Refractive Index Archive) refractive index data
are provided by the Oxford EODG and redistributed here with permission.
Users of this data should cite the original measurement sources listed in
the individual `.ri` file headers, and acknowledge the ARIA archive:
[https://eodg.atm.ox.ac.uk/ARIA/](https://eodg.atm.ox.ac.uk/ARIA/)

A formal open-source license for this package is yet to be determined.
Until one is assigned, all rights are reserved by the authors.
