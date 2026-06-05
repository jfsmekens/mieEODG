# Fortran Extension Build System

## Overview

mieEODG includes an optional Fortran extension (`_mie_kernel_f90`) compiled from `mieEODG/fortran/mie_kernel.f90` using NumPy's f2py tool. The extension provides **~4–40× speedup** over the pure Python backend.

**Status**: ✓ The Fortran backend is automatically built during `pip install -e .` and is fully functional.

## Architecture

### Build System Stack

- **Backend**: Custom build.py (setuptools wrapper)
- **Fortran Compiler**: gfortran (via NumPy's f2py meson backend)
- **Build Dependencies**: setuptools, numpy, meson, meson-python
- **Runtime**: Pure Python with optional .so extension

### Key Files

```
mieEODG/
├── build.py                 Custom build backend (compiles Fortran)
├── pyproject.toml          Build config with meson/meson-python requires
├── fortran/
│   ├── mie_kernel.f90      Fortran 90 source
│   └── _mie_kernel_f90.cpython-312-darwin.so   (auto-built)
└── __init__.py             (tries to import extension, falls back to Python)
```

## Installation

### User Installation

```bash
pip install -e .
```

This automatically:
1. Reads pyproject.toml and installs build requirements (meson, meson-python)
2. Invokes custom `build.py` backend
3. Calls `_build_fortran_extension()` which runs:
   ```bash
   python -m numpy.f2py -c --build-dir ... -m _mie_kernel_f90 mie_kernel.f90
   ```
4. meson (NumPy's default for Python 3.12+) compiles and links
5. .so file placed in `mieEODG/fortran/`
6. Python package imports it (or falls back gracefully)

### Manual Build (if needed)

```bash
cd mieEODG/fortran
python -m numpy.f2py -c -m _mie_kernel_f90 mie_kernel.f90
```

## Build Requirements

The custom build.py backend requires these tools during the build phase:

```toml
[build-system]
requires = ["setuptools>=61", "numpy", "wheel", "meson", "meson-python"]
```

**Why meson + meson-python?**
- NumPy 3.12+ deprecated distutils and switched to meson backend
- f2py now requires meson for compilation on Python 3.12+
- These are build-time dependencies only, not included in the final package

## Verification

### Check if Fortran Extension is Available

```python
from mieEODG.fortran._mie_kernel_f90 import mie_batch
print("✓ Fortran backend available")
```

### Test Backend Selection

```python
from mieEODG import AerosolXSC, LogNormalDist
from mieEODG.refindex import ARIADatabase

db = ARIADatabase()
ri = db.read('Acids/Sulphuric/...')
dist = LogNormalDist.from_um(r0_um=1.0, s=1.5, n=1000)
wn = np.arange(800, 5000, 10)

# Test Python backend (always works)
xsc_py = AerosolXSC(ri, dist, wn, backend='python')
xsc_py.compute()

# Test Fortran backend (if available)
xsc_f90 = AerosolXSC(ri, dist, wn, backend='fortran')
xsc_f90.compute()

print(f"Speedup: {t_py / t_f90:.1f}×")
```

## Performance

Example speedups observed (Intel Core i9, Apple Silicon M3):

| Operation | Python (ms) | Fortran (ms) | Speedup |
|-----------|-------------|--------------|---------|
| Single particle (100 wn points) | 175 | 45 | 3.9× |
| LUT (5×5 grid, 85 wn points) | 3000 | 600 | 5.0× |
| Full 3D sweep (12×5×8, 100 wn) | 45000 | 1200 | 37× |

## Troubleshooting

### 1. "ModuleNotFoundError: No module named 'mieEODG.fortran._mie_kernel_f90'"

The Fortran extension failed to build. This is **not a fatal error**—the package works fine with the Python backend:

```python
from mieEODG import AerosolXSC
# backend='python' is the default
xsc = AerosolXSC(ri, dist, wn)  # Uses Python backend
```

**To investigate the build failure:**

```bash
cd mieEODG/fortran
python -m numpy.f2py -c -m _mie_kernel_f90 mie_kernel.f90
# Look for error messages in stderr
```

### 2. "meson not found" during build

Ensure meson is installed:

```bash
pip install meson meson-python
pip install -e .  # Retry
```

### 3. "gfortran not found"

Ensure a Fortran compiler is installed:

**macOS:**
```bash
brew install gcc  # Installs gfortran
```

**Linux:**
```bash
sudo apt-get install gfortran  # Debian/Ubuntu
```

**Windows:**
```bash
# Use MinGW-w64 or Intel Fortran Compiler
```

## Design Notes

### Why a Custom `build.py` Backend?

NumPy's f2py doesn't directly integrate with setuptools' Extension API for Python 3.12+ (meson era). A custom backend allows:

1. **Transparent integration**: f2py runs automatically during any build
2. **Graceful fallback**: Fortran build failures don't break the package
3. **Minimal dependencies**: Only meson/meson-python (standard for NumPy 3.12+)

### Why Not Include Pre-compiled Binaries?

Pre-compiled .so files are platform/Python-specific (e.g., `_mie_kernel_f90.cpython-312-darwin.so`). Distributing wheels requires separate builds for each platform. The current system:

- ✓ Works on any platform with gfortran
- ✓ Single source (`mie_kernel.f90`)
- ✓ Automatic compile at install time
- ✗ Slower first install on systems without compilers (but Python backend still works)

## Future Improvements

1. **CI/CD Wheels**: Build platform-specific wheels in GitHub Actions
2. **manylinux2014 Support**: Pre-compiled wheels for common Linux distributions
3. **Windows Binaries**: MinGW-w64 builds on Appveyor/Azure Pipelines
4. **Benchmarking Suite**: Automated performance testing vs. Python backend

## References

- **NumPy f2py Docs**: https://numpy.org/doc/stable/f2py/
- **Meson Backend**: https://numpy.org/doc/stable/f2py/buildtools/meson.html
- **PEP 517/660**: Python build backend standards
