"""
Custom build backend hook to compile Fortran extension before package build.

This runs f2py to compile mie_kernel.f90 → _mie_kernel_f90.so before setuptools
processes the package data.

The Fortran extension provides ~4-40× speedup for Mie calculations and is optional.
If the build fails, the pure Python backend remains available.
"""

from pathlib import Path
import subprocess
import sys


def _build_fortran_extension() -> None:
    """Compile Fortran code using f2py with NumPy's meson backend.

    This function is called before building wheels or editable installs.
    Failures are silently caught since the Python backend is always available.
    """
    project_root = Path(__file__).parent
    fortran_src = project_root / "mieEODG" / "fortran" / "mie_kernel.f90"
    output_dir = project_root / "mieEODG" / "fortran"

    # Check if already built to avoid redundant builds
    so_file = output_dir / "_mie_kernel_f90.cpython-312-darwin.so"
    if so_file.exists():
        return

    try:
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c",
            "--build-dir", str(output_dir / ".f2py_build"),
            "-m", "_mie_kernel_f90",
            str(fortran_src),
        ]

        result = subprocess.run(
            cmd, cwd=str(output_dir), check=False,
            capture_output=True, text=True
        )
        # Silently succeed or fail - Python backend is always available

    except Exception:
        # Failures are not fatal - Python backend is always available
        pass


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel: compile Fortran extension first, then delegate to setuptools."""
    _build_fortran_extension()
    from setuptools.build_meta import build_wheel as _build_wheel
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution: compile Fortran extension first, then delegate to setuptools."""
    _build_fortran_extension()
    from setuptools.build_meta import build_sdist as _build_sdist
    return _build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings=None):
    """Return build requirements."""
    from setuptools.build_meta import get_requires_for_build_wheel as _get_requires
    return _get_requires(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata: compile Fortran first, then delegate to setuptools."""
    _build_fortran_extension()
    from setuptools.build_meta import prepare_metadata_for_build_wheel as _prepare
    return _prepare(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable wheel: compile Fortran extension first, then delegate to setuptools."""
    _build_fortran_extension()
    from setuptools.build_meta import build_editable as _build_editable
    return _build_editable(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable: compile Fortran first, then delegate to setuptools."""
    _build_fortran_extension()
    from setuptools.build_meta import prepare_metadata_for_build_editable as _prepare
    return _prepare(metadata_directory, config_settings)
