"""
Automated backend performance and accuracy tests.

Tests that both Python and Fortran backends produce correct results and
verifies that Fortran provides the expected speedup.

Configuration
-------------
  H2SO4 (72 wt%, 298 K) — Lund Myhre et al. (2003)
  10 log-spaced median radii (0.1–10 µm),  s = 1.5
  Spectral grid: 650–1350 cm⁻¹ at 1 cm⁻¹  (701 points)
  100 Gauss quadrature points per size
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure mieEODG can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mieEODG import ARIADatabase, LogNormalDist, AerosolXSC, h2so4_density, get_cmap
from demo_config import make_wavenumber_grid, get_output_path

# ============================================================================
# Fixtures and shared configuration
# ============================================================================

ARIA_PATH = (
    'Acids/Sulphuric/70%_to_79%/'
    '72%,_298_K_(Lund_Myhre_et_al._2003)/'
    'H2SO4_72_298K_Myhre_2003.ri'
)

@pytest.fixture(scope='module')
def setup_data():
    """Load RI and set up test configuration."""
    S = 1.5
    N_SIZES = 10  # Smaller for faster testing
    N_PTS = 100
    R0_UM = np.logspace(np.log10(0.1), np.log10(10.0), N_SIZES)
    wavenumber = make_wavenumber_grid('thermal_ir', wn_step=1.0)

    db = ARIADatabase()
    ri = db.read(ARIA_PATH, wavenumber=wavenumber, out_of_range='clip')
    density = h2so4_density(0.72)

    return {
        'ri': ri,
        'wavenumber': wavenumber,
        'R0_UM': R0_UM,
        'S': S,
        'N_PTS': N_PTS,
        'density': density,
    }


# ============================================================================
# Helper functions
# ============================================================================

def _fortran_available() -> bool:
    """Check if Fortran extension is available."""
    try:
        from mieEODG.fortran._mie_kernel_f90 import mie_batch  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


# ============================================================================
# Tests
# ============================================================================

class TestBackendAvailability:
    """Test that backends can be imported."""

    def test_python_backend_available(self, setup_data):
        """Python backend should always be available."""
        ri = setup_data['ri']
        dist = LogNormalDist.from_um(1.0, 1.5, n=1.0)
        wn = setup_data['wavenumber'][:10]  # Small test

        xsc = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='python')
        xsc.compute(progress=False)
        assert xsc.result is not None
        assert xsc.MEC is not None

    @pytest.mark.skipif(
        not _fortran_available(),
        reason="Fortran extension not built"
    )
    def test_fortran_backend_available(self, setup_data):
        """Fortran backend should be available if built."""
        ri = setup_data['ri']
        dist = LogNormalDist.from_um(1.0, 1.5, n=1.0)
        wn = setup_data['wavenumber'][:10]  # Small test

        xsc = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='fortran')
        xsc.compute(progress=False)
        assert xsc.result is not None
        assert xsc.MEC is not None


class TestBackendAccuracy:
    """Test numerical accuracy between backends."""

    @pytest.mark.skipif(
        not _fortran_available(),
        reason="Fortran extension not built"
    )
    def test_backends_produce_identical_results(self, setup_data):
        """Python and Fortran should produce identical results."""
        ri = setup_data['ri']
        R0_UM = setup_data['R0_UM'][:3]  # Use 3 sizes for speed
        wn = setup_data['wavenumber']
        S = setup_data['S']

        # Compute with both backends
        mec_py = []
        mec_f90 = []

        for r0_um in R0_UM:
            dist = LogNormalDist.from_um(r0_um, S, n=1.0)

            xsc_py = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='python')
            xsc_py.compute(progress=False)
            mec_py.append(xsc_py.MEC)

            xsc_f90 = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='fortran')
            xsc_f90.compute(progress=False)
            mec_f90.append(xsc_f90.MEC)

        mec_py = np.array(mec_py)
        mec_f90 = np.array(mec_f90)

        # Check they match within floating-point precision
        max_rel_diff = np.max(np.abs(mec_f90 - mec_py) / np.abs(mec_py))
        assert max_rel_diff < 1e-6, f"Max relative difference: {max_rel_diff:.2e}"


class TestBackendPerformance:
    """Test that Fortran is faster than Python."""

    @pytest.mark.skipif(
        not _fortran_available(),
        reason="Fortran extension not built"
    )
    def test_fortran_speedup(self, setup_data):
        """Fortran should be significantly faster than Python."""
        ri = setup_data['ri']
        R0_UM = setup_data['R0_UM'][:5]  # Use 5 sizes
        wn = setup_data['wavenumber']
        S = setup_data['S']

        # Time Python backend
        t_py_start = time.perf_counter()
        for r0_um in R0_UM:
            dist = LogNormalDist.from_um(r0_um, S, n=1.0)
            xsc = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='python')
            xsc.compute(progress=False)
        t_py = time.perf_counter() - t_py_start

        # Time Fortran backend
        t_f90_start = time.perf_counter()
        for r0_um in R0_UM:
            dist = LogNormalDist.from_um(r0_um, S, n=1.0)
            xsc = AerosolXSC(ri=ri, dist=dist, wn=wn, backend='fortran')
            xsc.compute(progress=False)
        t_f90 = time.perf_counter() - t_f90_start

        speedup = t_py / t_f90

        # Fortran should be at least 2x faster (conservative; usually 4-40x)
        assert speedup >= 2.0, (
            f"Fortran speedup {speedup:.1f}x is less than expected minimum 2.0x. "
            f"Python: {t_py:.2f}s, Fortran: {t_f90:.2f}s"
        )

        print(f"\nFortran speedup: {speedup:.1f}x  (Python: {t_py:.2f}s, Fortran: {t_f90:.2f}s)")
