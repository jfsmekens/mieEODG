"""
Tests for MonodisperseDist as a first-class computational input.

A monodisperse (single-size) population is integrated *exactly* by
``mieSizeDist`` / ``AerosolXSC`` via a single-node evaluation (no quadrature,
no width smearing).  These tests pin that behaviour:

1. ``AerosolXSC(MonodisperseDist)`` reproduces the analytic single-particle
   optics (Q from ``mieSingle``, with C/B/M derived by hand).
2. The standard ``AerosolResult`` relations hold (Q = B/G_avg, C = B/N_avg,
   MEC = B_ext/(ρ·V_avg)).
3. Intensive quantities (Q, C, MEC) are independent of the number density
   N_d, while the extensive B_* coefficients scale linearly with it.
4. The phase matrix is available for a monodisperse population.
5. A narrow distribution converges toward the monodisperse limit.
"""

import numpy as np
import pytest

from mieEODG import (
    ARIADatabase,
    AerosolXSC,
    MonodisperseDist,
    LogNormalDist,
)
from mieEODG.mie_single import mieSingle
from mieEODG.constants import aria_density

ARIA_PATH = (
    'Acids/Sulphuric/70%_to_79%/'
    '72%,_298_K_(Lund_Myhre_et_al._2003)/'
    'H2SO4_72_298K_Myhre_2003.ri'
)


@pytest.fixture(scope='module')
def ri():
    """Refractive index on a small thermal-IR grid."""
    wn = np.linspace(900.0, 1300.0, 30)
    db = ARIADatabase()
    return db.read(ARIA_PATH, wavenumber=wn, out_of_range='clip')


@pytest.fixture(scope='module')
def density(ri):
    return aria_density(
        category=ri.category, composition=ri.composition,
        T_K=ri.T_K, substance_key=ri.substance_key,
    )


def _analytic_single(ri, r0_um, n_d, density):
    """Independent reference: Q from mieSingle, C/B/M derived analytically."""
    r0_cm = r0_um * 1.0e-4
    wn = ri.wavenumber
    geo = np.pi * r0_cm ** 2
    vol = (4.0 / 3.0) * np.pi * r0_cm ** 3

    q_ext = np.empty(len(wn))
    q_sca = np.empty(len(wn))
    for j in range(len(wn)):
        res = mieSingle(2.0 * np.pi * r0_cm * wn[j], complex(ri.Cm[j]), silent=True)
        q_ext[j] = res.Qext[0]
        q_sca[j] = res.Qsca[0]

    c_ext = q_ext * geo
    c_sca = q_sca * geo
    return {
        'Q_ext': q_ext, 'Q_sca': q_sca, 'Q_abs': q_ext - q_sca,
        'C_ext': c_ext, 'C_sca': c_sca, 'C_abs': c_ext - c_sca,
        'B_ext': n_d * c_ext, 'B_sca': n_d * c_sca,
        'MEC': c_ext / (density * vol),
        'MSC': c_sca / (density * vol),
    }


class TestMonodisperseDist:
    """The MonodisperseDist class itself."""

    def test_construction_and_units(self):
        m = MonodisperseDist.from_um(1.0, n=1e6)
        assert m.dist_name == 'monodisperse'
        assert m.r0_um == pytest.approx(1.0)
        assert m.r0 == pytest.approx(1.0e-4)
        assert m.N == pytest.approx(1e6)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            MonodisperseDist(r0=-1.0)
        with pytest.raises(ValueError):
            MonodisperseDist(r0=1e-4, n=0.0)


class TestMonodisperseEquivalence:
    """AerosolXSC(MonodisperseDist) == analytic single particle."""

    def test_matches_analytic(self, ri, density):
        r0_um, n_d = 1.0, 1.0e6
        xsc = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(r0_um, n=n_d),
                         wn=ri.wavenumber, backend='python')
        xsc.compute(progress=False)
        ref = _analytic_single(ri, r0_um, n_d, density)

        for key, expected in ref.items():
            np.testing.assert_allclose(
                getattr(xsc, key), expected, rtol=1e-10,
                err_msg=f"{key} mismatch vs analytic single particle",
            )

    def test_result_relations(self, ri):
        """Q = B/G_avg, C = B/N_avg, MEC = B_ext/(ρ·V_avg)."""
        r0_um, n_d = 0.7, 5.0e5
        xsc = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(r0_um, n=n_d),
                         wn=ri.wavenumber, backend='python')
        xsc.compute(progress=False)

        np.testing.assert_allclose(xsc.Q_ext, xsc.B_ext / xsc.G_avg, rtol=1e-12)
        np.testing.assert_allclose(xsc.C_ext, xsc.B_ext / xsc.N_avg, rtol=1e-12)
        np.testing.assert_allclose(
            xsc.MEC, xsc.B_ext / (xsc.density * xsc.V_avg), rtol=1e-12)
        assert xsc.N_avg == pytest.approx(n_d)

    def test_intensive_vs_extensive_scaling(self, ri):
        """Q/C/MEC are N_d-independent; B scales linearly with N_d."""
        r0_um = 1.2
        xa = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(r0_um, n=1.0),
                        wn=ri.wavenumber, backend='python')
        xb = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(r0_um, n=1.0e6),
                        wn=ri.wavenumber, backend='python')
        xa.compute(progress=False)
        xb.compute(progress=False)

        # Intensive — identical
        np.testing.assert_allclose(xa.Q_ext, xb.Q_ext, rtol=1e-12)
        np.testing.assert_allclose(xa.C_ext, xb.C_ext, rtol=1e-12)
        np.testing.assert_allclose(xa.MEC, xb.MEC, rtol=1e-12)
        # Extensive — scales with N_d
        np.testing.assert_allclose(xb.B_ext, 1.0e6 * xa.B_ext, rtol=1e-10)

    def test_phase_matrix_available(self, ri):
        """A monodisperse population yields a phase matrix like a distribution."""
        xsc = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(1.0, n=1e6),
                         wn=ri.wavenumber, backend='python')
        theta = np.linspace(0.0, 180.0, 19)
        xsc.compute(theta_pts=theta, n_legendre=8, progress=False)

        assert xsc.phase_matrix.shape == (len(ri.wavenumber), len(theta), 4)
        assert np.all(np.isfinite(xsc.phase_matrix))
        assert xsc.theta.shape == (len(theta),)


class TestNarrowDistributionLimit:
    """A very narrow distribution should approach the monodisperse result."""

    def test_lognormal_converges(self, ri):
        r0_um, n_d = 1.0, 1.0e6
        mono = AerosolXSC(ri=ri, dist=MonodisperseDist.from_um(r0_um, n=n_d),
                          wn=ri.wavenumber, backend='python')
        mono.compute(progress=False)

        narrow = AerosolXSC(
            ri=ri, dist=LogNormalDist.from_um(r0_um, 1.02, n=n_d),
            wn=ri.wavenumber, backend='python', n_pts=400,
        )
        narrow.compute(progress=False)

        # Within a few percent for a 2 %-spread log-normal
        np.testing.assert_allclose(narrow.MEC, mono.MEC, rtol=5e-2)
