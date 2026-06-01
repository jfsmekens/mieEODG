"""
Mie scattering parameters for a single spherical particle.

Ported from:
    IDLsource/mie_single.pro   (G. Thomas, Sep 2003; multiple revisions to 2015)
    IDLsource/dlm-code/mieint.f (R. Grainger, 1990; G. Thomas, 2005-2012;
                                  G. McGarragh, 2015)

Reference:
    Bohren, C.F. and Huffman, D.R., "Absorption and Scattering of Light by
    Small Particles", Wiley-VCH, 1983.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MieSingleResult:
    """
    Output of :func:`mieSingle`.

    Attributes
    ----------
    Qext : np.ndarray, shape (N_sizes,)
        Extinction efficiency.
    Qsca : np.ndarray, shape (N_sizes,)
        Scattering efficiency.
    Qbk : np.ndarray, shape (N_sizes,)
        Backscattering efficiency (differs from the standard definition by
        4pi -- see Bohren & Huffman Sec. 4.6).
    g : np.ndarray, shape (N_sizes,)
        Asymmetry parameter.
    S1 : np.ndarray or None, shape (N_angles, N_sizes), complex
        First amplitude function (perpendicular polarisation).
        None if no scattering angles were requested.
    S2 : np.ndarray or None, shape (N_angles, N_sizes), complex
        Second amplitude function (parallel polarisation).
        None if no scattering angles were requested.
    SPM : np.ndarray or None, shape (4, N_angles, N_sizes)
        Scattering phase matrix elements [F11, F33, F12, F34], normalized
        to the standard radiative transfer convention:
        ``(1/4pi) * integral(F11, dOmega) = 1``,
        equivalently ``(1/2) * integral(F11, d(cosθ)) = 1``.
        None if no scattering angles were requested.
    """
    Qext: np.ndarray
    Qsca: np.ndarray
    Qbk:  np.ndarray
    g:    np.ndarray
    S1:   np.ndarray | None = None
    S2:   np.ndarray | None = None
    SPM:  np.ndarray | None = None


def _nstop(x: float) -> int:
    """
    Number of Mie series terms required for convergence (Wiscombe 1980).

    Parameters
    ----------
    x : float
        Size parameter.

    Returns
    -------
    int

    Ref: mie_single.pro lines 191-199
    """
    if x < 0.02:
        return 2
    if x <= 8.0:
        return int(x + 4.00 * x**(1.0 / 3.0) + 2.0)
    if x < 4200.0:
        return int(x + 4.05 * x**(1.0 / 3.0) + 2.0)
    return int(x + 4.00 * x**(1.0 / 3.0) + 2.0)


def _nstop_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized NStop for an array of size parameters.

    Parameters
    ----------
    x : np.ndarray
        Size parameters.

    Returns
    -------
    np.ndarray, dtype int64
        NStop for each element of x.
    """
    x      = np.asarray(x, dtype=np.float64)
    cbrt_x = x ** (1.0 / 3.0)
    ns = np.where(
        x < 0.02,
        2,
        np.where(
            x <= 8.0,
            (x + 4.00 * cbrt_x + 2.0).astype(np.int64),
            np.where(
                x < 4200.0,
                (x + 4.05 * cbrt_x + 2.0).astype(np.int64),
                (x + 4.00 * cbrt_x + 2.0).astype(np.int64),
            ),
        ),
    )
    return ns.astype(np.int64)


def _mie_kernel(
    x: float,
    m: complex,
    mu: np.ndarray | None,
) -> tuple:
    """
    Core Mie recurrence for a single (x, m) pair.

    This is the reference scalar implementation, retained for documentation
    and validation purposes.  The production path uses :func:`_mie_kernel_batch`.

    Parameters
    ----------
    x : float
        Size parameter (2 pi r nu).
    m : complex
        Complex refractive index.
    mu : np.ndarray or None
        Cosines of scattering angles. If None only efficiencies are computed.

    Returns
    -------
    Qext, Qsca, Qbk, g : float
    S1, S2 : np.ndarray (complex128) or None, shape (N_angles,)
    SPM : np.ndarray or None, shape (4, N_angles)

    Ref: mie_single.pro lines 184-285; mieint.f lines 104-210
    """
    # --- Logarithmic derivative D_n(mx) by downward recurrence ------------
    # Ref: mie_single.pro lines 188-207; mieint.f lines 101-127
    y      = x * m
    n_stop = _nstop(x)
    nmx    = int(max(n_stop, abs(y))) + 15

    D = np.zeros(nmx + 1, dtype=np.complex128)
    for n in range(nmx - 1, 0, -1):
        A1   = (n + 1.0) / y
        D[n] = A1 - 1.0 / (A1 + D[n + 1])

    # --- Angular function setup --------------------------------------------
    if mu is not None:
        if np.any(mu == -1.0):
            mu2 = mu
        else:
            mu2 = np.append(mu, -1.0)
        n_mu2 = len(mu2)
    else:
        mu2   = np.array([-1.0])
        n_mu2 = 1

    Sp  = np.zeros(n_mu2, dtype=np.complex128)
    Sm  = np.zeros(n_mu2, dtype=np.complex128)
    pi0 = np.zeros(n_mu2, dtype=np.complex128)
    pi1 = np.ones(n_mu2,  dtype=np.complex128)

    # --- Riccati-Bessel initialisations ------------------------------------
    # Ref: mie_single.pro lines 214-220; mieint.f lines 134-141
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 =  np.cos(x)
    xi1  = complex(psi1, chi1)

    g_acc  = 0.0
    Qsca   = 0.0
    Qext   = 0.0
    Qbk    = 0.0 + 0j
    tnp1   = 1.0
    ir     = 1.0 / m
    anm1   = 0.0 + 0j
    bnm1   = 0.0 + 0j

    # --- Main Mie series loop ----------------------------------------------
    # Ref: mie_single.pro lines 227-263; mieint.f lines 147-190
    for n in range(1, n_stop + 1):
        dn   = float(n)
        tnp1 = tnp1 + 2.0
        tnm1 = tnp1 - 2.0
        rnx  = dn / x

        psi  = tnm1 * psi1 / x - psi0
        chi  = tnm1 * chi1 / x - chi0
        xi   = complex(psi, chi)

        a = ((D[n] * ir + rnx) * psi - psi1) / ((D[n] * ir + rnx) * xi - xi1)
        b = ((D[n] * m  + rnx) * psi - psi1) / ((D[n] * m  + rnx) * xi - xi1)

        Qext += tnp1 * (a + b).real
        Qsca += tnp1 * (a * a.conjugate() + b * b.conjugate()).real
        Qbk  += tnp1 * ((-1) ** n) * (a - b)

        if n > 1:
            g_acc += (
                (dn * dn - 1.0) *
                (anm1 * a.conjugate() + bnm1 * b.conjugate()).real / dn
                + tnm1 * (anm1 * bnm1.conjugate()).real / (dn * dn - dn)
            )
        anm1 = a
        bnm1 = b

        S    = mu2 * pi1
        T    = S - pi0
        tau  = dn * T - pi0
        A2   = tnp1 / (dn * (dn + 1.0))
        Sp  += A2 * (a + b) * (pi1 + tau)
        Sm  += A2 * (a - b) * (pi1 - tau)

        pi0  = pi1.copy()
        pi1  = S + T * (dn + 1.0) / dn

        psi0 = psi1;  psi1 = psi
        chi0 = chi1;  chi1 = chi
        xi1  = complex(psi1, chi1)

    # --- Normalise efficiencies --------------------------------------------
    x2   = x * x
    Qsca = 2.0 * Qsca / x2
    Qext = 2.0 * Qext / x2
    Qbk  = (Qbk * Qbk.conjugate()).real / x2

    if g_acc > 0.0:
        g = 2.0 * g_acc / (Qsca * x2 / 2.0)
    else:
        g = 0.0

    # --- Amplitude functions and phase matrix ------------------------------
    if mu is not None:
        n_mu = len(mu)
        S1 = (Sp[:n_mu] + Sm[:n_mu]) / 2.0
        S2 = (Sp[:n_mu] - Sm[:n_mu]) / 2.0

        AA  = 2.0 / (x2 * Qsca)
        SPM = np.empty((4, n_mu), dtype=np.float64)
        c1 = S1 * S1.conjugate()
        c2 = S1 * S2.conjugate()
        c3 = S2 * S2.conjugate()
        c4 = S2 * S1.conjugate()
        SPM[0] =  AA * (c1 + c3).real
        SPM[1] =  AA * (c2 + c4).real
        SPM[2] = -AA * (c1 - c3).real
        SPM[3] = -AA * ((c2 - c4) * 1j).real
    else:
        S1  = None
        S2  = None
        SPM = None

    return Qext, Qsca, float(Qbk), g, S1, S2, SPM


def _mie_kernel_batch(
    x_arr: np.ndarray,
    m: complex,
    mu: np.ndarray | None,
) -> tuple:
    """
    Vectorized Mie recurrence for an array of size parameters at a single
    complex refractive index.

    Replaces the per-size loop in :func:`mieSingle` with NumPy array
    operations: the Mie series loop runs ``NStop`` Python iterations, but
    each iteration operates on all ``N`` size parameters simultaneously.
    This eliminates the innermost Python loop over sizes that dominated
    runtime in the previous scalar implementation.

    Parameters
    ----------
    x_arr : np.ndarray, shape (N,)
        Size parameters.
    m : complex
        Complex refractive index (scalar — identical for all sizes in x_arr).
    mu : np.ndarray or None
        Cosines of scattering angles.  If None only efficiencies are returned.

    Returns
    -------
    Qext, Qsca, Qbk, g : np.ndarray, shape (N,)
    S1, S2 : np.ndarray, shape (N_angles, N), complex — or None
    SPM : np.ndarray, shape (4, N_angles, N) — or None

    Notes
    -----
    NStop is determined by the largest size parameter in x_arr, so all
    elements are computed to the same order.  Elements where the series has
    already converged contribute negligibly for extra terms, so this
    does not introduce error.

    Memory: the logarithmic-derivative array D has shape (global_nmx+1, N).
    For N=1186 quadrature points (r0=10 µm) and global_nmx≈240 this is
    ~4.5 MB — well within normal limits.

    Ref: mie_single.pro lines 184-285; mieint.f lines 104-210
    """
    N = len(x_arr)
    m  = complex(m)
    ir = 1.0 / m

    # --- Logarithmic derivative D_n(y) by downward recurrence ---------------
    # y_arr shape (N,); D shape (global_nmx+1, N)
    # Ref: mie_single.pro lines 188-207; mieint.f lines 101-127
    y_arr     = x_arr * m                                          # (N,) complex
    nstop_arr = _nstop_vec(x_arr)                                  # (N,) int
    global_nstop = int(nstop_arr.max())
    nmx_arr   = np.maximum(nstop_arr,
                           np.ceil(np.abs(y_arr)).astype(np.int64)) + 15
    global_nmx = int(nmx_arr.max())

    D = np.zeros((global_nmx + 1, N), dtype=np.complex128)
    for n in range(global_nmx - 1, 0, -1):
        A1   = (n + 1.0) / y_arr       # (N,) complex
        D[n] = A1 - 1.0 / (A1 + D[n + 1])

    # --- Angular setup -------------------------------------------------------
    # Append mu=-1 for backscatter if not already present.
    if mu is not None:
        mu2 = mu if np.any(mu == -1.0) else np.append(mu, -1.0)
    else:
        mu2 = np.array([-1.0])
    n_mu2 = len(mu2)

    # Phase function accumulators: (n_mu2, N) complex
    # Angular recurrence arrays: (n_mu2, N) real (pi0, pi1 are real-valued)
    Sp  = np.zeros((n_mu2, N), dtype=np.complex128)
    Sm  = np.zeros((n_mu2, N), dtype=np.complex128)
    pi0 = np.zeros((n_mu2, N))
    pi1 = np.ones( (n_mu2, N))

    # --- Riccati-Bessel initialisations: shape (N,) --------------------------
    # Ref: mie_single.pro lines 214-220; mieint.f lines 134-141
    psi0 =  np.cos(x_arr)
    psi1 =  np.sin(x_arr)
    chi0 = -np.sin(x_arr)
    chi1 =  np.cos(x_arr)
    xi1  = psi1 + 1j * chi1

    # Efficiency accumulators: (N,)
    Qext  = np.zeros(N)
    Qsca  = np.zeros(N)
    Qbk   = np.zeros(N, dtype=np.complex128)
    g_acc = np.zeros(N)
    tnp1  = 1.0
    anm1  = np.zeros(N, dtype=np.complex128)
    bnm1  = np.zeros(N, dtype=np.complex128)

    # --- Main Mie series loop: one Python iteration per order n -------------
    # Each iteration processes all N sizes simultaneously with NumPy ops.
    # Ref: mie_single.pro lines 227-263; mieint.f lines 147-190
    for n in range(1, global_nstop + 1):
        dn   = float(n)
        tnp1 += 2.0
        tnm1  = tnp1 - 2.0
        rnx   = dn / x_arr                         # (N,)

        psi  = tnm1 * psi1 / x_arr - psi0          # (N,)
        chi  = tnm1 * chi1 / x_arr - chi0          # (N,)
        xi   = psi + 1j * chi                       # (N,) complex

        # Mie coefficients a_n, b_n
        # Ref: mie_single.pro lines 237-238; mieint.f lines 158-159
        # D[n] shape (N,); ir, m scalar
        a = ((D[n] * ir + rnx) * psi - psi1) / ((D[n] * ir + rnx) * xi - xi1)
        b = ((D[n] * m  + rnx) * psi - psi1) / ((D[n] * m  + rnx) * xi - xi1)

        # Efficiency accumulators
        Qext += tnp1 * (a + b).real
        Qsca += tnp1 * (a * a.conjugate() + b * b.conjugate()).real
        Qbk  += tnp1 * ((-1) ** n) * (a - b)

        # Asymmetry parameter accumulator
        if n > 1:
            g_acc += (
                (dn * dn - 1.0) *
                (anm1 * a.conjugate() + bnm1 * b.conjugate()).real / dn
                + tnm1 * (anm1 * bnm1.conjugate()).real / (dn * dn - dn)
            )
        anm1 = a.copy()
        bnm1 = b.copy()

        # Angular recurrence: mu2 (n_mu2,), pi1 (n_mu2, N)
        # Ref: mie_single.pro lines 247-255; mieint.f lines 171-183
        S    = mu2[:, None] * pi1                   # (n_mu2, N)
        T    = S - pi0                              # (n_mu2, N)
        tau  = dn * T - pi0                         # (n_mu2, N)
        A2   = tnp1 / (dn * (dn + 1.0))
        Sp  += A2 * (a + b)[None, :] * (pi1 + tau)
        Sm  += A2 * (a - b)[None, :] * (pi1 - tau)

        pi0[:] = pi1
        pi1    = S + T * (dn + 1.0) / dn

        psi0 = psi1.copy();  psi1 = psi.copy()
        chi0 = chi1.copy();  chi1 = chi.copy()
        xi1  = psi1 + 1j * chi1

    # --- Normalise efficiencies ----------------------------------------------
    # Ref: mie_single.pro lines 287-290; mieint.f lines 191-195
    x2   = x_arr ** 2
    Qsca = 2.0 * Qsca / x2
    Qext = 2.0 * Qext / x2
    Qbk  = (Qbk * Qbk.conjugate()).real / x2
    # g = 2*g_acc / Qsca_raw, where Qsca_raw = Qsca * x2 / 2
    g = np.where(
        g_acc > 0.0,
        2.0 * g_acc / (np.where(Qsca > 0.0, Qsca, 1.0) * x2 / 2.0),
        0.0,
    )

    # --- Amplitude functions and phase matrix --------------------------------
    # RT normalisation: (1/2)*integral(F11, d(cos theta)) = 1
    # Ref: srfm optical_properties.py::mie_ewp, AA definition
    if mu is not None:
        n_mu = len(mu)
        S1 = (Sp[:n_mu] + Sm[:n_mu]) / 2.0         # (n_mu, N) complex
        S2 = (Sp[:n_mu] - Sm[:n_mu]) / 2.0         # (n_mu, N) complex

        AA  = np.where(Qsca > 0.0, 2.0 / (x2 * Qsca), 0.0)   # (N,)
        c1  = (S1 * S1.conjugate()).real             # (n_mu, N)
        c2  =  S1 * S2.conjugate()                  # (n_mu, N) complex
        c3  = (S2 * S2.conjugate()).real             # (n_mu, N)
        c4  =  S2 * S1.conjugate()                  # (n_mu, N) complex
        SPM = np.empty((4, n_mu, N))
        SPM[0] =  AA[None, :] * (c1 + c3)
        SPM[1] =  AA[None, :] * (c2 + c4).real
        SPM[2] = -AA[None, :] * (c1 - c3)
        SPM[3] = -AA[None, :] * ((c2 - c4) * 1j).real
    else:
        S1 = S2 = SPM = None

    return Qext, Qsca, Qbk.real, g, S1, S2, SPM


def mieSingle(
    Dx: np.ndarray | float,
    Cm: complex | np.ndarray,
    angles: np.ndarray | None = None,
    backend: str = 'python',
    silent: bool = False,
) -> MieSingleResult:
    """
    Mie scattering parameters for an array of spherical particles.

    Parameters
    ----------
    Dx : array-like, shape (N_sizes,) or scalar
        Size parameter(s) x = 2 pi r nu.
    Cm : complex scalar or array-like, shape (N_nu,)
        Complex refractive index.  Convention: imaginary part should be
        **negative** for absorbing particles.  A positive imaginary part
        triggers a warning unless ``silent=True``.
        If ``Cm`` is an array of length N_nu, the result arrays will have a
        leading N_nu axis: shape (N_nu, N_sizes).
    angles : array-like or None
        Cosines of scattering angles at which to compute S1, S2, and the
        phase matrix.  If None, only efficiencies are returned.
    backend : {'python', 'fortran'}
        Computational backend.  ``'fortran'`` requires the optional f2py
        extension; falls back to ``'python'`` with a warning if unavailable.
        Default ``'python'``.
    silent : bool
        If True, suppress the positive-imaginary-part warning.

    Returns
    -------
    MieSingleResult
        Named result with fields Qext, Qsca, Qbk, g, and optionally
        S1, S2, SPM.

    Notes
    -----
    **Performance:** the Python backend uses :func:`_mie_kernel_batch`, which
    vectorizes over all N_sizes at each Mie order n.  The main loop runs
    NStop Python iterations (NStop ≈ x_max + 4·x_max^(1/3) + 2), with each
    iteration doing NumPy array ops on N_sizes elements.  This is typically
    100–1000× faster than the old per-size scalar kernel.

    The backscattering efficiency Qbk differs from the standard definition
    by a factor of 4 pi (Bohren & Huffman 1983, Sec. 4.6).

    Ported from IDLsource/mie_single.pro and IDLsource/dlm-code/mieint.f.
    """
    if backend == 'fortran':
        logger.warning(
            "Fortran backend not yet available; falling back to Python."
        )

    Dx = np.atleast_1d(np.asarray(Dx, dtype=np.float64))
    Cm = np.asarray(Cm, dtype=np.complex128)
    scalar_cm = (Cm.ndim == 0)
    Cm = np.atleast_1d(Cm)

    if not silent and np.any(Cm.imag > 0.0):
        logger.warning(
            "Imaginary part of refractive index is positive. Convention: "
            "imaginary part should be negative for absorbing particles. "
            "Pass silent=True to suppress this warning."
        )

    mu = None if angles is None else np.asarray(angles, dtype=np.float64)

    if scalar_cm:
        # --- Single refractive index, array of sizes -------------------------
        # One _mie_kernel_batch call handles all sizes in a single NStop loop.
        m = complex(Cm[0])
        Qext, Qsca, Qbk, g, S1, S2, SPM = _mie_kernel_batch(Dx, m, mu)
        # Outputs: Qext/Qsca/Qbk/g shape (N_sizes,); SPM shape (4, N_mu, N_sizes)

    else:
        # --- Array of refractive indices: (N_nu,) × (N_sizes,) --------------
        # One _mie_kernel_batch call per m value; result is (N_nu, N_sizes).
        n_nu    = len(Cm)
        n_sizes = len(Dx)
        Qext = np.empty((n_nu, n_sizes))
        Qsca = np.empty((n_nu, n_sizes))
        Qbk  = np.empty((n_nu, n_sizes))
        g    = np.empty((n_nu, n_sizes))
        if mu is not None:
            n_mu = len(mu)
            S1  = np.empty((n_nu, n_mu, n_sizes), dtype=np.complex128)
            S2  = np.empty((n_nu, n_mu, n_sizes), dtype=np.complex128)
            SPM = np.empty((n_nu, 4, n_mu, n_sizes))
        else:
            S1 = S2 = SPM = None

        for j, m in enumerate(Cm):
            qe, qs, qb, gi, s1j, s2j, spmj = _mie_kernel_batch(
                Dx, complex(m), mu
            )
            Qext[j] = qe
            Qsca[j] = qs
            Qbk[j]  = qb
            g[j]    = gi
            if mu is not None:
                S1[j]  = s1j
                S2[j]  = s2j
                SPM[j] = spmj

    return MieSingleResult(Qext=Qext, Qsca=Qsca, Qbk=Qbk, g=g,
                           S1=S1, S2=S2, SPM=SPM)
