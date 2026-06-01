"""
Analytical derivatives of Mie scattering parameters.

Ported from:
    IDLsource/mie_derivs.pro  (J. Lucas 2002; R. Grainger 2002;
                                G. Thomas Sep 2003; J. Graham Feb 2004;
                                A. Smith Aug 2010, Apr 2013)

Reference:
    Grainger, R.G., J. Lucas, G.E. Thomas, G. Ewan,
    "The Calculation of Mie Derivatives", Appl. Opt., 2004.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MieDerivResult:
    """
    Output of :func:`mieDerivs`.

    All arrays have shape ``(N_sizes,)`` unless noted otherwise.

    Attributes
    ----------
    Qext : np.ndarray
        Extinction efficiency.
    Qsca : np.ndarray
        Scattering efficiency.
    g : np.ndarray
        Asymmetry parameter.
    dQextdx : np.ndarray
        d Qext / d x  (size parameter).
    dQextdRem : np.ndarray
        d Qext / d Re(m).
    dQextdImm : np.ndarray
        d Qext / d Im(m).
    dQscadx : np.ndarray
        d Qsca / d x.
    dQscadRem : np.ndarray
        d Qsca / d Re(m).
    dQscadImm : np.ndarray
        d Qsca / d Im(m).
    dASYMdx : np.ndarray
        d g / d x.
    dASYMdRem : np.ndarray
        d g / d Re(m).
    dASYMdImm : np.ndarray
        d g / d Im(m).
    i1 : np.ndarray or None, shape (N_angles, N_sizes)
        First intensity function  |S1|².
    i2 : np.ndarray or None, shape (N_angles, N_sizes)
        Second intensity function |S2|².
    di1dx : np.ndarray or None, shape (N_angles, N_sizes)
        d i1 / d x.
    di2dx : np.ndarray or None, shape (N_angles, N_sizes)
        d i2 / d x.
    di1dRem : np.ndarray or None, shape (N_angles, N_sizes)
        d i1 / d Re(m).
    di1dImm : np.ndarray or None, shape (N_angles, N_sizes)
        d i1 / d Im(m).
    di2dRem : np.ndarray or None, shape (N_angles, N_sizes)
        d i2 / d Re(m).
    di2dImm : np.ndarray or None, shape (N_angles, N_sizes)
        d i2 / d Im(m).
    """
    Qext:     np.ndarray
    Qsca:     np.ndarray
    g:        np.ndarray
    dQextdx:  np.ndarray
    dQextdRem: np.ndarray
    dQextdImm: np.ndarray
    dQscadx:  np.ndarray
    dQscadRem: np.ndarray
    dQscadImm: np.ndarray
    dASYMdx:  np.ndarray
    dASYMdRem: np.ndarray
    dASYMdImm: np.ndarray
    i1:       np.ndarray | None = None
    i2:       np.ndarray | None = None
    di1dx:    np.ndarray | None = None
    di2dx:    np.ndarray | None = None
    di1dRem:  np.ndarray | None = None
    di1dImm:  np.ndarray | None = None
    di2dRem:  np.ndarray | None = None
    di2dImm:  np.ndarray | None = None


def _nstop_derivs(x: float) -> int:
    """
    NStop for mie_derivs (slightly different constants from mie_single).

    Ref: mie_derivs.pro lines 76-83
    """
    if x < 0.02:
        return 2
    if x <= 8.0:
        return int(x + 4.00 * x ** (1.0 / 3.0) + 1.0)
    if x < 4200.0:
        return int(x + 4.05 * x ** (1.0 / 3.0) + 2.0)
    return int(x + 4.00 * x ** (1.0 / 3.0) + 2.0)


def _mie_derivs_kernel(
    x: float,
    m: complex,
    mu: np.ndarray | None,
) -> tuple:
    """
    Analytical derivatives of Mie efficiencies and intensity functions
    for a single (x, m) pair.

    Parameters
    ----------
    x : float
        Size parameter.
    m : complex
        Complex refractive index.
    mu : np.ndarray or None
        Cosines of scattering angles.  If None, intensity outputs are skipped.

    Returns
    -------
    tuple of (Qext, Qsca, g,
              dQextdx, dQextdRem, dQextdImm,
              dQscadx, dQscadRem, dQscadImm,
              dASYMdx, dASYMdRem, dASYMdImm,
              i1, i2, di1dx, di2dx, di1dRem, di1dImm, di2dRem, di2dImm)
    where intensity arrays are None when mu is None.

    Ref: mie_derivs.pro lines 74-213
    """
    y      = m * x
    n_stop = _nstop_derivs(x)
    nmx    = int(max(n_stop, abs(y))) + 15

    # Logarithmic derivative downward recurrence
    # Ref: mie_derivs.pro lines 86-90
    D = np.zeros(nmx + 1, dtype=np.complex128)
    for n in range(nmx - 1, 0, -1):
        R    = (n + 1.0) / y
        D[n] = R - 1.0 / (R + D[n + 1])

    # Riccati-Bessel initialisation
    # Ref: mie_derivs.pro lines 92-95
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 =  np.cos(x)

    # Efficiency and derivative accumulators
    Qext_acc      = 0.0
    Qsca_acc      = 0.0
    dQextdx1_acc  = 0.0   # Σ(2n+1) Re(da/dx + db/dx)
    dQextdx2_acc  = 0.0   # Σ(2n+1) Re(a + b)  [= Qext_acc]
    dQextdRem_acc = 0.0
    dQextdImm_acc = 0.0
    dQscadx1_acc  = 0.0   # Σ(2n+1)(Re(a)Re(da/dx) + Im(a)Im(da/dx) + ...)
    dQscadx2_acc  = 0.0   # Σ(2n+1)(|a|^2 + |b|^2)  [= Qsca_acc]
    dQscadRem_acc = 0.0
    dQscadImm_acc = 0.0

    # Asymmetry parameter and derivative accumulators (run one term behind)
    ASYM_acc = 0.0
    dgdx_acc = 0.0
    dgdRem_acc = 0.0
    dgdImm_acc = 0.0

    # Angular function accumulators (vectorised over mu if provided)
    n_mu = len(mu) if mu is not None else 0
    calc_angles = (mu is not None) and (n_mu > 0)

    if calc_angles:
        Sp      = np.zeros(n_mu, dtype=np.complex128)
        Sm      = np.zeros(n_mu, dtype=np.complex128)
        dSpdx   = np.zeros(n_mu, dtype=np.complex128)
        dSmdx   = np.zeros(n_mu, dtype=np.complex128)
        dSpdRem = np.zeros(n_mu, dtype=np.complex128)
        dSpdImm = np.zeros(n_mu, dtype=np.complex128)
        dSmdRem = np.zeros(n_mu, dtype=np.complex128)
        dSmdImm = np.zeros(n_mu, dtype=np.complex128)
        Pi0     = np.zeros(n_mu, dtype=np.float64)
        Pi1     = np.ones(n_mu,  dtype=np.float64)

    anm1 = bnm1 = dadxm1 = dbdxm1 = dadnm1 = dbdnm1 = dadkm1 = dbdkm1 = 0.0 + 0j

    # Main loop
    # Ref: mie_derivs.pro lines 113-194
    for n in range(1, n_stop + 1):
        dn  = float(n)
        rnx = dn / x
        ry  = dn / y

        # Riccati-Bessel recurrence
        # Ref: mie_derivs.pro lines 115-116
        psi  = (2.0 * dn - 1.0) * psi1 / x - psi0
        chi  = (2.0 * dn - 1.0) * chi1 / x - chi0
        zeta    = complex(psi,  chi)
        zeta_m1 = complex(psi1, chi1)

        # Mie coefficients a_n, b_n
        # Ref: mie_derivs.pro lines 118-120
        denom_a = (D[n] / m + rnx) * zeta - zeta_m1
        denom_b = (D[n] * m + rnx) * zeta - zeta_m1
        a = ((D[n] / m + rnx) * psi - psi1) / denom_a
        b = ((D[n] * m + rnx) * psi - psi1) / denom_b

        # Derivatives of a_n and b_n
        # Ref: mie_derivs.pro lines 122-131
        i_unit = 1.0j
        Da2 = denom_a ** 2
        Db2 = denom_b ** 2
        mDa2 = (m * denom_a) ** 2   # denominator for dadRem/dadImm

        dadx   = i_unit * (D[n] ** 2 * (1.0 / m ** 2 - 1.0) +
                           dn * (dn + 1.0) * (1.0 / y ** 2 - 1.0 / x ** 2)) / Da2
        dbdx   = i_unit * (1.0 - m ** 2 +
                           dn * (dn + 1.0) * (m ** 2 / y ** 2 - 1.0 / x ** 2)) / Db2

        # Common numerator factor for m-derivatives
        # Ref: mie_derivs.pro lines 127-131
        num_a   = dn * (dn + 1.0) / y - y * (1.0 + D[n] ** 2) - D[n]
        num_b   = dn * (dn + 1.0) / y - y * (1.0 + D[n] ** 2) + D[n]
        dadRem  =  i_unit * num_a / mDa2
        dadImm  =          -num_a / mDa2
        dbdRem  =  i_unit * num_b / Db2
        dbdImm  =          -num_b / Db2

        # Efficiency accumulation
        # Ref: mie_derivs.pro lines 133-152
        w       = 2.0 * dn + 1.0
        Qext_acc      += w * (a + b).real
        Qsca_acc      += w * (a * a.conjugate() + b * b.conjugate()).real
        dQextdx1_acc  += w * (dadx + dbdx).real
        dQextdx2_acc  += w * (a    + b).real             # same as Qext_acc
        dQextdRem_acc += w * (dadRem + dbdRem).real
        dQextdImm_acc += w * (dadImm + dbdImm).real
        dQscadx1_acc  += w * (a.real * dadx.real  + a.imag * dadx.imag  +
                              b.real * dbdx.real  + b.imag * dbdx.imag)
        dQscadx2_acc  += w * (a * a.conjugate() + b * b.conjugate()).real  # = Qsca_acc
        dQscadRem_acc += w * (a.real * dadRem.real + a.imag * dadRem.imag +
                              b.real * dbdRem.real + b.imag * dbdRem.imag)
        dQscadImm_acc += w * (a.real * dadImm.real + a.imag * dadImm.imag +
                              b.real * dbdImm.real + b.imag * dbdImm.imag)

        # Asymmetry parameter — running one term behind (n >= 2 uses a_{n-1})
        # Ref: mie_derivs.pro lines 154-184
        if n > 1:
            dn1  = dn - 1.0   # previous n value
            pre0 = dn1 * (2.0 + dn1) / (dn1 + 1.0)    # (n-1)(n+1)/n
            pre1 = (2.0 * dn1 + 1.0) / dn1 / (dn1 + 1.0)  # (2n-1)/(n(n-1))

            ASYM_acc += (pre0 * (anm1 * a.conjugate() + bnm1 * b.conjugate()).real
                         + pre1 * (anm1 * bnm1.conjugate()).real)

            dgdx_acc += (
                pre0 * (a.conjugate() * dadxm1 + anm1 * dadx.conjugate() +
                        b.conjugate() * dbdxm1 + bnm1 * dbdx.conjugate()).real
                + pre1 * (dadxm1 * bnm1.conjugate() +
                          anm1 * dbdxm1.conjugate()).real
            )
            dgdRem_acc += (
                pre0 * (a.conjugate() * dadnm1 + anm1 * dadRem.conjugate() +
                        b.conjugate() * dbdnm1 + bnm1 * dbdRem.conjugate()).real
                + pre1 * (dadnm1 * bnm1.conjugate() +
                          anm1 * dbdnm1.conjugate()).real
            )
            dgdImm_acc += (
                pre0 * (a.conjugate() * dadkm1 + anm1 * dadImm.conjugate() +
                        b.conjugate() * dbdkm1 + bnm1 * dbdImm.conjugate()).real
                + pre1 * (dadkm1 * bnm1.conjugate() +
                          anm1 * dbdkm1.conjugate()).real
            )

        anm1  = a;     bnm1  = b
        dadxm1 = dadx; dbdxm1 = dbdx
        dadnm1 = dadRem; dbdnm1 = dbdRem
        dadkm1 = dadImm; dbdkm1 = dbdImm

        # Angular functions (vectorised over mu)
        # Ref: mie_derivs.pro lines 186-194
        if calc_angles:
            S    = mu * Pi1
            T    = S - Pi0
            tau  = dn * T - Pi0
            coef = w / (dn * (dn + 1.0))
            Sp      += coef * (a    + b)    * (Pi1 + tau)
            Sm      += coef * (a    - b)    * (Pi1 - tau)
            dSpdx   += coef * (dadx + dbdx) * (Pi1 + tau)
            dSmdx   += coef * (dadx - dbdx) * (Pi1 - tau)
            dSpdRem += coef * (dadRem + dbdRem) * (Pi1 + tau)
            dSpdImm += coef * (dadImm + dbdImm) * (Pi1 + tau)
            dSmdRem += coef * (dadRem - dbdRem) * (Pi1 - tau)
            dSmdImm += coef * (dadImm - dbdImm) * (Pi1 - tau)
            Pi0 = Pi1.copy()
            Pi1 = S + T * (dn + 1.0) / dn

        psi0 = psi1;  psi1 = psi
        chi0 = chi1;  chi1 = chi

    # --- Normalise efficiencies and derivatives ---
    # Ref: mie_derivs.pro lines 196-216
    x2   = x * x
    x3   = x2 * x

    Qext = 2.0 * Qext_acc / x2
    Qsca = 2.0 * Qsca_acc / x2

    dQextdx  = 2.0 * dQextdx1_acc / x2 - 4.0 * dQextdx2_acc / x3
    dQextdRem = 2.0 * dQextdRem_acc / x2
    dQextdImm = 2.0 * dQextdImm_acc / x2

    dQscadx   = 4.0 * dQscadx1_acc / x2 - 4.0 * dQscadx2_acc / x3
    dQscadRem = 4.0 * dQscadRem_acc / x2
    dQscadImm = 4.0 * dQscadImm_acc / x2

    # Asymmetry parameter
    # Ref: mie_derivs.pro lines 218-224
    fxx  = 4.0 / x2
    if Qsca > 0.0:
        g         = fxx * ASYM_acc / Qsca
        dASYMdx   = (fxx * dgdx_acc   - g * (dQscadx   + 2.0 * Qsca / x)) / Qsca
        dASYMdRem = (fxx * dgdRem_acc  - g * dQscadRem) / Qsca
        dASYMdImm = (fxx * dgdImm_acc  - g * dQscadImm) / Qsca
    else:
        g = dASYMdx = dASYMdRem = dASYMdImm = 0.0

    # --- Intensity functions and their derivatives ---
    # Ref: mie_derivs.pro lines 196-215
    if calc_angles:
        S1     = (Sp + Sm) / 2.0
        S2     = (Sp - Sm) / 2.0
        dS1dx  = (dSpdx   + dSmdx)   / 2.0
        dS2dx  = (dSpdx   - dSmdx)   / 2.0
        dS1dRem = (dSpdRem + dSmdRem) / 2.0
        dS1dImm = (dSpdImm + dSmdImm) / 2.0
        dS2dRem = (dSpdRem - dSmdRem) / 2.0
        dS2dImm = (dSpdImm - dSmdImm) / 2.0

        i1     = (S1 * S1.conjugate()).real
        i2     = (S2 * S2.conjugate()).real
        di1dx  = 2.0 * (S1.real * dS1dx.real  + S1.imag * dS1dx.imag)
        di2dx  = 2.0 * (S2.real * dS2dx.real  + S2.imag * dS2dx.imag)
        di1dRem = 2.0 * (S1.real * dS1dRem.real + S1.imag * dS1dRem.imag)
        di1dImm = 2.0 * (S1.real * dS1dImm.real + S1.imag * dS1dImm.imag)
        di2dRem = 2.0 * (S2.real * dS2dRem.real + S2.imag * dS2dRem.imag)
        di2dImm = 2.0 * (S2.real * dS2dImm.real + S2.imag * dS2dImm.imag)
    else:
        i1 = i2 = di1dx = di2dx = di1dRem = di1dImm = di2dRem = di2dImm = None

    return (Qext, Qsca, g,
            dQextdx, dQextdRem, dQextdImm,
            dQscadx, dQscadRem, dQscadImm,
            dASYMdx, dASYMdRem, dASYMdImm,
            i1, i2, di1dx, di2dx, di1dRem, di1dImm, di2dRem, di2dImm)


def mieDerivs(
    Dx: np.ndarray | float,
    Cm: complex,
    angles: np.ndarray | None = None,
    silent: bool = False,
) -> MieDerivResult:
    """
    Analytical derivatives of Mie scattering parameters.

    Computes Mie efficiencies, the asymmetry parameter, and their analytical
    derivatives with respect to the size parameter and the real and imaginary
    parts of the refractive index.

    Parameters
    ----------
    Dx : array-like or float
        Size parameter(s) x = 2 π r ν.
    Cm : complex
        Complex refractive index.  Convention: imaginary part should be
        **negative** for absorbing particles.  A positive imaginary part
        triggers a warning unless ``silent=True``.
    angles : array-like or None
        Cosines of scattering angles.  If provided, the intensity functions
        i1 = |S1|², i2 = |S2|² and all their derivatives are also returned.
        If None, intensity outputs are None.
    silent : bool
        Suppress the positive-imaginary-part warning.

    Returns
    -------
    MieDerivResult
        Named result with efficiencies, asymmetry parameter, and all
        analytical derivatives.

    Notes
    -----
    The analytical derivative expressions follow Grainger et al. (2004).

    Unlike :func:`mieSingle`, this function does not accept an array of
    refractive indices — ``Cm`` must be a scalar.  Use the size-distribution
    wrapper :func:`mieSterivsLn` for spectral / distribution-integrated
    derivatives.

    Ported from IDLsource/mie_derivs.pro.
    """
    Dx = np.atleast_1d(np.asarray(Dx, dtype=np.float64))
    m  = complex(Cm)

    if not silent and m.imag > 0.0:
        logger.warning(
            "Imaginary part of refractive index is positive. Convention: "
            "imaginary part should be negative for absorbing particles. "
            "Pass silent=True to suppress this warning."
        )

    mu = None if angles is None else np.asarray(angles, dtype=np.float64)

    n_sizes = len(Dx)
    Qext_arr     = np.empty(n_sizes)
    Qsca_arr     = np.empty(n_sizes)
    g_arr        = np.empty(n_sizes)
    dQextdx_arr  = np.empty(n_sizes)
    dQextdRem_arr = np.empty(n_sizes)
    dQextdImm_arr = np.empty(n_sizes)
    dQscadx_arr  = np.empty(n_sizes)
    dQscadRem_arr = np.empty(n_sizes)
    dQscadImm_arr = np.empty(n_sizes)
    dASYMdx_arr  = np.empty(n_sizes)
    dASYMdRem_arr = np.empty(n_sizes)
    dASYMdImm_arr = np.empty(n_sizes)

    if mu is not None:
        n_mu = len(mu)
        i1_arr     = np.empty((n_mu, n_sizes))
        i2_arr     = np.empty((n_mu, n_sizes))
        di1dx_arr  = np.empty((n_mu, n_sizes))
        di2dx_arr  = np.empty((n_mu, n_sizes))
        di1dRem_arr = np.empty((n_mu, n_sizes))
        di1dImm_arr = np.empty((n_mu, n_sizes))
        di2dRem_arr = np.empty((n_mu, n_sizes))
        di2dImm_arr = np.empty((n_mu, n_sizes))
    else:
        i1_arr = i2_arr = di1dx_arr = di2dx_arr = None
        di1dRem_arr = di1dImm_arr = di2dRem_arr = di2dImm_arr = None

    for i, xi in enumerate(Dx):
        (qe, qs, gi,
         dqedx, dqedRem, dqedImm,
         dqsdx, dqsdRem, dqsdImm,
         dgdx,  dgdRem,  dgdImm,
         ii1, ii2, dii1dx, dii2dx,
         dii1dRem, dii1dImm, dii2dRem, dii2dImm) = _mie_derivs_kernel(
             float(xi), m, mu
         )
        Qext_arr[i]      = qe
        Qsca_arr[i]      = qs
        g_arr[i]         = gi
        dQextdx_arr[i]   = dqedx
        dQextdRem_arr[i] = dqedRem
        dQextdImm_arr[i] = dqedImm
        dQscadx_arr[i]   = dqsdx
        dQscadRem_arr[i] = dqsdRem
        dQscadImm_arr[i] = dqsdImm
        dASYMdx_arr[i]   = dgdx
        dASYMdRem_arr[i] = dgdRem
        dASYMdImm_arr[i] = dgdImm
        if mu is not None:
            i1_arr[:, i]      = ii1
            i2_arr[:, i]      = ii2
            di1dx_arr[:, i]   = dii1dx
            di2dx_arr[:, i]   = dii2dx
            di1dRem_arr[:, i] = dii1dRem
            di1dImm_arr[:, i] = dii1dImm
            di2dRem_arr[:, i] = dii2dRem
            di2dImm_arr[:, i] = dii2dImm

    return MieDerivResult(
        Qext=Qext_arr,     Qsca=Qsca_arr,     g=g_arr,
        dQextdx=dQextdx_arr, dQextdRem=dQextdRem_arr, dQextdImm=dQextdImm_arr,
        dQscadx=dQscadx_arr, dQscadRem=dQscadRem_arr, dQscadImm=dQscadImm_arr,
        dASYMdx=dASYMdx_arr, dASYMdRem=dASYMdRem_arr, dASYMdImm=dASYMdImm_arr,
        i1=i1_arr, i2=i2_arr,
        di1dx=di1dx_arr, di2dx=di2dx_arr,
        di1dRem=di1dRem_arr, di1dImm=di1dImm_arr,
        di2dRem=di2dRem_arr, di2dImm=di2dImm_arr,
    )
