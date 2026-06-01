"""
Legendre polynomial expansion and reconstruction utilities.

Ported from:
    IDLsource/legpexp.pro  (Don Grainger / EODG; fortran subroutine Alegpexp)
    IDLsource/legpcrt.pro  (Don Grainger / EODG)
"""

from __future__ import annotations

import numpy as np


def legendreExpand(
    phase: np.ndarray,
    qv: np.ndarray,
    qw: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """
    Expand a function sampled at Legendre quadrature points as a Legendre series.

    Parameters
    ----------
    phase : np.ndarray, shape (N_pts,)
        Function values at the quadrature points.
    qv : np.ndarray, shape (N_pts,)
        Legendre quadrature abscissae (cosines of angles on [-1, 1]).
    qw : np.ndarray, shape (N_pts,)
        Legendre quadrature weights (sum = 2).
    tol : float
        Convergence threshold.  Expansion stops when ``|lc[n]| < tol``.
        Default 1e-9.

    Returns
    -------
    lc : np.ndarray, shape (N_pts,)
        Legendre coefficients, zero-padded beyond ``n_lc``.
    n_lc : int
        Number of significant coefficients (exclusive upper index):
        ``lc[0:n_lc]`` are the meaningful terms.

    Notes
    -----
    The coefficients are defined by the orthogonality relation

    .. math::

        lc_n = \\frac{2n+1}{2} \\int_{-1}^{1} f(x) P_n(x) \\,dx

    so that  :math:`f(x) = \\sum_{n=0}^{N} lc_n P_n(x)`.

    The function must be evaluated at the quadrature points used to compute
    the expansion (typically Gauss-Legendre points from :func:`quadrature`).

    Ported from IDLsource/legpexp.pro.
    """
    phase = np.asarray(phase, dtype=np.float64)
    qv    = np.asarray(qv,    dtype=np.float64)
    qw    = np.asarray(qw,    dtype=np.float64)
    n_pts = len(phase)

    lc = np.zeros(n_pts)

    # Ref: legpexp.pro lines 26-27
    lc[0] = 0.5  * np.sum(phase * qw)
    lc[1] = 1.5  * np.sum(phase * qv * qw)

    if n_pts <= 2:
        return lc, min(2, n_pts)

    P_nm2 = np.ones(n_pts)
    P_nm1 = qv.copy()
    n_lc  = 2

    # Ref: legpexp.pro lines 29-42
    for n in range(2, n_pts):
        dn    = float(n)
        P_n   = ((2.0 * dn - 1.0) / dn) * qv * P_nm1 - ((dn - 1.0) / dn) * P_nm2
        P_nm2 = P_nm1
        P_nm1 = P_n

        lc[n] = (2.0 * dn + 1.0) * np.sum(phase * P_n * qw) / 2.0
        if abs(lc[n]) < tol:
            # lc[n] is below threshold; n_lc = n (terms 0..n-1 are significant)
            n_lc = n
            break
        n_lc = n + 1

    return lc, n_lc


def legendreReconstruct(
    lc: np.ndarray,
    n_lc: int,
    qv: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a function from its Legendre series coefficients.

    Parameters
    ----------
    lc : np.ndarray, shape (N_coef,)
        Legendre coefficients as returned by :func:`legendreExpand`.
    n_lc : int
        Number of terms to use in the reconstruction.
        Typically the second return value of :func:`legendreExpand`.
    qv : np.ndarray, shape (N_pts,)
        Evaluation points (cosines of angles on [-1, 1]).

    Returns
    -------
    np.ndarray, shape (N_pts,)
        Reconstructed function values.

    Notes
    -----
    Computes :math:`f(x) = \\sum_{n=0}^{n_{lc}-1} lc_n P_n(x)`.

    Ported from IDLsource/legpcrt.pro.
    """
    qv  = np.asarray(qv, dtype=np.float64)
    lc  = np.asarray(lc, dtype=np.float64)
    n   = len(qv)

    phase = np.full(n, lc[0])

    if n_lc < 2:
        return phase

    # l = 1
    phase += lc[1] * qv

    # Ref: legpcrt.pro lines 15-23
    P_nm2 = np.ones(n)
    P_nm1 = qv.copy()

    for l in range(2, n_lc):
        dl   = float(l)
        P_l  = ((2.0 * dl - 1.0) / dl) * qv * P_nm1 - ((dl - 1.0) / dl) * P_nm2
        P_nm2 = P_nm1
        P_nm1 = P_l
        phase += lc[l] * P_l

    return phase
