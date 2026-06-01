"""
Generalised spherical functions and scattering phase matrix expansion.

Ported from:
    IDLsource/gen_sph_funcs.pro      (G. McGarragh, 28 Jul 2015;
                                      based on Mischenko 1991; from LMie)
    IDLsource/scat_phase_matrix_exp.pro (G. McGarragh, 28 Jul 2015; from LMie)

Reference:
    Mishchenko, M.I., "Light scattering by randomly oriented axially
    symmetric particles", J. Opt. Soc. Am. A, 8, 871–882, 1991.
"""

from __future__ import annotations

import numpy as np


def genSphFuncs(
    n_l: int,
    mu: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the generalised spherical functions P^l_{m,m'} required for
    expanding the single-scattering phase matrix.

    Parameters
    ----------
    n_l : int
        Maximum order (exclusive), i.e. functions are computed for
        l = 0, 1, ..., n_l-1.
    mu : array-like or float
        Cosines of scattering angles.

    Returns
    -------
    p00 : np.ndarray, shape (n_mu, n_l)
        Generalised spherical function P^l_{0,0} = Legendre polynomial P_l.
    p0p2 : np.ndarray, shape (n_mu, n_l)
        P^l_{0,2}.
    p2p2 : np.ndarray, shape (n_mu, n_l)
        P^l_{2,2}.
    p2m2 : np.ndarray, shape (n_mu, n_l)
        P^l_{2,-2}.

    Notes
    -----
    Only defined for l >= 2 for the off-diagonal functions; orders 0 and 1
    of p0p2, p2p2, p2m2 are zero.

    Ported from IDLsource/gen_sph_funcs.pro (Mischenko 1991).
    """
    mu   = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    n_mu = len(mu)

    p00  = np.zeros((n_mu, n_l))
    p0p2 = np.zeros((n_mu, n_l))
    p2p2 = np.zeros((n_mu, n_l))
    p2m2 = np.zeros((n_mu, n_l))

    if n_l == 0:
        return p00, p0p2, p2p2, p2m2

    # l = 0
    # Ref: gen_sph_funcs.pro lines 41-46
    p00[:, 0] = 1.0

    if n_l == 1:
        return p00, p0p2, p2p2, p2m2

    # l = 1
    # Ref: gen_sph_funcs.pro lines 48-53
    p00[:, 1] = mu

    if n_l == 2:
        return p00, p0p2, p2p2, p2m2

    # l = 2
    # Ref: gen_sph_funcs.pro lines 55-65
    da_0p2 = 0.25 * np.sqrt(6.0)
    mu2    = mu * mu
    p00[:, 2]  = (3.0 * mu2 - 1.0) / 2.0
    p0p2[:, 2] = da_0p2 * (mu2 - 1.0)
    p2p2[:, 2] = 0.25 * (1.0 + mu) ** 2
    p2m2[:, 2] = 0.25 * (1.0 - mu) ** 2

    # Recurrence for l >= 3
    # Ref: gen_sph_funcs.pro lines 67-96
    for j in range(2, n_l - 1):
        jf     = float(j)
        j_2    = jf * jf
        jp1    = jf + 1.0
        jjp1   = jf * jp1
        jp1_2  = jp1 * jp1
        twojp1 = 2.0 * jf + 1.0
        l      = j + 1    # index of the term being computed

        # p00: standard Legendre recurrence
        p00[:, l] = (twojp1 * mu * p00[:, j] - jf * p00[:, j - 1]) / jp1

        # p0p2 recurrence
        da_0p2 = np.sqrt(j_2 - 4.0)
        db_0p2 = np.sqrt(jp1_2 - 4.0)
        p0p2[:, l] = (twojp1 * mu * p0p2[:, j] - da_0p2 * p0p2[:, j - 1]) / db_0p2

        # p2p2 and p2m2 recurrence
        da_2x2 = twojp1 * 4.0
        db_2x2 = jp1 * (j_2 - 4.0)
        dc_2x2 = jf * (jp1_2 - 4.0)
        da     = twojp1 * jjp1 * mu

        p2p2[:, l] = ((da - da_2x2) * p2p2[:, j]
                      - db_2x2 * p2p2[:, j - 1]) / dc_2x2
        p2m2[:, l] = ((da + da_2x2) * p2m2[:, j]
                      - db_2x2 * p2m2[:, j - 1]) / dc_2x2

    return p00, p0p2, p2p2, p2m2


def scatPhaseMatrixExp(
    n_l: int,
    qx: np.ndarray,
    qw: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """
    Expand the scattering phase matrix in generalised spherical functions.

    Parameters
    ----------
    n_l : int
        Number of expansion orders.
    qx : np.ndarray, shape (N_q,)
        Quadrature abscissae (cosines of scattering angles).
    qw : np.ndarray, shape (N_q,)
        Quadrature weights as returned directly by :func:`quadrature`
        (sum = 2 for Gauss-Legendre).  These are used as-is; no rescaling
        is applied inside this function.
    F : np.ndarray, shape (6, N_q)
        Phase matrix elements at each quadrature point, ordered as::

            F[0, :] = F11
            F[1, :] = F22
            F[2, :] = F33
            F[3, :] = F44
            F[4, :] = F12
            F[5, :] = F34

    Returns
    -------
    coefs : np.ndarray, shape (6, n_l)
        Expansion coefficients in the Mishchenko (1991) generalised
        spherical function basis, ordered as::

            coefs[0, :] = α₁  (F11)
            coefs[1, :] = α₂  (F22)
            coefs[2, :] = α₃  (F33)
            coefs[3, :] = α₄  (F44)
            coefs[4, :] = β₁  (F12)
            coefs[5, :] = β₂  (F34)

    Notes
    -----
    Use the raw weights from :func:`quadrature` (summing to 2), which is
    consistent with the convention in :func:`legendreExpand`.
    The expansion coefficients then satisfy the same normalisation as
    ``legendreExpand``: ``coefs[0, l]`` equals ``lc[l]`` from
    ``legendreExpand(F11, qv, qw)``.

    Ported from IDLsource/scat_phase_matrix_exp.pro.
    """
    qx = np.asarray(qx, dtype=np.float64)
    qw = np.asarray(qw, dtype=np.float64)
    F  = np.asarray(F,  dtype=np.float64)

    # Compute generalised spherical functions at all quadrature points
    # Ref: scat_phase_matrix_exp.pro lines 42-66
    p00, p0p2, p2p2, p2m2 = genSphFuncs(n_l, qx)  # each (n_q, n_l)

    # Weighted phase matrix values
    # Ref: scat_phase_matrix_exp.pro line 49  b1 = F[:,i] * qw[i]
    b = F * qw[np.newaxis, :]    # (6, n_q)

    # Intermediate: b[1] ↔ F22+F33, b[2] ↔ F22-F33
    # Ref: scat_phase_matrix_exp.pro lines 51-53
    b22 = b[1].copy()
    b33 = b[2].copy()
    b[1] = b22 + b33
    b[2] = b22 - b33

    # Accumulate expansion
    # Ref: scat_phase_matrix_exp.pro lines 55-60
    # p arrays: (n_q, n_l); b: (6, n_q)
    # coefs[k, l] = sum_q p(q, l) * b[k, q]
    coefs = np.zeros((6, n_l))
    coefs[0] = p00.T  @ b[0]   # (n_l, n_q) @ (n_q,)
    coefs[1] = p2p2.T @ b[1]
    coefs[2] = p2m2.T @ b[2]
    coefs[3] = p00.T  @ b[3]
    coefs[4] = p0p2.T @ b[4]
    coefs[5] = p0p2.T @ b[5]

    # Unscramble coefs[1] and coefs[2]
    # Ref: scat_phase_matrix_exp.pro lines 62-65
    c1 = coefs[1].copy()
    c2 = coefs[2].copy()
    coefs[1] = (c1 + c2) / 2.0
    coefs[2] = (c1 - c2) / 2.0

    # Apply (2l+1)/2 scaling
    # Ref: scat_phase_matrix_exp.pro lines 67-69
    l_arr = np.arange(n_l, dtype=np.float64)
    coefs *= ((2.0 * l_arr + 1.0) / 2.0)[np.newaxis, :]

    return coefs
