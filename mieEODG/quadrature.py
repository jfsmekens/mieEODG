"""
Quadrature abscissae and weights for numerical integration on [-1, 1].

Ported from:
    IDLsource/quadrature.pro  (R. Grainger, 7 Jun 2005; G. Thomas, 10 Jun 2005)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers (mirror of the nested IDL functions in quadrature.pro)
# ---------------------------------------------------------------------------

def _legendre(nn: int, x: float) -> tuple[float, float, float]:
    """
    Evaluate Legendre polynomials P_{nn-1}, P_{nn} and their derivative proxy.

    Returns (P_{nn-2}, P_{nn-1}, P_{nn}) — the three-term values needed for
    Newton iteration.  For nn == 0 and nn == 1 the earlier terms are
    undefined; callers must handle these edge cases.

    Parameters
    ----------
    nn : int
        Polynomial order.
    x : float
        Evaluation point.

    Returns
    -------
    pl : float   P_{nn-2}  (undefined for nn < 2)
    pm : float   P_{nn-1}  (undefined for nn < 1)
    pn : float   P_{nn}
    """
    # Ref: quadrature.pro::Legendre, lines 44-64
    if nn == 0:
        return 1.0, 1.0, 1.0      # pl, pm undefined; pn = P_0 = 1
    if nn == 1:
        return 1.0, 1.0, float(x) # pl undefined; pm = P_0; pn = P_1
    pl = 1.0
    pm = 1.0
    pn = float(x)
    for n in range(2, nn + 1):
        inv_n = 1.0 / n
        pl = pm
        pm = pn
        pn = (2.0 - inv_n) * x * pm - (1.0 - inv_n) * pl
    return pl, pm, pn


def _bessel_zero(s: float) -> float:
    """
    Approximation to the s-th zero of the Bessel function J_0, used as an
    initial guess for Lobatto abscissae.

    Ref: quadrature.pro::BesselZero, lines 3-13
    """
    A1 = -0.60792710185402662866327677925836
    A2 =  6.159589352810601113491669960271217e-2
    A3 = -0.981080301612647885671934857251079
    A4 = 11.7507817547326698965409187559573
    A5 = -2731.24978203593727776707267899459
    b = 4.0 * s + 1.0
    return 0.25 * np.pi * b * (
        1.0 + A1/b**2 + A2/b**4 + A3/b**6 + A4/b**8 + A5/b**10
    )


def _first_guess(quad_type: str, n: int, term: int) -> float:
    """
    Initial guess for the term-th abscissa of an n-point quadrature rule.

    Ref: quadrature.pro::FirstGuess, lines 15-23
    """
    qt = quad_type.upper()
    if qt == 'G':
        return np.cos(np.pi * (term - 0.25) / (n + 0.5))
    if qt == 'R':
        return 0.5 * (
            np.cos(np.pi * (term - 0.25) / (n + 0.5)) +
            np.cos(np.pi * (term - 0.25) / (n - 0.5))
        )
    if qt == 'L':
        return np.cos(
            _bessel_zero(term) / np.sqrt((n - 0.5)**2 + (0.25 - 1.0/np.pi**2))
        )
    raise ValueError(f"Unknown quadrature type: {quad_type!r}")


def _newton_step(quad_type: str, n: int, x: float) -> float:
    """
    One Newton-Raphson correction step for locating a quadrature abscissa.

    Ref: quadrature.pro::NewtonG, lines 25-42
    """
    qt = quad_type.upper()
    pl, pm, pn = _legendre(n, x)
    if qt == 'G':
        return (1.0 - x**2) * pn / (n * (pm - x * pn))
    if qt == 'R':
        numerator   = (1.0 + x) * (pm + pn)
        denominator = (
            ((n - 1) * pl - (n - 1) * x * pm + n * pm - n * x * pn) / (1.0 - x)
            - (pm + pn)
        )
        return numerator / denominator
    if qt == 'L':
        pl2, pm2, pn2 = _legendre(n - 1, x)
        numerator   = (1.0 - x**2) * (pm2 - x * pn2)
        denominator = (
            n * pl2 + 2.0 * (1.0 - n) * x * pm2
            + (x**2 * (n - 1.0) - 1.0) * pn2
        )
        return numerator / denominator
    raise ValueError(f"Unknown quadrature type: {quad_type!r}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def quadrature(
    quad_type: str,
    n_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute abscissae and weights for integration on the interval [-1, 1].

    Parameters
    ----------
    quad_type : str
        Quadrature rule.  One of:
        ``'T'`` – Trapezoid
        ``'G'`` – Gauss-Legendre
        ``'R'`` – Gauss-Radau
        ``'L'`` – Gauss-Lobatto
    n_pts : int
        Number of quadrature points.

    Returns
    -------
    abscissa : np.ndarray, shape (n_pts,)
        Quadrature points on [-1, 1], sorted ascending.
    weight : np.ndarray, shape (n_pts,)
        Corresponding integration weights.

    Notes
    -----
    Weights sum to 2 for all rules (integral of 1 over [-1, 1]).

    Ported from IDLsource/quadrature.pro::Quadrature.
    """
    qt = quad_type.upper()
    abscissa = np.zeros(n_pts)
    weight   = np.zeros(n_pts)
    n        = float(n_pts)

    if qt == 'T':
        # Ref: quadrature.pro lines 88-93
        abscissa      = -1.0 + 2.0 * np.arange(n_pts) / (n - 1.0)
        weight[:]     =  2.0 / (n - 1.0)
        weight[0]     =  1.0 / (n - 1.0)
        weight[-1]    =  1.0 / (n - 1.0)
        return abscissa, weight

    if qt == 'S':
        raise NotImplementedError("Simpson quadrature is not yet implemented.")

    if n_pts > 2000:
        raise ValueError(
            f"Too many quadrature points for rule {quad_type!r}: {n_pts} > 2000."
        )

    # --- Determine starting conditions for each rule ----------------------
    # Ref: quadrature.pro lines 101-121
    if qt == 'G':
        term  = 0
        zeros = n_pts
    elif qt == 'R':
        abscissa[-1] = -1.0
        weight[-1]   =  2.0 / n**2
        term  = 0
        zeros = n_pts - 1
    elif qt == 'L':
        abscissa[0]  =  1.0
        abscissa[-1] = -1.0
        weight[0]    =  2.0 / (n * (n - 1.0))
        weight[-1]   =  2.0 / (n * (n - 1.0))
        term  = 1
        zeros = n_pts - 2
    else:
        raise ValueError(f"Unknown quadrature type: {quad_type!r}")

    # Newton iteration to locate each zero
    # Ref: quadrature.pro lines 123-138
    sigfig = 14  # target decimal digits (double precision)
    tol    = 10.0**(-sigfig)

    for zero in range(1, zeros + 1):
        x  = _first_guess(qt, n_pts, zero)
        lx = 19262.0   # arbitrary starting value (Don Grainger's birthday)
        while abs(x - lx) > tol:
            lx = x
            x  = x - _newton_step(qt, n_pts, x)
        abscissa[term] = x
        _, pm, pn = _legendre(n_pts, x)
        if qt == 'G':
            weight[term] = 2.0 * (1.0 - x**2) / (n * pm)**2
        elif qt == 'R':
            weight[term] = (1.0 - x) / (n * pm)**2
        elif qt == 'L':
            weight[term] = 2.0 / (n * (n - 1.0) * pm**2)
        term += 1

    # IDL reverses the arrays at the end (ascending order)
    # Ref: quadrature.pro lines 139-140
    abscissa = abscissa[::-1].copy()
    weight   = weight[::-1].copy()
    return abscissa, weight


def shiftQuadrature(
    abscissa: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift quadrature abscissae and weights from [-1, 1] to an arbitrary [a, b].

    Parameters
    ----------
    abscissa : np.ndarray
        Quadrature points on [-1, 1], as returned by :func:`quadrature`.
    weights : np.ndarray
        Corresponding weights, as returned by :func:`quadrature`.
    a : float
        Lower bound of the target interval.
    b : float
        Upper bound of the target interval.

    Returns
    -------
    new_abscissa : np.ndarray
        Quadrature points on [a, b].
    new_weights : np.ndarray
        Rescaled weights for integration on [a, b].

    Ported from IDLsource/shift_quadrature.pro.
    """
    new_abscissa = ((a + b) + (b - a) * abscissa) / 2.0
    new_weights  = (b - a) * weights / 2.0
    return new_abscissa, new_weights
