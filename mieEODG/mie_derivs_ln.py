"""
Mie scattering parameters and analytical derivatives for a log-normal
size distribution.

Ported from:
    IDLsource/mie_derivs_ln.pro  (G. Thomas, Sep 2003; Feb 2004)

Reference:
    Grainger, R.G., J. Lucas, G.E. Thomas, G. Ewan,
    "The Calculation of Mie Derivatives", Appl. Opt., 2004.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .mie_single import mieSingle
from .quadrature import quadrature, shiftQuadrature
from .size_dist import _TQ

logger = logging.getLogger(__name__)

# Safety cap on size parameter
_X_MAX = 12_000.0


@dataclass
class MieDerivLnResult:
    """
    Output of :func:`mieSterivsLn`.

    Attributes
    ----------
    Bext : float
        Volume extinction coefficient.
    Bsca : float
        Volume scattering coefficient.
    dBextdN : float
        d Bext / d N  (proportional to Bext/N since Bext ∝ N).
    dBextdRm : float
        d Bext / d Rm.
    dBextdS : float
        d Bext / d S.
    dBscadN : float
        d Bsca / d N.
    dBscadRm : float
        d Bsca / d Rm.
    dBscadS : float
        d Bsca / d S.
    i1 : np.ndarray or None, shape (N_angles,)
        Integrated first intensity function Σ n(r) |S1(r,θ)|² dr,
        evaluated at each requested angle.  None if no angles requested.
    i2 : np.ndarray or None, shape (N_angles,)
        Integrated second intensity function.
    di1dN : np.ndarray or None, shape (N_angles,)
        d i1 / d N.
    di1dRm : np.ndarray or None, shape (N_angles,)
        d i1 / d Rm.
    di1dS : np.ndarray or None, shape (N_angles,)
        d i1 / d S.
    di2dN : np.ndarray or None, shape (N_angles,)
        d i2 / d N.
    di2dRm : np.ndarray or None, shape (N_angles,)
        d i2 / d Rm.
    di2dS : np.ndarray or None, shape (N_angles,)
        d i2 / d S.
    n_pts : int
        Number of quadrature points used.
    x_min : float
        Minimum size parameter.
    x_max : float
        Maximum size parameter.
    """
    Bext:    float
    Bsca:    float
    dBextdN:  float
    dBextdRm: float
    dBextdS:  float
    dBscadN:  float
    dBscadRm: float
    dBscadS:  float
    i1:      np.ndarray | None
    i2:      np.ndarray | None
    di1dN:   np.ndarray | None
    di1dRm:  np.ndarray | None
    di1dS:   np.ndarray | None
    di2dN:   np.ndarray | None
    di2dRm:  np.ndarray | None
    di2dS:   np.ndarray | None
    n_pts:   int
    x_min:   float
    x_max:   float


def mieSterivsLn(
    N: float,
    Rm: float,
    S: float,
    wavenumber: float,
    Cm: complex,
    angles: np.ndarray | None = None,
    n_pts: int | None = None,
    xres: float = 0.1,
    silent: bool = False,
) -> MieDerivLnResult:
    """
    Mie scattering and analytical derivatives for a log-normal size distribution.

    Computes the bulk extinction and scattering coefficients and their
    analytical derivatives with respect to the log-normal distribution
    parameters (number density, median radius, spread).

    Parameters
    ----------
    N : float
        Number density of the particle distribution.
    Rm : float
        Median radius of the distribution.  Units must be consistent with
        ``wavenumber`` (e.g. both in μm⁻¹ and μm, or cm⁻¹ and cm).
    S : float
        Spread parameter; the standard deviation of ln(r) is ln(S).
        Must be > 1.
    wavenumber : float
        Wavenumber of the incident radiation.  Positive scalar.
    Cm : complex
        Complex refractive index.  Convention: imaginary part should be
        **negative** for absorbing particles.
    angles : array-like or None
        Cosines of scattering angles for the integrated intensity functions.
        If None, intensity outputs are None.
    n_pts : int or None
        Number of trapezoid quadrature points.  If None, set automatically
        from ``xres``: ``max(int(2π (Ru − Rl) ν / 0.1), 200)``.
    xres : float
        Target size-parameter spacing.  Default 0.1.  Ignored if ``n_pts``
        is given.
    silent : bool
        Suppress the positive-imaginary-part warning.

    Returns
    -------
    MieDerivLnResult

    Notes
    -----
    The derivatives are computed analytically by differentiating the
    quadrature integral through the log-normal weight function.
    Specifically, for a generic integral ``I = Σ w(R) n(R) Q(R) π R²``:

    * ``dI/dN  = I / N``   (since n(R) ∝ N)
    * ``dI/dRm = Σ w n Q π R² · ln(R/Rm) / (Rm ln²(S))``
    * ``dI/dS  = [Σ w n Q π R² · ln²(R/Rm) / ln²(S) - Bext] / (S ln(S))``

    Ref: mie_derivs_ln.pro, Grainger et al. (2004).

    Ported from IDLsource/mie_derivs_ln.pro.
    """
    m = complex(Cm)

    if not silent and m.imag > 0.0:
        logger.warning(
            "Imaginary part of refractive index is positive. Convention: "
            "imaginary part should be negative for absorbing particles. "
            "Pass silent=True to suppress this warning."
        )

    mu = None if angles is None else np.asarray(angles, dtype=np.float64)

    # Radius integration bounds (same formula as mieSizeDist log-normal)
    # Ref: mie_derivs_ln.pro lines 76-78
    ln_s = np.log(S)
    Rl   = np.exp(np.log(Rm) + _TQ * ln_s)                  # lower tail
    Ru   = np.exp(np.log(Rm) - _TQ * ln_s + np.log(4.0))    # upper tail

    if 2.0 * np.pi * Ru * wavenumber >= _X_MAX:
        Ru = (_X_MAX - 1.0) / (2.0 * np.pi * wavenumber)
        if not silent:
            logger.warning(
                f"Upper radius bound truncated to Ru={Ru:.4g} to keep x < {_X_MAX}."
            )

    # Quadrature points
    # Ref: mie_derivs_ln.pro lines 82-87
    if n_pts is None:
        n_pts = max(int(2.0 * np.pi * (Ru - Rl) * wavenumber / xres), 200)

    absc, wght = quadrature('T', n_pts)
    R, Wghtr   = shiftQuadrature(absc, wght, Rl, Ru)

    # Log-normal PDF weights (W1 = N * Tmp)
    # Ref: mie_derivs_ln.pro lines 89-91
    Tmp = (np.exp(-0.5 * (np.log(R / Rm) / ln_s) ** 2)
           / (np.sqrt(2.0 * np.pi) * ln_s * R))
    W1  = N * Tmp   # = n(R)

    Dx     = 2.0 * np.pi * R * wavenumber
    x_min  = float(Dx[0])
    x_max  = float(Dx[-1])
    lnRRm  = np.log(R / Rm)   # pre-compute for speed (Ref: mie_derivs_ln.pro line 102)

    # Mie calculation for all sizes at once
    # Ref: mie_derivs_ln.pro lines 93-100
    res = mieSingle(Dx, m, angles=mu, silent=True)

    # Area-weighted quadrature factor
    WQpi = Wghtr * W1 * np.pi * R ** 2    # shape (n_pts,)

    # Bulk extinction and scattering
    # Ref: mie_derivs_ln.pro lines 103-114
    Bext = float(np.sum(WQpi * res.Qext))
    Bsca = float(np.sum(WQpi * res.Qsca))

    # Derivatives with respect to distribution parameters
    # Ref: mie_derivs_ln.pro lines 104-116
    # dBext/dN = Bext/N  (Bext ∝ N)
    dBextdN  = Bext / N
    dBscadN  = Bsca / N

    # dBext/dRm: derivative of n(R) wrt Rm gives n(R)*ln(R/Rm)/(Rm*ln²(S))
    dBextdRm = float(np.sum(WQpi * res.Qext * lnRRm) / (ln_s ** 2 * Rm))
    dBscadRm = float(np.sum(WQpi * res.Qsca * lnRRm) / (ln_s ** 2 * Rm))

    # dBext/dS: derivative of n(R) wrt S gives n(R)*[ln²(R/Rm)/ln²(S) - 1]/(S*ln(S))
    dBextdS  = float((np.sum(WQpi * res.Qext * lnRRm ** 2) / ln_s ** 2
                      - Bext) / (S * ln_s))
    dBscadS  = float((np.sum(WQpi * res.Qsca * lnRRm ** 2) / ln_s ** 2
                      - Bsca) / (S * ln_s))

    # Intensity functions and derivatives
    # Ref: mie_derivs_ln.pro lines 117-138
    if mu is not None and res.S1 is not None:
        # S1, S2 shape: (N_angles, N_sizes)
        i1_raw = (res.S1 * res.S1.conjugate()).real   # |S1|^2, shape (N_mu, n_pts)
        i2_raw = (res.S2 * res.S2.conjugate()).real

        # Integrate over sizes with W1 weights (no π R^2 — intensity convention)
        # Ref: mie_derivs_ln.pro line 119
        W_i = Wghtr * W1                              # shape (n_pts,)

        i1  = np.sum(W_i[np.newaxis, :] * i1_raw, axis=1)   # (N_mu,)
        i2  = np.sum(W_i[np.newaxis, :] * i2_raw, axis=1)

        di1dN  = i1 / N
        di2dN  = i2 / N

        # Weighted sums for Rm and S derivatives (same weight derivative as Bext)
        WlnRRm = W_i * lnRRm     # shape (n_pts,)

        di1dRm = np.sum(WlnRRm[np.newaxis, :] * i1_raw, axis=1) / (ln_s ** 2 * Rm)
        di2dRm = np.sum(WlnRRm[np.newaxis, :] * i2_raw, axis=1) / (ln_s ** 2 * Rm)

        di1dS  = ((np.sum((W_i * lnRRm ** 2)[np.newaxis, :] * i1_raw, axis=1)
                   / ln_s ** 2 - i1) / (S * ln_s))
        di2dS  = ((np.sum((W_i * lnRRm ** 2)[np.newaxis, :] * i2_raw, axis=1)
                   / ln_s ** 2 - i2) / (S * ln_s))
    else:
        i1 = i2 = None
        di1dN = di1dRm = di1dS = None
        di2dN = di2dRm = di2dS = None

    return MieDerivLnResult(
        Bext=Bext,     Bsca=Bsca,
        dBextdN=dBextdN,   dBextdRm=dBextdRm,   dBextdS=dBextdS,
        dBscadN=dBscadN,   dBscadRm=dBscadRm,   dBscadS=dBscadS,
        i1=i1, i2=i2,
        di1dN=di1dN, di1dRm=di1dRm, di1dS=di1dS,
        di2dN=di2dN, di2dRm=di2dRm, di2dS=di2dS,
        n_pts=n_pts, x_min=x_min, x_max=x_max,
    )
