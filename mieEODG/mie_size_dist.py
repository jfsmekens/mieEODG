"""
Mie scattering parameters integrated over a particle size distribution.

Ported from:
    IDLsource/mie_size_dist.pro  (G. Thomas, Sep 2003; multiple revisions to 2015)

References:
    Bohren, C.F. and Huffman, D.R., "Absorption and Scattering of Light by
    Small Particles", Wiley-VCH, 1983.
    Hansen, J.E. and Travis, L.D., "Light scattering in planetary atmospheres",
    Space Sci. Rev., 16, 527–610, 1974.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .mie_single import mieSingle
from .quadrature import quadrature, shiftQuadrature
from .size_dist import SizeDistribution, LogNormalDist, ModifiedGammaDist

logger = logging.getLogger(__name__)

# Safety cap: do not allow size parameters above this value
_X_MAX = 10_000.0

# Optional tqdm support — imported once at module load
try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


@dataclass
class MieSizeDistResult:
    """
    Output of :func:`mieSizeDist`.

    Attributes
    ----------
    Bext : np.ndarray, shape (N_nu,) or scalar float64
        Extinction coefficient [same units as Nd * cross-section].
    Bsca : np.ndarray
        Scattering coefficient.
    w : np.ndarray
        Single-scatter albedo = Bsca / Bext.
    g : np.ndarray
        Asymmetry parameter, weighted average over the size distribution.
    Bbac : np.ndarray
        Backscattering coefficient.
    Gavg : np.ndarray
        Mean projected cross-sectional area per particle [same units as r²].
    Vavg : np.ndarray
        Mean volume per particle [same units as r³].
    Ravg : np.ndarray
        Number-weighted mean radius.
    RVW : np.ndarray
        Volume-weighted mean radius.
    SPM : np.ndarray or None, shape (4, N_angles) or (4, N_angles, N_nu)
        Distribution-averaged scattering phase matrix [F11, F33, F12, F34],
        normalized to the RT convention: (1/2) * integral(F11, d(cosθ)) = 1.
        None if no scattering angles were requested.
    n_pts : int
        Number of quadrature points used in the size integration.
    x_min : float
        Minimum size parameter in the integration grid.
    x_max : float
        Maximum size parameter in the integration grid.
    """
    Bext:  np.ndarray
    Bsca:  np.ndarray
    w:     np.ndarray
    g:     np.ndarray
    Bbac:  np.ndarray
    Gavg:  np.ndarray
    Vavg:  np.ndarray
    Ravg:  np.ndarray
    RVW:   np.ndarray
    SPM:   np.ndarray | None
    n_pts: int
    x_min: float
    x_max: float


def _dist_from_args(
    dist_name: str | SizeDistribution,
    Nd: float,
    params: np.ndarray | list | None,
) -> SizeDistribution:
    """
    Resolve a distribution object from either a (dist_name, params) pair or
    a pre-constructed :class:`SizeDistribution` instance.

    This bridges the legacy string API and the new OOP interface.
    """
    if isinstance(dist_name, SizeDistribution):
        return dist_name
    params = np.asarray(params, dtype=np.float64)
    if dist_name == 'log_normal':
        return LogNormalDist(r0=params[0], s=params[1], n=Nd)
    elif dist_name == 'modified_gamma':
        return ModifiedGammaDist(
            a=params[0], b=params[1], r_min=params[2], r_max=params[3], n=Nd
        )
    else:
        raise ValueError(
            f"Unknown size distribution: {dist_name!r}. "
            "Choose 'log_normal' or 'modified_gamma', or pass a SizeDistribution object."
        )


def _single_wavenumber(
    dist: SizeDistribution,
    nu: float,
    m: complex,
    mu: np.ndarray | None,
    n_pts: int,
    Rl: float,
    Ru: float,
) -> tuple:
    """
    Integrate Mie properties over the size distribution at one wavenumber.

    Parameters
    ----------
    dist : SizeDistribution
        Pre-constructed distribution object.
    nu : float
        Wavenumber [cm⁻¹ or consistent units with radius units].
    m : complex
        Complex refractive index.
    mu : np.ndarray or None
        Cosines of scattering angles.
    n_pts : int
        Number of trapezoid quadrature points.
    Rl, Ru : float
        Pre-computed radius bounds (may differ from dist.bounds() if
        the upper bound was capped for spectral runs).

    Returns
    -------
    Bext, Bsca, w, g, Bbac, Gavg, Vavg, Ravg, RVW : float
    SPM : np.ndarray (4, N_angles) or None
    x_min, x_max : float

    Ref: mie_size_dist.pro lines 120-184
    """
    if getattr(dist, 'dist_name', None) == 'monodisperse':
        # Delta n(r) = N·δ(r − r₀): the size integral collapses to a single
        # evaluation at r₀ with weight ∫ N·δ(r − r₀) dr = N.  No quadrature,
        # no smearing — the result is the exact single-particle optics scaled
        # by the number density N.
        R   = np.array([dist.r0], dtype=np.float64)
        W1P = np.array([dist.N],  dtype=np.float64)
    else:
        # Trapezoid quadrature on [Rl, Ru]
        # Ref: mie_size_dist.pro lines 136-137
        absc, wght = quadrature('T', n_pts)
        R, W1 = shiftQuadrature(absc, wght, Rl, Ru)

        # PDF weights at quadrature points via the distribution object
        # Ref: mie_size_dist.pro lines 123-130
        nR  = dist.pdf(R)
        W1P = W1 * nR                    # quadrature weight × n(R)

    # Size parameters
    Dx    = 2.0 * np.pi * R * nu         # shape (n_pts,)
    x_min = float(Dx[0])
    x_max = float(Dx[-1])

    # Mie calculation for all sizes at once
    # Ref: mie_size_dist.pro lines 144-168
    res = mieSingle(Dx, m, angles=mu, silent=True)

    # Area and volume quadrature weights
    # Ref: mie_size_dist.pro lines 170-171
    W1PA = W1P * np.pi * R ** 2          # π r² n(R) dr weight
    W1PV = W1PA * (4.0 / 3.0) * R       # (4/3)π r³ n(R) dr weight

    # Bulk coefficients
    # Ref: mie_size_dist.pro lines 173-177
    Bext = float(np.sum(W1PA * res.Qext))
    Bsca = float(np.sum(W1PA * res.Qsca))
    Bbac = float(np.sum(W1PA * res.Qbk))
    g    = float(np.sum(W1PA * res.g * res.Qsca) / Bsca) if Bsca > 0.0 else 0.0
    w    = Bsca / Bext if Bext > 0.0 else 0.0

    # Geometric averages
    Gavg = float(np.sum(W1PA))
    Vavg = float(np.sum(W1PV))
    Ravg = float(np.sum(W1P * R))
    RVW  = float(np.sum(W1PV * R) / Vavg) if Vavg > 0.0 else 0.0

    # Distribution-averaged phase matrix
    # Ref: mie_size_dist.pro lines 178-184
    # res.SPM has shape (4, N_angles, N_sizes); integrate over last axis.
    if mu is not None and res.SPM is not None:
        # weights: W1PA * Qsca, shape (N_sizes,) → broadcast over (4, N_angles)
        wt  = (W1PA * res.Qsca)[np.newaxis, np.newaxis, :]   # (1, 1, N_sizes)
        SPM = np.sum(res.SPM * wt, axis=2) / Bsca             # (4, N_angles)
    else:
        SPM = None

    return Bext, Bsca, w, g, Bbac, Gavg, Vavg, Ravg, RVW, SPM, x_min, x_max


def mieSizeDist(
    dist_name: str | SizeDistribution,
    Nd: float,
    params: np.ndarray | list | None,
    wavenumber: float | np.ndarray,
    Cm: complex | np.ndarray,
    angles: np.ndarray | None = None,
    n_pts: int | None = None,
    xres: float = 0.1,
    silent: bool = False,
    progress: bool | str = False,
) -> MieSizeDistResult:
    """
    Mie scattering parameters integrated over a particle size distribution.

    Parameters
    ----------
    dist_name : str
        Name of the size distribution.  One of:

        ``'log_normal'``
            Log-normal distribution.  ``params = [r0, s]``:

            * ``r0`` – median radius
            * ``s``  – spread parameter; standard deviation of ln(r) is ln(s)

        ``'modified_gamma'``
            Modified gamma distribution (Hansen & Travis 1974).
            ``params = [a, b, r_min, r_max]``:

            * ``a`` – effective radius parameter
            * ``b`` – effective variance parameter
            * ``r_min``, ``r_max`` – explicit radius bounds
    Nd : float
        Total number density of the particle distribution.
    params : array-like
        Size distribution parameters (see above).
    wavenumber : float or array-like, shape (N_nu,)
        Wavenumber(s) of the incident radiation.  Units must be consistent
        with the radius units used in ``params``
        (e.g. both in cm⁻¹ and cm, or both in μm⁻¹ and μm).
    Cm : complex scalar or array-like, shape (N_nu,)
        Complex refractive index.  Convention: imaginary part should be
        **negative** for absorbing particles.  If ``wavenumber`` is an array,
        ``Cm`` must be broadcastable to the same shape.
    angles : array-like or None
        Cosines of scattering angles for the phase matrix.  If None, only
        bulk efficiencies are returned.
    n_pts : int or None
        Number of trapezoid quadrature points for size integration.  If None,
        the number is set automatically from ``xres`` and the integration
        range: ``max(int(2π (Ru − Rl) ν_max / xres), 200)``.
    xres : float
        Target spacing in size parameter for automatic ``n_pts`` selection.
        Default 0.1.  Ignored if ``n_pts`` is provided.
    silent : bool
        If True, suppress warnings.
    progress : bool or str
        Show a ``tqdm`` progress bar over the wavenumber loop (spectral path
        only; ignored for scalar ``wavenumber``).  Pass ``True`` for a bar
        with a generic description, or a string to set the bar description
        explicitly (useful when calling inside a larger loop, e.g.
        ``progress="H2SO4 r0=0.77µm"``).  Silently ignored if ``tqdm`` is
        not installed.  Default ``False``.

    Returns
    -------
    MieSizeDistResult
        Named result with bulk optical properties and (optionally) the
        distribution-averaged scattering phase matrix.

    Notes
    -----
    The phase matrix ``SPM`` is normalized to the radiative-transfer
    convention: ``(1/2) * integral(F11, d(cosθ)) = 1``.  This is identical
    to the convention used by :func:`mieSingle`.

    For spectral runs (``wavenumber`` array), ``Bext``, ``Bsca``, etc. have
    shape ``(N_nu,)`` and ``SPM`` has shape ``(4, N_angles, N_nu)``.
    The geometric averages ``Gavg``, ``Vavg``, ``Ravg``, ``RVW`` do NOT
    depend on wavenumber — they are returned as arrays for API consistency
    but are identical across wavelengths.

    Ported from IDLsource/mie_size_dist.pro.
    """
    wavenumber = np.atleast_1d(np.asarray(wavenumber, dtype=np.float64))
    Cm_in      = np.atleast_1d(np.asarray(Cm, dtype=np.complex128))
    n_nu       = len(wavenumber)

    if Cm_in.shape != wavenumber.shape:
        if Cm_in.size == 1:
            Cm_in = np.broadcast_to(Cm_in, wavenumber.shape).copy()
        else:
            raise ValueError(
                f"wavenumber and Cm must have the same length; "
                f"got {wavenumber.shape} and {Cm_in.shape}."
            )

    if not silent and np.any(Cm_in.imag > 0.0):
        logger.warning(
            "Imaginary part of refractive index is positive. Convention: "
            "imaginary part should be negative for absorbing particles. "
            "Pass silent=True to suppress this warning."
        )

    mu = None if angles is None else np.asarray(angles, dtype=np.float64)

    # Build distribution object (bridges string API and OOP interface)
    dist = _dist_from_args(dist_name, Nd, params)

    # Determine integration bounds
    # Ref: mie_size_dist.pro lines 103-140
    Rl, Ru = dist.bounds()
    nu_max = float(np.max(wavenumber))

    if 2.0 * np.pi * Rl * nu_max >= _X_MAX:
        raise ValueError(
            f"Lower radius bound Rl={Rl:.4g} gives size parameter "
            f"{2*np.pi*Rl*nu_max:.1f} >= {_X_MAX} at ν={nu_max}. "
            "Check your distribution parameters and wavenumber units."
        )

    if 2.0 * np.pi * Ru * nu_max >= _X_MAX:
        Ru_cap = (_X_MAX - 1.0) / (2.0 * np.pi * nu_max)
        if not silent:
            logger.warning(
                f"Upper radius bound Ru={Ru:.4g} truncated to {Ru_cap:.4g} "
                f"to keep x < {_X_MAX} at ν={nu_max}."
            )
        Ru = Ru_cap

    # Determine number of quadrature points
    # Ref: mie_size_dist.pro lines 141-143
    if n_pts is None:
        n_pts = max(int(2.0 * np.pi * (Ru - Rl) * nu_max / xres), 200)

    # --- Scalar wavenumber path ---
    if n_nu == 1:
        nu = float(wavenumber[0])
        m  = complex(Cm_in[0])
        (Bext, Bsca, w, g, Bbac,
         Gavg, Vavg, Ravg, RVW,
         SPM, x_min, x_max) = _single_wavenumber(
            dist, nu, m, mu, n_pts, Rl, Ru
        )
        return MieSizeDistResult(
            Bext  = np.float64(Bext),
            Bsca  = np.float64(Bsca),
            w     = np.float64(w),
            g     = np.float64(g),
            Bbac  = np.float64(Bbac),
            Gavg  = np.float64(Gavg),
            Vavg  = np.float64(Vavg),
            Ravg  = np.float64(Ravg),
            RVW   = np.float64(RVW),
            SPM   = SPM,
            n_pts = n_pts,
            x_min = x_min,
            x_max = x_max,
        )

    # --- Spectral path: iterate over wavenumbers ---
    # Build the iterator — optionally wrapped with a tqdm progress bar.
    # tqdm.auto picks the right backend (notebook vs. terminal) automatically.
    def _spectral_iter():
        it = zip(wavenumber, Cm_in)
        if progress is not False and _TQDM_AVAILABLE:
            desc = progress if isinstance(progress, str) else 'mieSizeDist'
            return _tqdm(it, total=n_nu, desc=desc, unit='ν',
                         leave=False, mininterval=0.5)
        return it

    Bext_arr = np.empty(n_nu)
    Bsca_arr = np.empty(n_nu)
    w_arr    = np.empty(n_nu)
    g_arr    = np.empty(n_nu)
    Bbac_arr = np.empty(n_nu)
    Gavg_arr = np.empty(n_nu)
    Vavg_arr = np.empty(n_nu)
    Ravg_arr = np.empty(n_nu)
    RVW_arr  = np.empty(n_nu)
    x_min_arr = np.empty(n_nu)
    x_max_arr = np.empty(n_nu)

    if mu is not None:
        SPM_arr = np.empty((4, len(mu), n_nu))
    else:
        SPM_arr = None

    for j, (nu, m) in enumerate(_spectral_iter()):
        (bext, bsca, wj, gj, bbac,
         gavg, vavg, ravg, rvw,
         spm, xmin, xmax) = _single_wavenumber(
            dist, float(nu), complex(m), mu, n_pts, Rl, Ru
        )
        Bext_arr[j]  = bext
        Bsca_arr[j]  = bsca
        w_arr[j]     = wj
        g_arr[j]     = gj
        Bbac_arr[j]  = bbac
        Gavg_arr[j]  = gavg
        Vavg_arr[j]  = vavg
        Ravg_arr[j]  = ravg
        RVW_arr[j]   = rvw
        x_min_arr[j] = xmin
        x_max_arr[j] = xmax
        if mu is not None and spm is not None:
            SPM_arr[:, :, j] = spm

    return MieSizeDistResult(
        Bext  = Bext_arr,
        Bsca  = Bsca_arr,
        w     = w_arr,
        g     = g_arr,
        Bbac  = Bbac_arr,
        Gavg  = Gavg_arr,
        Vavg  = Vavg_arr,
        Ravg  = Ravg_arr,
        RVW   = RVW_arr,
        SPM   = SPM_arr,
        n_pts = n_pts,
        x_min = float(np.min(x_min_arr)),
        x_max = float(np.max(x_max_arr)),
    )
