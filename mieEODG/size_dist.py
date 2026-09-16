"""
Particle size distribution classes.

Provides log-normal and modified-gamma size distributions with:
  - PDF evaluation
  - Analytical moment properties (number density, surface area density,
    volume density, mean radius, effective radius)
  - Integration bounds for numerical quadrature

Adapted from:
    srfm.size_distribution  (D. Grainger, A. Knizek; EODG, Jan 2025)
    IDLsource/mie_size_dist.pro  (G. Thomas; EODG, Sep 2003)

Reference:
    Hansen, J.E. and Travis, L.D., "Light scattering in planetary
    atmospheres", Space Sci. Rev., 16, 527–610, 1974.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm as _scipy_norm

# NumPy compatibility: ``np.trapezoid`` was added in NumPy 2.0 and ``np.trapz``
# removed in later 2.x.  Prefer ``trapezoid``; only touch ``trapz`` on NumPy <2
# (a plain ``getattr(np, "trapezoid", np.trapz)`` fails because the default is
# evaluated eagerly and ``np.trapz`` no longer exists).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Gaussian critical value: P(Z > Tq) = 0.999  (≈ -3.0902)
# Used to set log-normal integration bounds at the 0.1 % tail.
_TQ = float(_scipy_norm.ppf(0.001))


class SizeDistribution(ABC):
    """Abstract base class for particle size distributions."""

    @abstractmethod
    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        """Evaluate the number size distribution n(r) at radius r."""

    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Return (r_min, r_max) integration bounds."""

    @property
    @abstractmethod
    def dist_name(self) -> str:
        """String identifier used by :func:`~mieEODG.mieSizeDist`."""

    @property
    @abstractmethod
    def params(self) -> list:
        """Parameter list in the form expected by :func:`~mieEODG.mieSizeDist`."""


class LogNormalDist(SizeDistribution):
    """
    Log-normal particle size distribution.

    The number size distribution is:

    .. math::

        n(r) = \\frac{N}{\\sqrt{2\\pi}\\, r \\ln S}
               \\exp\\!\\left(-\\frac{(\\ln r - \\ln r_0)^2}{2 \\ln^2 S}\\right)

    Parameters
    ----------
    r0 : float
        Median radius.  Units must be consistent with the wavenumber used in
        :func:`~mieEODG.mieSizeDist`.
    s : float
        Spread parameter; the standard deviation of ln(r) is ln(s).
        Must be > 1 for a non-degenerate distribution.
    n : float or None
        Total number density.  Exactly one of ``n``, ``surface_area_density``,
        or ``volume_density`` must be supplied.
    surface_area_density : float or None
        Total particle surface area density (4πr² integrated over the
        distribution).  Units: [r²_units · N_units].
    volume_density : float or None
        Total particle volume density.  Units: [r³_units · N_units].

    Notes
    -----
    Analytical moment relationships for the log-normal distribution:

    * ``surface_area_density = N · 4π r0² · exp(2 ln²(s))``
    * ``volume_density       = N · (4π/3) r0³ · exp(9/2 ln²(s))``
    * ``mean_r               = r0 · exp(0.5 ln²(s))``
    * ``r_eff                = r0 · exp(2.5 ln²(s))``   (effective radius,
      cross-section-weighted)
    """

    dist_name = 'log_normal'

    def __init__(
        self,
        r0: float,
        s: float,
        *,
        n: float | None = None,
        surface_area_density: float | None = None,
        volume_density: float | None = None,
    ) -> None:
        if s <= 1.0:
            raise ValueError(f"Spread s must be > 1; got s={s}.")
        if r0 <= 0.0:
            raise ValueError(f"Median radius r0 must be > 0; got r0={r0}.")

        self.r0  = float(r0)
        self.s   = float(s)
        self._lnr0 = np.log(r0)
        self._lns  = np.log(s)

        # Analytical moments
        lns2 = self._lns ** 2
        self._sad_per_n = 4.0 * np.pi * r0 ** 2 * np.exp(2.0 * lns2)
        self._vd_per_n  = (4.0 * np.pi / 3.0) * r0 ** 3 * np.exp(4.5 * lns2)

        # Resolve N from whichever parameter was given
        n_given   = sum(x is not None for x in (n, surface_area_density, volume_density))
        if n_given != 1:
            raise ValueError(
                "Exactly one of n, surface_area_density, or volume_density "
                "must be supplied."
            )
        if n is not None:
            if n <= 0:
                raise ValueError(f"Number density n must be > 0; got n={n}.")
            self.N = float(n)
        elif surface_area_density is not None:
            if surface_area_density <= 0:
                raise ValueError("surface_area_density must be > 0.")
            self.N = float(surface_area_density) / self._sad_per_n
        else:
            if volume_density <= 0:
                raise ValueError("volume_density must be > 0.")
            self.N = float(volume_density) / self._vd_per_n

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def surface_area_density(self) -> float:
        """Total surface area density (analytical moment)."""
        return self.N * self._sad_per_n

    @property
    def volume_density(self) -> float:
        """Total volume density (analytical moment)."""
        return self.N * self._vd_per_n

    @property
    def mean_r(self) -> float:
        """Arithmetic mean radius = r0 · exp(0.5 ln²(s))."""
        return self.r0 * np.exp(0.5 * self._lns ** 2)

    @property
    def r_eff(self) -> float:
        """Effective radius (cross-section-weighted) = r0 · exp(2.5 ln²(s))."""
        return self.r0 * np.exp(2.5 * self._lns ** 2)

    @property
    def params(self) -> list:
        """[r0, s] as expected by :func:`~mieEODG.mieSizeDist`."""
        return [self.r0, self.s]

    # ------------------------------------------------------------------
    # SizeDistribution interface
    # ------------------------------------------------------------------

    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        """
        Evaluate n(r) at radius r.

        Parameters
        ----------
        r : array-like or float
            Radius value(s).

        Returns
        -------
        np.ndarray
            Number size distribution n(r) [same units as N / r].
        """
        r = np.asarray(r, dtype=np.float64)
        return (
            self.N / (np.sqrt(2.0 * np.pi) * r * self._lns) *
            np.exp(-0.5 * ((np.log(r) - self._lnr0) / self._lns) ** 2)
        )

    def bounds(self) -> tuple[float, float]:
        """
        Return (r_min, r_max) integration bounds at the 0.1 % tails.

        Uses the same criterion as :func:`~mieEODG.mieSizeDist` to ensure
        consistency when a distribution object is passed directly.
        """
        r_min = np.exp(self._lnr0 + _TQ * self._lns)           # lower tail
        r_max = np.exp(self._lnr0 - _TQ * self._lns + np.log(4.0))  # upper tail
        return float(r_min), float(r_max)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_um(
        cls,
        r0_um: float,
        s: float,
        *,
        n: float | None = None,
        surface_area_density: float | None = None,
        volume_density: float | None = None,
    ) -> 'LogNormalDist':
        """
        Construct a :class:`LogNormalDist` with *r0* supplied in µm.

        All other parameters are identical to :meth:`__init__`.  Internally,
        *r0_um* is converted to cm (``r0 = r0_um * 1e-4``) so that the
        object is consistent with the cm-based Mie kernel.

        Parameters
        ----------
        r0_um : float
            Median radius [µm].
        s : float
            Spread parameter (> 1).
        n, surface_area_density, volume_density
            See :meth:`__init__`.

        Returns
        -------
        LogNormalDist
        """
        return cls(
            r0=float(r0_um) * 1.0e-4,
            s=s,
            n=n,
            surface_area_density=surface_area_density,
            volume_density=volume_density,
        )

    # ------------------------------------------------------------------
    # Unit-aware properties
    # ------------------------------------------------------------------

    @property
    def r0_um(self) -> float:
        """Median radius in µm (``r0 * 1e4``)."""
        return self.r0 * 1.0e4

    def __repr__(self) -> str:
        return (
            f"LogNormalDist(r0={self.r0:.4g}, s={self.s:.4g}, N={self.N:.4g})"
        )


class ModifiedGammaDist(SizeDistribution):
    """
    Modified gamma particle size distribution (Hansen & Travis 1974).

    The number size distribution is:

    .. math::

        n(r) = N \\cdot r^{(1 - 3b)/b} \\exp(-r / (a b))

    where ``a`` is the effective radius and ``b`` is the effective variance.

    Parameters
    ----------
    a : float
        Effective radius parameter.
    b : float
        Effective variance parameter.  Must be > 0.
    r_min : float
        Lower radius cutoff.
    r_max : float
        Upper radius cutoff.
    n : float
        Total number density (scalar multiplier).

    Notes
    -----
    Unlike the log-normal distribution, the modified-gamma does not have
    simple closed-form moment expressions for all r_min/r_max combinations.
    The bounds ``(r_min, r_max)`` are explicit parameters rather than
    derived from a tail probability.
    """

    dist_name = 'modified_gamma'

    def __init__(
        self,
        a: float,
        b: float,
        r_min: float,
        r_max: float,
        n: float = 1.0,
    ) -> None:
        if b <= 0.0:
            raise ValueError(f"Effective variance b must be > 0; got b={b}.")
        if r_min <= 0.0 or r_max <= r_min:
            raise ValueError(
                f"Need 0 < r_min < r_max; got r_min={r_min}, r_max={r_max}."
            )
        self.a     = float(a)
        self.b     = float(b)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.N     = float(n)

    @property
    def params(self) -> list:
        """[a, b, r_min, r_max] as expected by :func:`~mieEODG.mieSizeDist`."""
        return [self.a, self.b, self.r_min, self.r_max]

    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        """
        Evaluate n(r) at radius r.

        Parameters
        ----------
        r : array-like or float
            Radius value(s).

        Returns
        -------
        np.ndarray
            Number size distribution n(r).
        """
        r = np.asarray(r, dtype=np.float64)
        return self.N * r ** ((1.0 - 3.0 * self.b) / self.b) * np.exp(-r / (self.a * self.b))

    def bounds(self) -> tuple[float, float]:
        """Return ``(r_min, r_max)`` as specified at construction."""
        return self.r_min, self.r_max

    @property
    def mean_r(self) -> float:
        """Numerical mean radius (approximate, via midpoint of bounds)."""
        # Exact moment requires Γ functions; return the mode instead:
        # mode of modified-gamma = a*(1 - 3b + sqrt((1-3b)^2 + 4b)) ... complex.
        # Just return the distribution midpoint for a practical estimate.
        return (self.r_min + self.r_max) / 2.0

    @classmethod
    def from_um(
        cls,
        a_um:     float,
        b:        float,
        r_min_um: float,
        r_max_um: float,
        *,
        n: float = 1.0,
    ) -> 'ModifiedGammaDist':
        """
        Construct a :class:`ModifiedGammaDist` with radii specified in µm.

        Parameters
        ----------
        a_um : float
            Effective radius parameter [µm].
        b : float
            Effective variance parameter (dimensionless).  Must be > 0.
        r_min_um : float
            Lower radius cutoff [µm].
        r_max_um : float
            Upper radius cutoff [µm].
        n : float
            Total number density.  Default 1.0.

        Returns
        -------
        ModifiedGammaDist
            Instance with all radii converted to cm.
        """
        return cls(
            a     = a_um     * 1e-4,
            b     = b,
            r_min = r_min_um * 1e-4,
            r_max = r_max_um * 1e-4,
            n     = n,
        )

    @property
    def a_um(self) -> float:
        """Effective radius parameter [µm]."""
        return self.a * 1e4

    @property
    def r_min_um(self) -> float:
        """Lower radius cutoff [µm]."""
        return self.r_min * 1e4

    @property
    def r_max_um(self) -> float:
        """Upper radius cutoff [µm]."""
        return self.r_max * 1e4

    def __repr__(self) -> str:
        return (
            f"ModifiedGammaDist(a={self.a:.4g}, b={self.b:.4g}, "
            f"r_min={self.r_min:.4g}, r_max={self.r_max:.4g}, N={self.N:.4g})"
        )


class GaussianDist(SizeDistribution):
    """
    Gaussian (normal) particle size distribution in linear radius space.

    The number size distribution is:

    .. math::

        n(r) = \\frac{N}{\\sigma \\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{(r - r_0)^2}{2\\sigma^2}\\right)

    The distribution is defined for r > 0.  Integration bounds are set at
    ±4σ from r₀, clipped to a positive minimum radius.

    Parameters
    ----------
    r0 : float
        Mean (peak) radius [same units as used in the Mie calculation, cm].
    sigma : float
        Standard deviation [cm].  For a comparable spread to a log-normal
        with spread parameter s, a useful approximation is
        ``sigma ≈ r0 * ln(s)``.
    n : float or None
        Total number density.  Exactly one of ``n``, ``surface_area_density``,
        or ``volume_density`` must be supplied.
    surface_area_density : float or None
    volume_density : float or None

    Notes
    -----
    Unlike the log-normal distribution, the Gaussian has support over all reals.
    The integration bounds clip the lower limit to ``max(r_min_abs, r0 - 4σ)``
    where ``r_min_abs = 1e-4 * r0`` to avoid r ≤ 0.  For well-behaved
    parameterisations (σ < r0 / 4) this has negligible effect.
    """

    dist_name = 'gaussian'

    def __init__(
        self,
        r0:    float,
        sigma: float,
        *,
        n: float | None = None,
        surface_area_density: float | None = None,
        volume_density: float | None = None,
    ) -> None:
        if r0 <= 0.0:
            raise ValueError(f"Mean radius r0 must be > 0; got r0={r0}.")
        if sigma <= 0.0:
            raise ValueError(f"Standard deviation sigma must be > 0; got sigma={sigma}.")

        self.r0    = float(r0)
        self.sigma = float(sigma)

        # Numerical moments over the positive half-line are needed for
        # surface_area_density / volume_density normalisation.  For σ << r0
        # the Gaussian integrals over (0, ∞) are indistinguishable from
        # the full-line integrals; we use the exact full-line values here
        # and accept negligible error for the parameter ranges of interest.
        sigma2 = sigma ** 2
        self._sad_per_n = 4.0 * np.pi * (r0 ** 2 + sigma2)
        self._vd_per_n  = (4.0 * np.pi / 3.0) * (r0 ** 3 + 3.0 * r0 * sigma2)

        n_given = sum(x is not None for x in (n, surface_area_density, volume_density))
        if n_given != 1:
            raise ValueError(
                "Exactly one of n, surface_area_density, or volume_density "
                "must be supplied."
            )
        if n is not None:
            if n <= 0:
                raise ValueError(f"Number density n must be > 0; got n={n}.")
            self.N = float(n)
        elif surface_area_density is not None:
            if surface_area_density <= 0:
                raise ValueError("surface_area_density must be > 0.")
            self.N = float(surface_area_density) / self._sad_per_n
        else:
            if volume_density <= 0:
                raise ValueError("volume_density must be > 0.")
            self.N = float(volume_density) / self._vd_per_n

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def surface_area_density(self) -> float:
        """Total surface area density."""
        return self.N * self._sad_per_n

    @property
    def volume_density(self) -> float:
        """Total volume density."""
        return self.N * self._vd_per_n

    @property
    def mean_r(self) -> float:
        """Mean radius = r0 (by definition)."""
        return self.r0

    @property
    def params(self) -> list:
        """[r0, sigma]."""
        return [self.r0, self.sigma]

    # ------------------------------------------------------------------
    # SizeDistribution interface
    # ------------------------------------------------------------------

    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        """
        Evaluate n(r) at radius r.

        Parameters
        ----------
        r : array-like or float

        Returns
        -------
        np.ndarray
            Number size distribution n(r).
        """
        r = np.asarray(r, dtype=np.float64)
        return (
            self.N / (self.sigma * np.sqrt(2.0 * np.pi)) *
            np.exp(-0.5 * ((r - self.r0) / self.sigma) ** 2)
        )

    def bounds(self) -> tuple[float, float]:
        """
        Return integration bounds at ±4σ, clipped to positive radius.
        """
        r_min = max(1e-4 * self.r0, self.r0 - 4.0 * self.sigma)
        r_max = self.r0 + 4.0 * self.sigma
        return float(r_min), float(r_max)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_um(
        cls,
        r0_um: float,
        sigma_um: float,
        *,
        n: float | None = None,
        surface_area_density: float | None = None,
        volume_density: float | None = None,
    ) -> 'GaussianDist':
        """
        Construct a :class:`GaussianDist` with *r0* and *sigma* supplied in µm.

        Both radii are converted to cm internally (``* 1e-4``).

        Parameters
        ----------
        r0_um : float
            Mean radius [µm].
        sigma_um : float
            Standard deviation [µm].
        n, surface_area_density, volume_density
            See :meth:`__init__`.

        Returns
        -------
        GaussianDist
        """
        return cls(
            r0=float(r0_um) * 1.0e-4,
            sigma=float(sigma_um) * 1.0e-4,
            n=n,
            surface_area_density=surface_area_density,
            volume_density=volume_density,
        )

    # ------------------------------------------------------------------
    # Unit-aware properties
    # ------------------------------------------------------------------

    @property
    def r0_um(self) -> float:
        """Mean radius in µm (``r0 * 1e4``)."""
        return self.r0 * 1.0e4

    @property
    def sigma_um(self) -> float:
        """Standard deviation in µm (``sigma * 1e4``)."""
        return self.sigma * 1.0e4

    def __repr__(self) -> str:
        return (
            f"GaussianDist(r0={self.r0:.4g}, sigma={self.sigma:.4g}, N={self.N:.4g})"
        )


class MonodisperseDist(SizeDistribution):
    """
    Monodisperse (single-size) population: all particles share radius r₀.

    This represents a delta function n(r) = N · δ(r − r₀): a population of
    ``N`` identical particles per unit volume.  It is **not** integrated by
    the quadrature machinery in :func:`~mieEODG.mieSizeDist` (a delta cannot
    be sampled at Gauss nodes); single-particle optical properties are
    obtained directly from :func:`~mieEODG.mieSingle`.  This class exists so
    a single size can be carried, labelled, and visualised (e.g. by
    :func:`~mieEODG.plot_size_dist`, which renders it as a stem at r₀)
    consistently with the distribution classes.

    Parameters
    ----------
    r0 : float
        Particle radius [cm].
    n : float
        Number density [cm⁻³].  Default 1.0.
    """

    dist_name = 'monodisperse'

    def __init__(self, r0: float, *, n: float = 1.0) -> None:
        if r0 <= 0.0:
            raise ValueError(f"Radius r0 must be > 0; got r0={r0}.")
        if n <= 0.0:
            raise ValueError(f"Number density n must be > 0; got n={n}.")
        self.r0 = float(r0)
        self.N  = float(n)

    @classmethod
    def from_um(cls, r0_um: float, *, n: float = 1.0) -> 'MonodisperseDist':
        """Construct with ``r0`` supplied in µm (converted to cm internally)."""
        return cls(r0_um * 1.0e-4, n=n)

    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        """Delta function — zero everywhere except (formally) at r₀.

        Returned as zeros for array sampling; the mass is concentrated at a
        single point and is not representable on a continuous grid.  Callers
        that need the single-size optics should use :func:`mieSingle`.
        """
        return np.zeros_like(np.atleast_1d(np.asarray(r, dtype=np.float64)))

    def bounds(self) -> tuple[float, float]:
        """Return a tight bracket around r₀ (±1 %)."""
        return (0.99 * self.r0, 1.01 * self.r0)

    @property
    def params(self) -> list:
        return [self.r0]

    @property
    def r0_um(self) -> float:
        """Radius in µm (``r0 * 1e4``)."""
        return self.r0 * 1.0e4

    def __repr__(self) -> str:
        return f"MonodisperseDist(r0={self.r0:.4g}, N={self.N:.4g})"


class TabulatedDist(SizeDistribution):
    """
    Arbitrary size distribution defined by a tabulated n(r) curve.

    Wraps measured or user-supplied size-distribution data so it can be used
    like the analytic distributions (e.g. fed to :class:`~mieEODG.AerosolXSC`).
    ``pdf`` linearly interpolates the table and is zero outside its range.

    Parameters
    ----------
    r : array-like
        Radii [cm], strictly increasing (sorted on construction).
    n : array-like
        Number size distribution n(r) at each radius (same length as ``r``).
        Units arbitrary; the total number density ``N`` is the trapezoidal
        integral ∫ n(r) dr.
    """

    dist_name = 'tabulated'

    def __init__(self, r: np.ndarray, n: np.ndarray) -> None:
        r = np.asarray(r, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        if r.ndim != 1 or r.shape != n.shape:
            raise ValueError("r and n must be 1-D arrays of equal length.")
        if r.size < 2:
            raise ValueError("Tabulated distribution needs at least 2 points.")
        order = np.argsort(r)
        self.r = r[order]
        self._n = np.clip(n[order], 0.0, None)   # number density is non-negative
        if not np.all(self.r > 0):
            raise ValueError("All radii must be > 0.")
        self.N = float(_trapz(self._n, self.r))

    @classmethod
    def from_dndlnr(cls, r_um: np.ndarray, dndlnr: np.ndarray,
                    *, n: float | None = None) -> 'TabulatedDist':
        """Construct from the plotted quantity dN/d ln r vs radius [µm].

        ``dN/d ln r = r · n(r)`` ⟹ ``n(r) = (dN/d ln r) / r``.  If ``n`` is
        given, the curve is rescaled so the total number density equals it
        (useful since a measured PDF is often shape-only / normalised).
        """
        r_cm = np.asarray(r_um, dtype=np.float64) * 1e-4
        n_r = np.asarray(dndlnr, dtype=np.float64) / r_cm
        d = cls(r_cm, n_r)
        if n is not None and d.N > 0:
            d._n *= (n / d.N)
            d.N = float(n)
        return d

    def dndlnr_um(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(radius_µm, dN/d ln r)`` for plotting / export."""
        return self.r * 1e4, self._n * self.r

    def pdf(self, r: np.ndarray | float) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        return np.interp(r, self.r, self._n, left=0.0, right=0.0)

    def bounds(self) -> tuple[float, float]:
        return float(self.r[0]), float(self.r[-1])

    @property
    def params(self) -> list:
        return []

    @property
    def r0_um(self) -> float:
        """Number-weighted mean radius [µm] (for labelling/colouring)."""
        if self.N <= 0:
            return float('nan')
        return float(_trapz(self._n * self.r, self.r) / self.N) * 1e4

    def __repr__(self) -> str:
        return f"TabulatedDist({len(self.r)} pts, N={self.N:.4g})"


def _make_dist(
    dist_type: str,
    r0_cm: float,
    s: float,
) -> SizeDistribution:
    """
    Instantiate a size distribution for a single LUT grid point (r₀ sweep).

    Shared by :class:`~mieEODG.AerosolLUT` and
    :class:`~mieEODG.lut_multi.MultiDimLUT` so both sweep r₀ identically.

    Parameters
    ----------
    dist_type : {'log_normal', 'gaussian', 'monodisperse'}
    r0_cm : float
        Median / mean radius [cm].
    s : float
        Spread parameter.  For log-normal: geometric std > 1.  For Gaussian:
        ``sigma = r0 * ln(s)`` (comparable width).  Ignored for monodisperse.
    """
    if dist_type == 'log_normal':
        return LogNormalDist(r0=r0_cm, s=s, n=1.0)
    elif dist_type == 'gaussian':
        sigma = r0_cm * np.log(s)   # matches lognormal spread near r0
        return GaussianDist(r0=r0_cm, sigma=sigma, n=1.0)
    elif dist_type == 'monodisperse':
        return MonodisperseDist(r0=r0_cm, n=1.0)   # single size; s ignored
    else:
        raise ValueError(f"Unsupported dist_type for LUT: {dist_type!r}")


def createDist(
    dist_type: str,
    **kwargs,
) -> SizeDistribution:
    """
    Factory function for size distributions.

    Parameters
    ----------
    dist_type : str
        ``'log_normal'`` or ``'modified_gamma'``.
    **kwargs
        Passed to :class:`LogNormalDist` or :class:`ModifiedGammaDist`.

    Returns
    -------
    SizeDistribution

    Examples
    --------
    >>> d = createDist('log_normal', r0=0.5e-4, s=2.0, n=100.0)
    >>> d2 = createDist('log_normal', r0=0.5e-4, s=2.0, surface_area_density=1e-5)
    >>> d3 = createDist('modified_gamma', a=0.2e-4, b=0.1, r_min=0.01e-4, r_max=5e-4, n=50.0)
    """
    if dist_type == 'log_normal':
        return LogNormalDist(**kwargs)
    elif dist_type == 'modified_gamma':
        return ModifiedGammaDist(**kwargs)
    elif dist_type == 'gaussian':
        return GaussianDist(**kwargs)
    else:
        raise ValueError(
            f"Unknown distribution type: {dist_type!r}. "
            "Choose 'log_normal', 'modified_gamma', or 'gaussian'."
        )
