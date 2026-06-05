"""
AerosolLUT — precomputed look-up table of Mie mass coefficients.

A LUT is a grid of (MEC, MSC, MAC) arrays computed over a range of median
radii for a fixed refractive index, size distribution type, and spectral grid.
At load time, PCHIP interpolators are instantiated over log(r0) so that the
forward call — returning a full spectrum for a given r0 — costs only a single
binary search and a vectorised multiply across all wavenumber channels.

Optional phase function caching
--------------------------------
Phase matrices and Legendre coefficients can be computed alongside the LUT
and stored in the cache file for fast DISORT integration. This is controlled
by the ``cache_phase`` parameter:

- **cache_phase=False** (default): Phase functions are computed on-demand
  at query time (~50-100 ms per r0). LUT files stay small (~100 KB).
  Recommended for most interactive use.

- **cache_phase=True**: Phase functions for all r0 nodes are computed during
  LUT creation and stored in the cache (~30-50 MB per LUT with 180 angles).
  Queries are instant. Recommended for optimization/Jacobian work with many
  repeated queries.

Typical workflow
----------------
::

    from mieEODG import ARIADatabase, AerosolLUT
    import numpy as np

    db  = ARIADatabase()
    ri  = db.read('Volcanic_Ash/Etna/...', wavenumber=wn, out_of_range='clip')

    # Compute once and cache; reload automatically on subsequent calls
    lut = AerosolLUT.load_or_compute(ri, r0_um=np.logspace(-1, 1, 20), s=1.5, wn=wn)
    spec = lut.forward(r0_um=1.2)   # {'MEC': ..., 'MSC': ..., 'MAC': ...}

    # With optional phase caching for DISORT
    lut_phase = AerosolLUT.load_or_compute(
        ri, r0_um=np.logspace(-1, 1, 20), s=1.5, wn=wn,
        theta_pts=np.linspace(0, 180, 180),
        n_legendre=50,
        cache_phase=True,
    )
    phase_result = lut_phase.phase_matrix_forward(1.2)
    # phase_result = {'phase_matrix': (N_wn, N_theta, 4),
    #                 'pmom': (N_wn, n_legendre),
    #                 'theta': np.array([...])}

    # Inspect the local cache
    AerosolLUT.list_cache()

Cache
-----
LUTs are stored in ``~/.mieEODG/lut_cache/`` by default (override with
``cache_dir``).  Each file is named by a SHA-256 hash of the defining
parameters (ri path, dist_type, s, r0 grid, wn grid, n_pts, and phase
parameters) so that any change in inputs automatically triggers a recompute.
Backend choice does not affect the cache key since both backends produce
identical results.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .aerosol import AerosolXSC
from .refindex import ARIADatabase, RefrIndex
from .size_dist import GaussianDist, LogNormalDist, SizeDistribution

logger = logging.getLogger(__name__)

# Default cache location — in package folder for easy access and management
# Falls back to user home if package folder isn't writable
_PACKAGE_CACHE_DIR = Path(__file__).parent / 'cache' / 'lut'
try:
    _PACKAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _DEFAULT_CACHE_DIR = _PACKAGE_CACHE_DIR
except (OSError, PermissionError):
    # Fallback if package folder isn't writable (e.g., system Python)
    _DEFAULT_CACHE_DIR = Path.home() / '.mieEODG' / 'lut_cache'

try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False


# ---------------------------------------------------------------------------
# Helper: distribution factory for LUT sweeps
# ---------------------------------------------------------------------------

def _make_dist(
    dist_type: str,
    r0_cm: float,
    s: float,
) -> SizeDistribution:
    """
    Instantiate a size distribution for a single LUT grid point.

    Parameters
    ----------
    dist_type : {'log_normal', 'gaussian'}
    r0_cm : float
        Median / mean radius [cm].
    s : float
        Spread parameter.  For log-normal: geometric std > 1.
        For Gaussian: sigma = r0 * ln(s), giving comparable width.
    """
    if dist_type == 'log_normal':
        return LogNormalDist(r0=r0_cm, s=s, n=1.0)
    elif dist_type == 'gaussian':
        sigma = r0_cm * np.log(s)   # matches lognormal spread near r0
        return GaussianDist(r0=r0_cm, sigma=sigma, n=1.0)
    else:
        raise ValueError(f"Unsupported dist_type for LUT: {dist_type!r}")


# ---------------------------------------------------------------------------
# AerosolLUT
# ---------------------------------------------------------------------------

class AerosolLUT:
    """
    Precomputed aerosol mass-coefficient look-up table.

    Stores MEC, MSC, MAC on a (r0, wavenumber) grid and exposes a fast
    forward method via PCHIP interpolators precomputed over log(r0).

    Attributes
    ----------
    wavenumber : np.ndarray, shape (N_wn,)
        Wavenumber grid [cm⁻¹].
    r0_um : np.ndarray, shape (N_r0,)
        LUT r0 grid [µm].
    MEC : np.ndarray, shape (N_r0, N_wn)
        Mass extinction coefficient [cm²/g].
    MSC : np.ndarray, shape (N_r0, N_wn)
        Mass scattering coefficient [cm²/g].
    MAC : np.ndarray, shape (N_r0, N_wn)
        Mass absorption coefficient [cm²/g].
    s : float
        Size distribution spread parameter used to generate the LUT.
    dist_type : str
        ``'log_normal'`` or ``'gaussian'``.
    density : float
        Bulk density [g/cm³] used in the mass coefficient calculation.
    category : str
        ARIA category key (from RefrIndex).
    substance : str
        ARIA substance name (SUBSTANCE header).
    ri_path : str
        Absolute path to the source ``.ri`` file.
    """

    def __init__(
        self,
        wavenumber: np.ndarray,
        r0_um:     np.ndarray,
        MEC:       np.ndarray,
        MSC:       np.ndarray,
        MAC:       np.ndarray,
        s:         float,
        dist_type: str,
        density:   float,
        category:  str = "",
        substance: str = "",
        ri_path:   str = "",
        theta_pts: np.ndarray | None = None,
        n_legendre: int | None = None,
        cache_phase: bool = False,
        phase_matrix_full: np.ndarray | None = None,
        pmom_full: np.ndarray | None = None,
    ) -> None:
        self.wavenumber = np.asarray(wavenumber, dtype=np.float64)
        self.r0_um      = np.asarray(r0_um,     dtype=np.float64)
        self.MEC        = np.asarray(MEC,        dtype=np.float64)
        self.MSC        = np.asarray(MSC,        dtype=np.float64)
        self.MAC        = np.asarray(MAC,        dtype=np.float64)
        self.s          = float(s)
        self.dist_type  = dist_type
        self.density    = float(density)
        self.category   = category
        self.substance  = substance
        self.ri_path    = ri_path

        # Phase function metadata and data
        self.theta_pts  = np.asarray(theta_pts, dtype=np.float64) if theta_pts is not None else None
        self.n_legendre = int(n_legendre) if n_legendre is not None else None
        self.cache_phase = bool(cache_phase)
        self.phase_matrix_full = np.asarray(phase_matrix_full, dtype=np.float64) if phase_matrix_full is not None else None
        self.pmom_full = np.asarray(pmom_full, dtype=np.float64) if pmom_full is not None else None

        # LRU cache for on-demand recomputation
        self._phase_cache: dict = {}

        self._build_interpolators()

    # ------------------------------------------------------------------
    # Interpolator construction (called at init and after load)
    # ------------------------------------------------------------------

    def _build_interpolators(self) -> None:
        """
        Precompute PCHIP interpolators over log(r0) for each coefficient.

        Interpolating in log(r0) space is appropriate because r0 grids are
        log-spaced and Mie cross-sections vary smoothly on a log scale.
        Cost: O(N_r0 × N_wn) at load time; negligible per forward call.
        """
        log_r0 = np.log(self.r0_um)
        # PchipInterpolator over axis=0 (r0 axis); operates on all wn at once
        self._interp_MEC = PchipInterpolator(log_r0, self.MEC, axis=0)
        self._interp_MSC = PchipInterpolator(log_r0, self.MSC, axis=0)
        self._interp_MAC = PchipInterpolator(log_r0, self.MAC, axis=0)

    # ------------------------------------------------------------------
    # Forward call
    # ------------------------------------------------------------------

    def forward(self, r0_um: float) -> dict[str, np.ndarray]:
        """
        Interpolate mass coefficients at a given median radius.

        Parameters
        ----------
        r0_um : float
            Median / mean radius [µm].  Must lie within the LUT r0 range;
            PCHIP will extrapolate but accuracy degrades outside the grid.

        Returns
        -------
        dict with keys ``'MEC'``, ``'MSC'``, ``'MAC'``
            Each value is a 1-D array of shape ``(N_wn,)`` [cm²/g].
        """
        log_r0 = np.log(float(r0_um))
        return {
            'MEC': self._interp_MEC(log_r0),
            'MSC': self._interp_MSC(log_r0),
            'MAC': self._interp_MAC(log_r0),
        }

    # ------------------------------------------------------------------
    # Phase function interface
    # ------------------------------------------------------------------

    def phase_matrix_forward(
        self,
        r0_um: float,
        *,
        theta_pts: np.ndarray | None = None,
        n_legendre: int | None = None,
    ) -> dict[str, np.ndarray | None]:
        """
        Return phase matrix and/or Legendre coefficients for a single r0.

        If ``cache_phase=True`` and data was precomputed, returns cached values.
        Otherwise, recomputes from scratch and caches the result in an LRU cache.

        Parameters
        ----------
        r0_um : float
            Median / mean radius [µm].
        theta_pts : np.ndarray or None
            Override the stored theta_pts. If None, uses self.theta_pts.
        n_legendre : int or None
            Override the stored n_legendre. If None, uses self.n_legendre.

        Returns
        -------
        dict
            Keys: ``'phase_matrix'``, ``'pmom'``, ``'theta'``.
            Values are ndarrays or None if not available.

        Raises
        ------
        RuntimeError
            If no phase data is available (neither cached nor can be computed).
        ValueError
            If ri_path is not set or refractive index cannot be loaded.
        """
        # Resolve which angles/Legendre to use
        _theta_pts = theta_pts if theta_pts is not None else self.theta_pts
        _n_legendre = n_legendre if n_legendre is not None else self.n_legendre

        if _theta_pts is None and _n_legendre is None:
            raise RuntimeError(
                "No phase function metadata for this LUT. "
                "Call load_or_compute(..., theta_pts=...) or (..., n_legendre=...) "
                "at creation time, or pass theta_pts/n_legendre to this method."
            )

        # Check LRU cache (in-memory)
        cache_key = (r0_um, tuple(_theta_pts) if _theta_pts is not None else None, _n_legendre)
        if cache_key in self._phase_cache:
            return self._phase_cache[cache_key]

        # If cached on disk, use it
        if self.cache_phase and (self.phase_matrix_full is not None or self.pmom_full is not None):
            # Find nearest r0 index
            idx = int(np.argmin(np.abs(self.r0_um - r0_um)))
            result = {
                'phase_matrix': self.phase_matrix_full[idx] if self.phase_matrix_full is not None else None,  # (N_wn, N_theta, 4)
                'pmom': self.pmom_full[idx] if self.pmom_full is not None else None,  # (N_wn, n_legendre)
                'theta': self.theta_pts,
            }
            self._phase_cache[cache_key] = result
            return result

        # Recompute on demand
        if not self.ri_path:
            raise ValueError(
                "Cannot recompute phase function: ri_path not set. "
                "LUT may have been created without storing source information."
            )

        logger.debug("Phase matrix recompute @ r0=%.4f µm", r0_um)

        from .refindex import ARIADatabase

        db = ARIADatabase()
        ri = db.read(self.ri_path, wavenumber=self.wavenumber, out_of_range='clip')

        dist = _make_dist(self.dist_type, r0_um * 1e-4, self.s)
        xsc = AerosolXSC(ri=ri, dist=dist, wn=self.wavenumber, density=self.density)
        xsc.compute(theta_pts=_theta_pts, n_legendre=_n_legendre, progress=False)

        result = {
            'phase_matrix': xsc.result.phase_matrix,
            'pmom': xsc.result.pmom,
            'theta': xsc.theta,
        }

        self._phase_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # RFM export
    # ------------------------------------------------------------------

    def to_rfm_xsc(
        self,
        path:  str | Path,
        r0_um: float,
        **kwargs,
    ) -> None:
        """
        Forward-evaluate the LUT at ``r0_um`` and export to a HITRAN ``.xsc``
        file for RFM.

        Equivalent to calling :meth:`forward`, wrapping the result in a
        temporary :class:`~mieEODG.aerosol.AerosolXSC`-like object, then
        passing it to :func:`~mieEODG.rfm.write_rfm_xsc`.

        Parameters
        ----------
        path : str or Path
            Output file path (use ``.xsc`` extension).
        r0_um : float
            Median radius [µm] at which to evaluate the LUT.
        **kwargs
            Passed to :func:`~mieEODG.rfm.write_rfm_xsc`:
            ``convention``, ``concentration``, ``quantity``, ``T_K``,
            ``P_torr``.

        Notes
        -----
        Multi-tabulation export (one tabulation per temperature node) will be
        supported once the 2-D LUT (temperature sweep) is implemented.
        """
        from .rfm import write_rfm_xsc

        spec = self.forward(r0_um)

        # Build a lightweight proxy that write_rfm_xsc can query
        class _LUTProxy:
            def __init__(self, lut, spec, r0_um):
                self.wn      = lut.wavenumber
                self.MEC     = spec['MEC']
                self.MSC     = spec['MSC']
                self.MAC     = spec['MAC']
                self.ri      = lut._ri_proxy(r0_um)
                self.result  = True   # sentinel — signals compute() was called

            def __getattr__(self, name):
                raise RuntimeError(
                    f"'{name}' is not available on a LUT-derived proxy."
                )

        proxy = _LUTProxy(self, spec, r0_um)
        write_rfm_xsc(proxy, path, **kwargs)

    def _ri_proxy(self, r0_um: float):
        """Return a lightweight object exposing ri.substance and ri.T_K."""
        class _RI:
            pass
        ri = _RI()
        ri.substance = f"{self.substance} r0={r0_um:.2f}µm" if self.substance else f"Aerosol r0={r0_um:.2f}µm"
        ri.T_K       = None
        return ri

    # ------------------------------------------------------------------
    # LUT construction
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        ri:        RefrIndex,
        r0_um:     np.ndarray,
        s:         float,
        wn:        np.ndarray,
        dist_type: str = 'log_normal',
        n_pts:     int = 100,
        progress:  bool = True,
        backend:   str = 'python',
        theta_pts: np.ndarray | None = None,
        n_legendre: int | None = None,
        cache_phase: bool = False,
    ) -> 'AerosolLUT':
        """
        Compute a LUT by running :class:`~mieEODG.AerosolXSC` on a r0 grid.

        Parameters
        ----------
        ri : RefrIndex
            Pre-loaded refractive index, already interpolated onto ``wn``.
            Must have been read via :class:`~mieEODG.refindex.ARIADatabase`
            so that ``ri.category`` and density metadata are populated.
        r0_um : array-like
            Median / mean radius grid [µm].  Log-spaced grids are recommended.
        s : float
            Size distribution spread parameter.
        wn : np.ndarray
            Wavenumber grid [cm⁻¹].
        dist_type : {'log_normal', 'gaussian'}
            Size distribution type.  Default ``'log_normal'``.
        n_pts : int
            Number of Gauss quadrature points per size integration.
        progress : bool
            Show a tqdm progress bar.  Default True.
        theta_pts : np.ndarray or None
            Scattering angles [degrees] for phase matrix computation.
            Only used if ``cache_phase=True``.
        n_legendre : int or None
            Number of Legendre terms for DISORT pmom computation.
            Only used if ``cache_phase=True``.
        cache_phase : bool
            If True and theta_pts or n_legendre is supplied, compute and
            store the phase matrix / Legendre coefficients for all r0 nodes
            in the returned LUT object. Default False (compute on demand).

        Returns
        -------
        AerosolLUT
        """
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw): return it  # type: ignore[misc]

        r0_um  = np.asarray(r0_um, dtype=np.float64)
        r0_cm  = r0_um * 1e-4
        n_r0   = len(r0_um)
        n_wn   = len(wn)

        MEC_all = np.empty((n_r0, n_wn))
        MSC_all = np.empty_like(MEC_all)
        MAC_all = np.empty_like(MEC_all)

        # Allocate phase storage if caching is requested
        phase_matrix_all = None
        pmom_all = None
        if cache_phase and (theta_pts is not None or n_legendre is not None):
            if theta_pts is not None:
                phase_matrix_all = np.empty((n_r0, n_wn, len(theta_pts), 4))
            if n_legendre is not None:
                pmom_all = np.empty((n_r0, n_wn, n_legendre))

        label = f'LUT [{dist_type}]  {ri.substance or ri.category}'
        it    = _tqdm(range(n_r0), desc=label, unit='r0') if progress else range(n_r0)

        for i in it:
            dist = _make_dist(dist_type, r0_cm[i], s)
            xsc  = AerosolXSC(ri=ri, dist=dist, wn=wn, n_pts=n_pts, backend=backend)
            xsc.compute(progress=False, theta_pts=theta_pts if cache_phase else None,
                       n_legendre=n_legendre if cache_phase else None)
            MEC_all[i] = xsc.result.MEC
            MSC_all[i] = xsc.result.MSC
            MAC_all[i] = xsc.result.MAC

            if cache_phase:
                if theta_pts is not None and xsc.result.phase_matrix is not None:
                    phase_matrix_all[i] = xsc.result.phase_matrix
                if n_legendre is not None and xsc.result.pmom is not None:
                    pmom_all[i] = xsc.result.pmom

        density = xsc.density  # same for all grid points

        return cls(
            wavenumber = wn,
            r0_um      = r0_um,
            MEC        = MEC_all,
            MSC        = MSC_all,
            MAC        = MAC_all,
            s          = s,
            dist_type  = dist_type,
            density    = density,
            category   = ri.category,
            substance  = ri.substance,
            ri_path    = ri.filepath,
            theta_pts  = theta_pts,
            n_legendre = n_legendre,
            cache_phase = cache_phase,
            phase_matrix_full = phase_matrix_all,
            pmom_full = pmom_all,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the LUT to a NetCDF4 file via xarray.

        Parameters
        ----------
        path : str or Path
            Output ``.nc`` file path.  Parent directories are created
            automatically.

        Raises
        ------
        ImportError
            If xarray is not installed.
        """
        if not _HAS_XARRAY:
            raise ImportError(
                "Saving AerosolLUT requires xarray: pip install xarray"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import xarray as xr

        data_vars = {
            'MEC': xr.DataArray(
                self.MEC,
                dims=('r0', 'wavenumber'),
                attrs={'units': 'cm2/g', 'long_name': 'Mass extinction coefficient'},
            ),
            'MSC': xr.DataArray(
                self.MSC,
                dims=('r0', 'wavenumber'),
                attrs={'units': 'cm2/g', 'long_name': 'Mass scattering coefficient'},
            ),
            'MAC': xr.DataArray(
                self.MAC,
                dims=('r0', 'wavenumber'),
                attrs={'units': 'cm2/g', 'long_name': 'Mass absorption coefficient'},
            ),
        }

        # Optionally store phase data if cached
        if self.cache_phase and self.phase_matrix_full is not None:
            data_vars['phase_matrix'] = xr.DataArray(
                self.phase_matrix_full,
                dims=('r0', 'wavenumber', 'theta', 'stokes'),
                attrs={'units': 'dimensionless', 'convention': 'RT normalized'},
            )

        if self.cache_phase and self.pmom_full is not None:
            data_vars['pmom'] = xr.DataArray(
                self.pmom_full,
                dims=('r0', 'wavenumber', 'legendre'),
                attrs={'units': 'dimensionless', 'convention': 'DISORT pmom'},
            )

        attrs = {
            's':         self.s,
            'dist_type': self.dist_type,
            'density':   self.density,
            'category':  self.category,
            'substance': self.substance,
            'ri_path':   self.ri_path,
            'cache_phase': int(self.cache_phase),  # NetCDF doesn't support bool; convert to 0/1
        }

        # Store phase metadata for reconstruction at load
        if self.theta_pts is not None:
            attrs['theta_pts_len'] = len(self.theta_pts)
        if self.n_legendre is not None:
            attrs['n_legendre'] = self.n_legendre

        ds = xr.Dataset(
            data_vars,
            coords={
                'r0':         xr.DataArray(self.r0_um, dims='r0',
                                           attrs={'units': 'um', 'long_name': 'Median/mean radius'}),
                'wavenumber': xr.DataArray(self.wavenumber, dims='wavenumber',
                                           attrs={'units': 'cm-1'}),
            },
            attrs=attrs,
        )

        try:
            import netCDF4  # noqa: F401
            encoding = {v: {'zlib': True, 'complevel': 4} for v in ('MEC', 'MSC', 'MAC')}
            if self.phase_matrix_full is not None:
                encoding['phase_matrix'] = {'zlib': True, 'complevel': 4}
            if self.pmom_full is not None:
                encoding['pmom'] = {'zlib': True, 'complevel': 4}
            ds.to_netcdf(path, engine='netcdf4', encoding=encoding)
        except ImportError:
            logger.warning("netCDF4 not available — saving without compression (scipy backend)")
            ds.to_netcdf(path, engine='scipy')

        file_info = f"{len(self.r0_um)} r0 × {len(self.wavenumber)} wn"
        if self.cache_phase:
            file_info += " [with phase]"
        logger.info("LUT saved → %s  (%s)", path, file_info)

    @classmethod
    def load(cls, path: str | Path) -> 'AerosolLUT':
        """
        Load a LUT from a NetCDF4 file and precompute interpolators.

        Parameters
        ----------
        path : str or Path
            Path to a ``.nc`` file written by :meth:`save`.

        Returns
        -------
        AerosolLUT
            Fully initialised instance with PCHIP interpolators ready.
        """
        if not _HAS_XARRAY:
            raise ImportError(
                "Loading AerosolLUT requires xarray: pip install xarray"
            )
        import xarray as xr
        path = Path(path)
        ds   = xr.load_dataset(path)

        # Extract phase metadata
        cache_phase = bool(ds.attrs.get('cache_phase', False))
        theta_pts_len = int(ds.attrs.get('theta_pts_len', 0))
        n_legendre = int(ds.attrs.get('n_legendre', 0)) or None

        # Reconstruct theta_pts if it was stored
        theta_pts = None
        if theta_pts_len > 0:
            # Uniform grid from 0 to 180 degrees
            theta_pts = np.linspace(0, 180, theta_pts_len)

        lut = cls(
            wavenumber = ds['wavenumber'].values,
            r0_um      = ds['r0'].values,
            MEC        = ds['MEC'].values,
            MSC        = ds['MSC'].values,
            MAC        = ds['MAC'].values,
            s          = float(ds.attrs['s']),
            dist_type  = str(ds.attrs['dist_type']),
            density    = float(ds.attrs['density']),
            category   = str(ds.attrs.get('category', '')),
            substance  = str(ds.attrs.get('substance', '')),
            ri_path    = str(ds.attrs.get('ri_path', '')),
            theta_pts  = theta_pts,
            n_legendre = n_legendre,
            cache_phase = cache_phase,
            phase_matrix_full = ds['phase_matrix'].values if 'phase_matrix' in ds else None,
            pmom_full = ds['pmom'].values if 'pmom' in ds else None,
        )
        file_info = f"{len(lut.r0_um)} r0 × {len(lut.wavenumber)} wn"
        if cache_phase:
            file_info += " [with phase]"
        logger.info(
            "LUT loaded ← %s  (%s, %s)",
            path, file_info, lut.dist_type,
        )
        return lut

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        ri:        RefrIndex,
        r0_um:     np.ndarray,
        s:         float,
        wn:        np.ndarray,
        dist_type: str,
        n_pts:     int,
        theta_pts: np.ndarray | None = None,
        n_legendre: int | None = None,
        cache_phase: bool = False,
    ) -> str:
        """
        Compute a SHA-256 cache key from the parameters that uniquely define
        a LUT.  Backend is excluded — both backends produce identical results.
        Phase parameters are included so different phase configs create
        different cache entries.

        Returns
        -------
        str
            64-character hex digest.
        """
        r0_um = np.asarray(r0_um, dtype=np.float64)
        wn    = np.asarray(wn,    dtype=np.float64)

        # Hash theta_pts if present
        theta_pts_hash = None
        if theta_pts is not None and cache_phase:
            theta_pts_arr = np.asarray(theta_pts, dtype=np.float64)
            theta_pts_hash = hashlib.md5(theta_pts_arr.tobytes()).hexdigest()

        payload = json.dumps({
            'ri_path':   str(ri.filepath),
            'dist_type': dist_type,
            's':         float(s),
            'r0_min':    float(r0_um[0]),
            'r0_max':    float(r0_um[-1]),
            'n_r0':      int(len(r0_um)),
            'wn_min':    float(wn[0]),
            'wn_max':    float(wn[-1]),
            'n_wn':      int(len(wn)),
            'n_pts':     int(n_pts),
            'theta_pts_hash': theta_pts_hash,
            'n_legendre': int(n_legendre) if n_legendre is not None else None,
            'cache_phase': bool(cache_phase),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load_or_compute(
        cls,
        ri:        RefrIndex,
        r0_um:     np.ndarray,
        s:         float,
        wn:        np.ndarray,
        dist_type: str = 'log_normal',
        n_pts:     int = 100,
        progress:  bool = True,
        backend:   str = 'python',
        theta_pts: np.ndarray | None = None,
        n_legendre: int | None = None,
        cache_phase: bool = False,
        cache_dir: str | Path | None = None,
        recompute: bool = False,
    ) -> 'AerosolLUT':
        """
        Return a LUT from the local cache if one exists for the given
        parameters, otherwise compute it, cache it, and return it.

        The cache key is a SHA-256 hash of ``ri.filepath``, ``dist_type``,
        ``s``, the r0 and wavenumber grids, ``n_pts``, and phase parameters.
        Any change in these parameters automatically triggers a recompute.
        Backend choice does not affect the key.

        Parameters
        ----------
        ri : RefrIndex
            Pre-loaded refractive index (must have ``filepath`` populated).
        r0_um : array-like
            Median radius grid [µm].
        s : float
            Size distribution spread parameter.
        wn : np.ndarray
            Wavenumber grid [cm⁻¹].
        dist_type : {'log_normal', 'gaussian'}
            Size distribution type.
        n_pts : int
            Quadrature points per size integration.
        progress : bool
            Show tqdm progress bar during compute.
        backend : {'python', 'fortran'}
            Mie kernel to use during compute.
        theta_pts : np.ndarray or None
            Scattering angles [degrees] for phase matrix computation.
            If supplied with ``cache_phase=True``, the phase matrix is
            computed and stored in the LUT cache. Default None.
        n_legendre : int or None
            Number of Legendre terms for DISORT pmom computation.
            If supplied with ``cache_phase=True``, Legendre coefficients are
            computed and stored in the LUT cache. Default None.
        cache_phase : bool
            If True and theta_pts or n_legendre is supplied, compute and
            cache phase data. If False (default), phase data is recomputed
            on demand at query time. Default False.
        cache_dir : str, Path, or None
            Directory for cached ``.nc`` files.  Defaults to
            ``~/.mieEODG/lut_cache/``.
        recompute : bool
            If True, ignore any existing cached file and recompute from
            scratch, overwriting the cache entry.  Default False.

        Returns
        -------
        AerosolLUT
        """
        cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        key      = cls._cache_key(ri, r0_um, s, wn, dist_type, n_pts,
                                   theta_pts, n_legendre, cache_phase)
        nc_path  = cache_dir / f'{key}.nc'
        tag      = ri.substance or ri.category or 'unknown'

        if nc_path.exists() and not recompute:
            logger.info("LUT cache hit  [%s]  ← %s", tag, nc_path.name)
            lut = cls.load(nc_path)
            # Refresh metadata from the live ri object in case the cache
            # was written before a category/substance fix (not part of the
            # hash — computation is unchanged).
            lut.category  = ri.category
            lut.substance = ri.substance or ""
            return lut

        if nc_path.exists() and recompute:
            logger.info("LUT forced recompute [%s] — overwriting %s", tag, nc_path.name)
        else:
            logger.info("LUT cache miss [%s] — computing …", tag)

        lut = cls.compute(
            ri         = ri,
            r0_um      = r0_um,
            s          = s,
            wn         = wn,
            dist_type  = dist_type,
            n_pts      = n_pts,
            progress   = progress,
            backend    = backend,
            theta_pts  = theta_pts,
            n_legendre = n_legendre,
            cache_phase = cache_phase,
        )
        lut.save(nc_path)
        logger.info("LUT cached     [%s]  → %s", tag, nc_path.name)
        return lut

    @classmethod
    def list_cache(cls, cache_dir: str | Path | None = None) -> None:
        """
        Print a summary table of all LUTs in the cache directory.

        Parameters
        ----------
        cache_dir : str, Path, or None
            Cache directory to inspect.  Defaults to ``~/.mieEODG/lut_cache/``.
        """
        cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

        if not cache_dir.exists():
            print(f'Cache directory does not exist: {cache_dir}')
            return

        nc_files = sorted(cache_dir.glob('*.nc'))
        if not nc_files:
            print(f'Cache is empty: {cache_dir}')
            return

        header = (
            f'{"#":<4} {"substance":<30} {"dist":<12} {"s":>5} '
            f'{"N_r0":>5} {"r0 range (µm)":<18} '
            f'{"N_wn":>5} {"wn range (cm⁻¹)":<22} {"file"}'
        )
        print(f'\nCache directory: {cache_dir}')
        print(header)
        print('─' * len(header))

        for i, path in enumerate(nc_files):
            try:
                lut = cls.load(path)
                tag    = lut.substance or lut.category or '—'
                r0_rng = f'{lut.r0_um[0]:.3g}–{lut.r0_um[-1]:.3g}'
                wn_rng = f'{lut.wavenumber[0]:.0f}–{lut.wavenumber[-1]:.0f}'
                print(
                    f'{i+1:<4} {tag:<30} {lut.dist_type:<12} {lut.s:>5.2f} '
                    f'{len(lut.r0_um):>5} {r0_rng:<18} '
                    f'{len(lut.wavenumber):>5} {wn_rng:<22} {path.name}'
                )
            except Exception as exc:
                print(f'{i+1:<4} {"(unreadable)":<30}  {path.name}  [{exc}]')

        print()

    # ------------------------------------------------------------------
    # TODO: Multi-dimensional LUT redesign
    # ------------------------------------------------------------------
    #
    # The current implementation is strictly 1-D: it sweeps over r0 for a
    # fixed refractive index (composition and temperature).  A generalised
    # N-D LUT is planned to support:
    #
    #   - r0 × composition  (e.g. volcanic ash SiO₂ fraction)
    #   - r0 × temperature  (e.g. ice at varying T_K)
    #   - composition × temperature  (e.g. H₂SO₄ wt% × T for liquid aerosols)
    #
    # Key design decisions to resolve before implementing:
    #
    #   1. LUTAxis dataclass — name, values, units, log_interp flag.
    #   2. Arrays become (*axis_dims, N_wn); forward(**kwargs) takes one
    #      keyword per axis (e.g. lut.forward(r0_um=1.2, composition=0.72)).
    #   3. Interpolation: PchipInterpolator (1-D) →
    #      RegularGridInterpolator with method='pchip' (N-D, scipy ≥ 1.9).
    #   4. Compute loop: r0 axis varies the size distribution; composition
    #      and temperature axes vary the RefrIndex (one ri per node).
    #      For a 2-D r0 × composition grid, ri is a 1-D list aligned with
    #      the composition axis.
    #   5. Continuous composition interpolation vs discrete measured nodes.
    #   6. Cache key must encode all axis definitions and all ri source paths.
    #
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r_lo, r_hi = self.r0_um[0], self.r0_um[-1]
        return (
            f"AerosolLUT(\n"
            f"  substance = '{self.substance or self.category}',\n"
            f"  dist_type = '{self.dist_type}',  s = {self.s},\n"
            f"  r0        = [{r_lo:.3g}, {r_hi:.3g}] µm  ({len(self.r0_um)} nodes),\n"
            f"  wn        = [{self.wavenumber[0]:.1f}, {self.wavenumber[-1]:.1f}] cm⁻¹"
            f"  ({len(self.wavenumber)} pts),\n"
            f"  density   = {self.density:.4f} g/cm³\n"
            f")"
        )
