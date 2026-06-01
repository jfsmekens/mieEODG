"""
AerosolLUT — precomputed look-up table of Mie mass coefficients.

A LUT is a grid of (MEC, MSC, MAC) arrays computed over a range of median
radii for a fixed refractive index, size distribution type, and spectral grid.
At load time, PCHIP interpolators are instantiated over log(r0) so that the
forward call — returning a full spectrum for a given r0 — costs only a single
binary search and a vectorised multiply across all wavenumber channels.

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

    # Inspect the local cache
    AerosolLUT.list_cache()

Cache
-----
LUTs are stored in ``~/.mieEODG/lut_cache/`` by default (override with
``cache_dir``).  Each file is named by a SHA-256 hash of the defining
parameters (ri path, dist_type, s, r0 grid, wn grid, n_pts) so that any
change in inputs automatically triggers a recompute.  Backend choice does
not affect the cache key since both backends produce identical results.
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

# Default cache location — user-level, persists across projects
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

        label = f'LUT [{dist_type}]  {ri.substance or ri.category}'
        it    = _tqdm(range(n_r0), desc=label, unit='r0') if progress else range(n_r0)

        for i in it:
            dist = _make_dist(dist_type, r0_cm[i], s)
            xsc  = AerosolXSC(ri=ri, dist=dist, wn=wn, n_pts=n_pts, backend=backend)
            xsc.compute(progress=False)
            MEC_all[i] = xsc.result.MEC
            MSC_all[i] = xsc.result.MSC
            MAC_all[i] = xsc.result.MAC

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

        ds = xr.Dataset(
            {
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
            },
            coords={
                'r0':         xr.DataArray(self.r0_um, dims='r0',
                                           attrs={'units': 'um', 'long_name': 'Median/mean radius'}),
                'wavenumber': xr.DataArray(self.wavenumber, dims='wavenumber',
                                           attrs={'units': 'cm-1'}),
            },
            attrs={
                's':         self.s,
                'dist_type': self.dist_type,
                'density':   self.density,
                'category':  self.category,
                'substance': self.substance,
                'ri_path':   self.ri_path,
            },
        )

        try:
            import netCDF4  # noqa: F401
            encoding = {v: {'zlib': True, 'complevel': 4} for v in ('MEC', 'MSC', 'MAC')}
            ds.to_netcdf(path, engine='netcdf4', encoding=encoding)
        except ImportError:
            logger.warning("netCDF4 not available — saving without compression (scipy backend)")
            ds.to_netcdf(path, engine='scipy')
        logger.info("LUT saved → %s  (%d r0 × %d wn)", path, len(self.r0_um), len(self.wavenumber))

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
        )
        logger.info(
            "LUT loaded ← %s  (%d r0 × %d wn, %s)",
            path, len(lut.r0_um), len(lut.wavenumber), lut.dist_type,
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
    ) -> str:
        """
        Compute a SHA-256 cache key from the parameters that uniquely define
        a LUT.  Backend is excluded — both backends produce identical results.

        Returns
        -------
        str
            64-character hex digest.
        """
        r0_um = np.asarray(r0_um, dtype=np.float64)
        wn    = np.asarray(wn,    dtype=np.float64)

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
        cache_dir: str | Path | None = None,
        recompute: bool = False,
    ) -> 'AerosolLUT':
        """
        Return a LUT from the local cache if one exists for the given
        parameters, otherwise compute it, cache it, and return it.

        The cache key is a SHA-256 hash of ``ri.filepath``, ``dist_type``,
        ``s``, the r0 and wavenumber grids, and ``n_pts``.  Any change in
        these parameters automatically triggers a recompute.  Backend choice
        does not affect the key.

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

        key      = cls._cache_key(ri, r0_um, s, wn, dist_type, n_pts)
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
            ri        = ri,
            r0_um     = r0_um,
            s         = s,
            wn        = wn,
            dist_type = dist_type,
            n_pts     = n_pts,
            progress  = progress,
            backend   = backend,
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
