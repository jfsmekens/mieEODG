"""
Multi-dimensional LUT with source-consistent ARIA database integration.

This module implements N-dimensional look-up tables (LUTs) for aerosol optical
properties with **single-source consistency enforcement**. A MultiDimLUT sweep
must load all RI data from exactly one ARIA study to prevent mixing measurement
techniques, wavelength ranges, and processing methods across interpolation axes.

Key classes:
- LUTAxis: defines one sweep dimension (r0, composition, temperature)
- RIConstraint: specifies substance, source, and which axes vary the RI
- MultiDimLUT: N-D LUT with source validation and audit-driven constraints

Architecture
============

Each MultiDimLUT is locked to a single source at construction time:

    lut = MultiDimLUT.from_aria_database(
        substance='Sulphuric acid',
        source='Myhre 2003',      # ← enforced immutably
        axes=[r0_axis, composition_axis, temperature_axis],
        wavenumber=wn,
    )

The sweep axes must correspond to audit-verified coverage in aria_metadata_registry.json.
Any attempt to sweep over an unsupported dimension or source raises ValueError.

Configuration
=============

aria_metadata_registry.json specifies canonical sources per substance:

    {
      "Sulphuric acid": {
        "preferred_source": "Myhre 2003",
        "composition_points": [38, 48, 58, 65, 72],
        "temperature_points": [203, 208, ..., 298],
        "sweep_axes": ["composition", "temperature_K"],
        ...
      }
    }

Usage (Phase 1 — constructor & validation)
===========================================

    import numpy as np
    from mieEODG.lut_multi import LUTAxis, MultiDimLUT

    axes = [
        LUTAxis('r0', values=np.logspace(-1, 1, 20), units='µm',
                log_spacing=True, fixed_for_ri_load=True),
        LUTAxis('composition', values=np.array([38, 48, 58, 65, 72]), units='wt%',
                fixed_for_ri_load=False),
        LUTAxis('temperature_K', values=np.array([203, 223, 243, 263, 283, 298]), units='K',
                fixed_for_ri_load=False),
    ]

    lut = MultiDimLUT.from_aria_database(
        substance='Sulphuric acid',
        source='Myhre 2003',
        axes=axes,
        wavenumber=wn,
        quantity='MEC',
    )

    # lut.axes, lut.ri_constraint, lut.shape are set; data is None until compute()
    # lut.ri_constraint.source is immutable; guarantees consistent source

Notes
=====

- Phase 1 implements dataclass definitions and constraint validation.
- Phase 2 will add ARIA database loader and compute() method.
- Phase 3 will add N-D interpolation (RegularGridInterpolator) and forward().
- Extended validation: sweep values must fall within audit-verified grids (warning if extrapolated).

"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Load metadata registry at module level
_REGISTRY_PATH = Path(__file__).parent / 'data' / 'aria_metadata_registry.json'
try:
    with open(_REGISTRY_PATH, 'r') as f:
        _ARIA_METADATA = json.load(f)
except FileNotFoundError as e:
    raise ImportError(
        f"aria_metadata_registry.json not found at {_REGISTRY_PATH}. "
        "Ensure the ARIA audit has been completed."
    ) from e

logger.info(f"Loaded ARIA metadata registry ({_ARIA_METADATA['database_info']['total_files']} files)")


# ---------------------------------------------------------------------------
# LUTAxis: dimension specification
# ---------------------------------------------------------------------------

@dataclass
class LUTAxis:
    """
    Defines one dimension of a multi-dimensional LUT.

    Parameters
    ----------
    name : str
        Dimension name. Common values: 'r0' (radius), 'composition' (wt%),
        'temperature_K' (temperature).
    values : np.ndarray
        Grid points for this dimension. Must be unique.
    units : str
        Physical unit (e.g., 'µm', 'wt%', 'K').
    log_spacing : bool, optional
        True if log-spaced (typical for r0), False if linear (typical for T, C).
        Default: False.
    fixed_for_ri_load : bool, optional
        True if this axis does not require loading different RI files (e.g., r0
        is fixed; only T and C are loaded from ARIA). False if each grid point
        on this axis requires a separate RI load. Default: False.

    Attributes
    ----------
    values : np.ndarray
        Grid points as float64 array. Validated to have no duplicates.

    Raises
    ------
    ValueError
        If values contains duplicates.
    """

    name: str
    values: np.ndarray
    units: str
    log_spacing: bool = False
    fixed_for_ri_load: bool = False

    def __post_init__(self):
        """Validate and normalize values."""
        self.values = np.asarray(self.values, dtype=np.float64)
        if len(self.values) == 0:
            raise ValueError(f"Axis {self.name!r} must have at least one value")
        if len(np.unique(self.values)) != len(self.values):
            raise ValueError(f"Axis {self.name!r} has duplicate values: {self.values}")
        logger.debug(f"LUTAxis created: {self.name} ({len(self.values)} points, {self.units})")


# ---------------------------------------------------------------------------
# RIConstraint: source consistency enforcement
# ---------------------------------------------------------------------------

@dataclass
class RIConstraint:
    """
    Specifies where and how to load RI files from the ARIA database.

    Enforces single-source consistency: all RI files for a given LUT sweep must
    come from exactly one study/source. This prevents silent data corruption from
    mixing measurement techniques across interpolation axes.

    Parameters
    ----------
    substance : str
        Aerosol substance name. Must match a key in aria_metadata_registry.json.
        Examples: 'Sulphuric acid', 'HNO3', 'Water ice'.
    source : str
        Canonical ARIA study source. Must be 'preferred_source' or listed in
        'alternatives' for the substance. Examples: 'Myhre 2003', 'Biermann 2000'.
        Once set, this constraint is **immutable** across the lifetime of the LUT.
    sweep_axes : list[str]
        Names of LUT axes that vary the RI (i.e., require different RI files).
        Fixed axes like 'r0' should NOT appear here. Example: ['composition', 'temperature_K'].
    available_coverage : dict, optional
        Maps each sweep_axis to the audit-verified grid points available from
        this source. Loaded from aria_metadata_registry.json automatically if
        not provided. Example: {'composition': [38, 48, 58, 65, 72],
        'temperature_K': [203, 208, ..., 298]}.

    Attributes
    ----------
    substance : str
        (immutable) Substance name.
    source : str
        (immutable) ARIA study source.
    sweep_axes : list[str]
        Which dimensions vary the RI.
    available_coverage : dict
        Audit-verified grid per axis.

    Raises
    ------
    ValueError
        If substance or source is not found in aria_metadata_registry.json.
    """

    substance: str
    source: str
    sweep_axes: list[str]
    available_coverage: dict = field(default_factory=dict)

    def __post_init__(self):
        """Load and validate metadata from registry."""
        # Validate substance
        if self.substance not in _ARIA_METADATA['canonical_sources']:
            available = list(_ARIA_METADATA['canonical_sources'].keys())
            raise ValueError(
                f"Substance {self.substance!r} not in ARIA metadata registry. "
                f"Available: {available}"
            )

        metadata = _ARIA_METADATA['canonical_sources'][self.substance]

        # Validate source
        canonical = metadata.get('preferred_source')
        alternatives = metadata.get('alternatives', [])
        valid_sources = [canonical] + alternatives if canonical else alternatives

        if self.source not in valid_sources:
            raise ValueError(
                f"Source {self.source!r} not available for {self.substance}. "
                f"Valid sources: {valid_sources}"
            )

        # Build available_coverage from metadata if not provided
        if not self.available_coverage:
            coverage = {}
            if 'composition_points' in metadata:
                coverage['composition'] = metadata['composition_points']
            if 'temperature_points' in metadata:
                coverage['temperature_K'] = metadata['temperature_points']
            self.available_coverage = coverage

        logger.info(
            f"RIConstraint created: {self.substance} from {self.source}, "
            f"sweep axes: {self.sweep_axes}"
        )


# ---------------------------------------------------------------------------
# MultiDimLUT: N-dimensional LUT with validation
# ---------------------------------------------------------------------------

class MultiDimLUT:
    """
    N-dimensional look-up table with source-consistent ARIA integration.

    A MultiDimLUT represents precomputed optical properties (MEC, MSC, MAC) over
    an N-dimensional parameter grid. The grid is defined by one or more LUTAxis
    objects (e.g., r0, composition, temperature). All RI data is loaded from a
    single ARIA study source, enforced at construction time.

    Parameters
    ----------
    axes : list[LUTAxis]
        LUT sweep dimensions. Order matters for later interpolation.
        Example: [r0_axis, composition_axis, temperature_axis].
    ri_constraint : RIConstraint
        Specifies substance, source, and sweep parameters.
    wavenumber : np.ndarray
        Spectral grid [cm⁻¹], shape (n_wn,).
    quantity : {'MEC', 'MSC', 'MAC'}, optional
        Which optical property to compute. Default: 'MEC'.

    Attributes
    ----------
    axes : list[LUTAxis]
        Sweep dimensions (read-only).
    ri_constraint : RIConstraint
        Source and substance constraint (immutable).
    wavenumber : np.ndarray
        Spectral grid [cm⁻¹].
    quantity : str
        Optical quantity.
    shape : tuple
        LUT data shape: (n_ax0, n_ax1, ..., n_wn).
    data : np.ndarray or None
        Precomputed optical properties. None until compute() is called.

    Methods
    -------
    from_aria_database(substance, source, axes, wavenumber, quantity='MEC')
        Class method to construct and validate a MultiDimLUT from ARIA metadata.
    _validate_constraint()
        Check that all sweep_axes are available in ARIA for this source.

    Examples
    --------
    Create a 3-D LUT for H₂SO₄ (r0 × composition × temperature):

        axes = [
            LUTAxis('r0', values=np.logspace(-1, 1, 20), units='µm',
                    log_spacing=True, fixed_for_ri_load=True),
            LUTAxis('composition', values=np.array([38, 48, 58, 65, 72]), units='wt%',
                    fixed_for_ri_load=False),
            LUTAxis('temperature_K', values=np.array([203, 223, 243, 263, 283, 298]),
                    units='K', fixed_for_ri_load=False),
        ]

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
        )

        # After Phase 2 implementation:
        # lut.compute()  # Load RIs and compute
        # spec = lut.forward(r0_um=1.5, composition=65.0, temperature_K=250)

    Notes
    -----
    Phase 1 (current) implements initialization and constraint validation.
    Phase 2 will add ARIA RI loader and compute().
    Phase 3 will add N-D interpolation (RegularGridInterpolator) and forward().
    """

    def __init__(
        self,
        axes: list[LUTAxis],
        ri_constraint: RIConstraint,
        wavenumber: np.ndarray,
        quantity: str = 'MEC',
    ):
        """
        Initialize a multi-dimensional LUT.

        Parameters
        ----------
        axes : list[LUTAxis]
            Sweep dimensions.
        ri_constraint : RIConstraint
            Source and substance specification.
        wavenumber : np.ndarray
            Spectral grid [cm⁻¹].
        quantity : {'MEC', 'MSC', 'MAC'}, optional
            Optical quantity. Default: 'MEC'.

        Raises
        ------
        ValueError
            If quantity is not recognized.
        TypeError
            If axes is not a list of LUTAxis objects.
        """
        # Validate axes
        if not isinstance(axes, list):
            raise TypeError(f"axes must be a list, got {type(axes)}")
        if len(axes) == 0:
            raise ValueError("axes must contain at least one LUTAxis")
        for i, ax in enumerate(axes):
            if not isinstance(ax, LUTAxis):
                raise TypeError(f"axes[{i}] must be LUTAxis, got {type(ax)}")

        # Validate quantity
        if quantity not in ('MEC', 'MSC', 'MAC'):
            raise ValueError(f"quantity must be 'MEC', 'MSC', or 'MAC', got {quantity!r}")

        self.axes = axes
        self.ri_constraint = ri_constraint
        self.wavenumber = np.asarray(wavenumber, dtype=np.float64)
        self.quantity = quantity

        # Shape of data array: (n_ax0, n_ax1, ..., n_wn)
        self.shape = tuple(len(ax.values) for ax in axes) + (len(self.wavenumber),)

        # Data is populated by compute()
        self.data = None

        # Interpolators (to be populated in Phase 3)
        self.interpolators = {}

        logger.info(
            f"MultiDimLUT initialized: {len(axes)} axes, shape {self.shape}, "
            f"source={ri_constraint.source}"
        )

    def _validate_constraint(self) -> bool:
        """
        Verify that the RI constraint is satisfied.

        Checks that all sweep axes correspond to available ARIA data for this
        source. Raises ValueError if any axis is not in the audit-verified coverage.

        Returns
        -------
        bool
            True if constraint is satisfied.

        Raises
        ------
        ValueError
            If any sweep axis is not available from this source for this substance.
        """
        for axis_name in self.ri_constraint.sweep_axes:
            if axis_name not in self.ri_constraint.available_coverage:
                available = list(self.ri_constraint.available_coverage.keys())
                raise ValueError(
                    f"Sweep axis {axis_name!r} not available from "
                    f"{self.ri_constraint.source} for {self.ri_constraint.substance}. "
                    f"Available axes: {available}"
                )

        logger.debug(
            f"Constraint validated: {self.ri_constraint.substance} from "
            f"{self.ri_constraint.source} supports sweep axes {self.ri_constraint.sweep_axes}"
        )
        return True

    @classmethod
    def from_aria_database(
        cls,
        substance: str,
        source: str,
        axes: list[LUTAxis],
        wavenumber: np.ndarray,
        quantity: str = 'MEC',
    ) -> MultiDimLUT:
        """
        Construct a MultiDimLUT by loading metadata and constraints from ARIA database.

        This class method enforces single-source consistency at construction time.
        The LUT is locked to the specified source and cannot be changed without
        creating a new LUT.

        Parameters
        ----------
        substance : str
            Aerosol substance. Must be in aria_metadata_registry.json.
            Examples: 'Sulphuric acid', 'HNO3', 'Water ice'.
        source : str
            ARIA study source. Must be canonical or listed as alternative for the
            substance. Examples: 'Myhre 2003', 'Biermann 2000', 'Curtis 2005'.
        axes : list[LUTAxis]
            LUT sweep dimensions.
        wavenumber : np.ndarray
            Spectral grid [cm⁻¹].
        quantity : {'MEC', 'MSC', 'MAC'}, optional
            Optical quantity. Default: 'MEC'.

        Returns
        -------
        MultiDimLUT
            Newly constructed LUT with ri_constraint set and constraint validated.

        Raises
        ------
        ValueError
            If substance or source not in registry, or if sweep axes are not
            available from the specified source.

        Examples
        --------
        >>> axes = [
        ...     LUTAxis('r0', ..., fixed_for_ri_load=True),
        ...     LUTAxis('composition', ..., fixed_for_ri_load=False),
        ...     LUTAxis('temperature_K', ..., fixed_for_ri_load=False),
        ... ]
        >>> lut = MultiDimLUT.from_aria_database(
        ...     'Sulphuric acid', 'Myhre 2003', axes, wn
        ... )
        >>> lut.ri_constraint.source  # 'Myhre 2003' (immutable)
        """
        # Create constraint (validates substance and source)
        sweep_axes = [ax.name for ax in axes if not ax.fixed_for_ri_load]
        ri_constraint = RIConstraint(
            substance=substance,
            source=source,
            sweep_axes=sweep_axes,
        )

        # Create LUT instance
        lut = cls(axes, ri_constraint, wavenumber, quantity)

        # Validate that all sweep axes are available
        lut._validate_constraint()

        logger.info(
            f"MultiDimLUT.from_aria_database: {substance} ({source}), "
            f"{len(axes)} axes, shape {lut.shape}"
        )

        return lut

    def compute(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        progress: bool = True,
        backend: str = 'python',
    ) -> None:
        """
        Compute optical properties for all grid points.

        Loads RI files from ARIA database for each grid point on the sweep axes,
        computes optical properties (MEC, MSC, MAC), and populates ``self.data``.

        Parameters
        ----------
        progress_callback : Callable[[int, int], None] or None, optional
            Callback function called at each grid point: ``callback(step, total)``.
            If provided, suppresses tqdm output. Default: None (use tqdm if
            progress=True).
        progress : bool, optional
            If True (default) and progress_callback is None, show a tqdm
            progress bar. No effect if progress_callback is provided.
        backend : {'python', 'fortran'}, optional
            Computational backend for Mie calculations. 'python' (default) uses
            pure NumPy; 'fortran' uses compiled extension (~40× faster).
            Default: 'python'.

        Raises
        ------
        RuntimeError
            If an RI file cannot be loaded or an optical computation fails.
        """
        from .refindex import ARIADatabase
        from .aerosol import AerosolXSC
        from .size_dist import LogNormalDist

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Validate constraint before computing
        self._validate_constraint()

        # Initialize data array
        self.data = np.zeros(self.shape, dtype=np.float64)

        db = ARIADatabase()

        # Iterate over all grid points
        # Shape is (n_ax0, n_ax1, ..., n_wn) so we need to iterate over all
        # combinations of axis values
        axis_indices = np.ndindex(tuple(len(ax.values) for ax in self.axes))

        # Calculate total steps for progress reporting
        total_steps = np.prod([len(ax.values) for ax in self.axes])
        step = 0

        pbar = None
        if progress and progress_callback is None and tqdm is not None:
            pbar = tqdm(total=total_steps, desc="Computing LUT", unit="node")

        try:
            for idx in axis_indices:
                step += 1

                # Update progress
                if progress_callback is not None:
                    progress_callback(step, total_steps)
                elif pbar is not None:
                    pbar.update(1)

                # Extract values for this grid point
                grid_values = {self.axes[i].name: self.axes[i].values[idx[i]]
                               for i in range(len(self.axes))}

                logger.debug(f"  Computing grid point {step}/{total_steps}: {grid_values}")

                # Load RI for sweep axes only
                # Fixed axes (fixed_for_ri_load=True) are not used to query ARIA
                try:
                    ri = self._load_ri_for_point(db, grid_values)
                except FileNotFoundError as e:
                    logger.error(f"Failed to load RI for {grid_values}: {e}")
                    # Skip this grid point or use a sentinel value?
                    # For now, fill with NaN
                    self.data[idx] = np.nan
                    continue

                # Create size distribution with r0 from grid
                r0_um = grid_values.get('r0', None)
                if r0_um is None:
                    # r0 is not in sweep_axes, should be fixed_for_ri_load=True
                    # This shouldn't happen in Phase 2 (r0 is always fixed)
                    logger.warning("r0 not found in grid point")
                    self.data[idx] = np.nan
                    continue

                # Use default size distribution (LogNormal with s=1.5)
                dist = LogNormalDist(r0=r0_um * 1e-4, s=1.5, n=1.0)  # r0 in cm

                # Compute optical properties
                try:
                    xsc = AerosolXSC(
                        ri=ri,
                        dist=dist,
                        wn=self.wavenumber,
                        backend=backend,
                    )
                    xsc.compute(progress=False)

                    # Extract the requested quantity
                    if self.quantity == 'MEC':
                        quantity_values = xsc.result.MEC
                    elif self.quantity == 'MSC':
                        quantity_values = xsc.result.MSC
                    elif self.quantity == 'MAC':
                        quantity_values = xsc.result.MAC
                    else:
                        raise ValueError(f"Unknown quantity: {self.quantity}")

                    # Store in data array
                    self.data[idx] = quantity_values

                except Exception as e:
                    logger.error(f"Failed to compute XSC for {grid_values}: {e}")
                    self.data[idx] = np.nan

        finally:
            if pbar is not None:
                pbar.close()

        logger.info(f"LUT computation complete: {self.quantity} array shape {self.shape}")

    def forward(self, **kwargs) -> dict[str, np.ndarray]:
        """
        Query the LUT for optical properties at a specific grid point.

        Uses N-dimensional interpolation (RegularGridInterpolator) to interpolate
        across all axes. Returns the full spectral array at the queried point.

        Parameters
        ----------
        **kwargs
            Grid point specification. Must include one value for each LUT axis
            (except wavenumber, which is always returned in full). Axis names
            are drawn from the ``axes`` list. Example:
            ``lut.forward(r0_um=1.5, composition=65.0, temperature_K=250)``.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'MEC', 'MSC', 'MAC' (or whichever quantity was
            specified at construction). Values are spectral arrays of shape
            ``(n_wavenumber,)``, interpolated to the requested point.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called (data is None).
        ValueError
            If required axes are missing from kwargs, or if interpolation point
            is out of bounds.
        """
        if self.data is None:
            raise RuntimeError("Cannot call forward() before compute()")

        # Build interpolator if not already done
        if not self.interpolators:
            self._build_interpolator()

        # Extract query point from kwargs
        query_point = []
        for ax in self.axes:
            if ax.name not in kwargs:
                raise ValueError(f"Missing required axis: {ax.name!r}")
            query_point.append(kwargs[ax.name])

        query_point = np.array([query_point])  # RegularGridInterpolator expects (1, n_axes)

        # Interpolate for each wavenumber
        # RegularGridInterpolator returns shape (n_query_points, ..., n_data_dims)
        # We need to handle the spectral dimension carefully
        result_spectrum = np.zeros(len(self.wavenumber), dtype=np.float64)

        for i_wn in range(len(self.wavenumber)):
            # Get the 2-D slice of data for this wavenumber (across all grid axes)
            data_slice = self.data[..., i_wn]

            # Interpolate this slice
            interp = RegularGridInterpolator(
                points=tuple(ax.values for ax in self.axes),
                values=data_slice,
                bounds_error=False,
                fill_value=np.nan,
            )
            result_spectrum[i_wn] = interp(query_point)[0]

        logger.debug(
            f"LUT forward query: {kwargs}, spectrum shape {result_spectrum.shape}"
        )

        return {self.quantity: result_spectrum}

    def _build_interpolator(self) -> None:
        """
        Precompute and cache RegularGridInterpolator objects.

        For efficiency, we could cache one interpolator per wavenumber, but for
        Phase 3 we'll use on-demand interpolation in forward() to keep memory usage
        reasonable. This method is a placeholder for future optimization.
        """
        # Placeholder for future optimization (per-wavenumber caching)
        pass

    def save(self, path: str | Path) -> None:
        """
        Save the computed LUT to a NetCDF file.

        Parameters
        ----------
        path : str or Path
            Output file path. Should use ``.nc`` extension.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called.
        ImportError
            If xarray/netCDF4 is not installed.
        """
        if self.data is None:
            raise RuntimeError("Cannot save before compute() is called")

        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "NetCDF persistence requires xarray and netCDF4. "
                "Install with: pip install xarray netCDF4"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build a dataset with dimensions and coordinates
        coords = {ax.name: ax.values for ax in self.axes}
        coords['wavenumber'] = self.wavenumber

        # Data array with all dimensions
        dim_names = tuple(ax.name for ax in self.axes) + ('wavenumber',)
        data_array = xr.DataArray(
            self.data,
            coords=coords,
            dims=dim_names,
            name=self.quantity,
        )

        # Add metadata as attributes
        ds = data_array.to_dataset()
        ds.attrs['quantity'] = self.quantity
        ds.attrs['substance'] = self.ri_constraint.substance
        ds.attrs['source'] = self.ri_constraint.source
        ds.attrs['sweep_axes'] = ','.join(self.ri_constraint.sweep_axes)

        # Store axis metadata
        for ax in self.axes:
            ds.attrs[f'axis_{ax.name}_units'] = ax.units
            ds.attrs[f'axis_{ax.name}_log_spacing'] = ax.log_spacing
            ds.attrs[f'axis_{ax.name}_fixed_for_ri_load'] = ax.fixed_for_ri_load

        # Save to NetCDF
        ds.to_netcdf(path, engine='netcdf4')
        logger.info(f"LUT saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'MultiDimLUT':
        """
        Load a previously saved LUT from a NetCDF file.

        Parameters
        ----------
        path : str or Path
            Path to the NetCDF file.

        Returns
        -------
        MultiDimLUT
            Reconstructed LUT with data populated.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If xarray/netCDF4 is not installed.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "NetCDF persistence requires xarray and netCDF4. "
                "Install with: pip install xarray netCDF4"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LUT file not found: {path}")

        ds = xr.open_dataset(path, engine='netcdf4')

        # Reconstruct axes from coordinates
        quantity = ds.attrs.get('quantity', 'MEC')
        substance = ds.attrs.get('substance')
        source = ds.attrs.get('source')
        sweep_axes = ds.attrs.get('sweep_axes', '').split(',')

        # Get data and wavenumber
        wavenumber = ds.coords['wavenumber'].values

        # Reconstruct LUTAxis objects
        axes = []
        for var_name in ds.data_vars:
            if var_name == quantity:
                continue
            # Coordinates are the axes
            pass

        # Reconstruct from coordinates
        axes = []
        coord_names = [name for name in ds.coords if name != 'wavenumber']
        for coord_name in coord_names:
            units = ds.attrs.get(f'axis_{coord_name}_units', '')
            log_spacing = ds.attrs.get(f'axis_{coord_name}_log_spacing', False)
            fixed_for_ri = ds.attrs.get(f'axis_{coord_name}_fixed_for_ri_load', False)

            ax = LUTAxis(
                name=coord_name,
                values=ds.coords[coord_name].values,
                units=units,
                log_spacing=log_spacing,
                fixed_for_ri_load=fixed_for_ri,
            )
            axes.append(ax)

        # Create RIConstraint
        ri_constraint = RIConstraint(
            substance=substance,
            source=source,
            sweep_axes=[name for name in coord_names if name in sweep_axes],
        )

        # Create LUT instance
        lut = cls(axes, ri_constraint, wavenumber, quantity=quantity)

        # Load data
        lut.data = ds[quantity].values

        logger.info(f"LUT loaded from {path}: {substance} ({source}), shape {lut.shape}")
        ds.close()

        return lut

    def _load_ri_for_point(
        self,
        db: object,  # ARIADatabase
        grid_values: dict[str, float],
    ) -> object:  # RefrIndex
        """
        Load a refractive index file matching the grid point.

        Parameters
        ----------
        db : ARIADatabase
            ARIA database instance.
        grid_values : dict[str, float]
            Grid point values, e.g. {'r0': 1.5, 'composition': 65.0, 'temperature_K': 250}.

        Returns
        -------
        RefrIndex
            Loaded RI file, interpolated to self.wavenumber.

        Raises
        ------
        FileNotFoundError
            If no matching RI file is found.
        """
        from .refindex import readRI

        substance = self.ri_constraint.substance
        source = self.ri_constraint.source

        # Extract composition and temperature from grid point
        composition_wt = grid_values.get('composition', None)
        temperature_K = grid_values.get('temperature_K', None)

        # Map substance to ARIA category and substance directory
        # This is a simplified mapping; may need to be made more robust
        substance_aria_map = {
            'Sulphuric acid': 'Sulphuric',
            'HNO3': 'Nitric acid',
            'Water ice': 'Ice',
            'Supercooled water': 'Water',
            'Water (liquid)': 'Water',
        }

        aria_substance = substance_aria_map.get(substance, substance)

        # Search ARIA for matching file
        # Pattern: source is in filename (e.g., "Myhre_2003", "Biermann_2000")
        # Composition and temperature are also in filename
        source_pattern = source.replace(' ', '_')

        candidates = db.list(
            category='Acids',  # H₂SO₄ and HNO₃ are in Acids category
            pattern=aria_substance.lower(),
        )

        # Further filter by source
        source_candidates = [
            c for c in candidates
            if source_pattern.lower() in c.lower()
        ]

        if not source_candidates:
            raise FileNotFoundError(
                f"No ARIA files found for {substance} from {source}. "
                f"Searched: {candidates}"
            )

        # Find best match based on composition and temperature
        best_match = None
        best_score = -1

        for candidate_path in source_candidates:
            # Parse filename for composition and temperature
            c_val, t_val = self._parse_comp_temp_from_path(candidate_path)

            # Score based on matching composition and temperature
            score = 0
            if composition_wt is not None and c_val is not None:
                if abs(c_val - composition_wt) < 1.0:  # Within 1% tolerance
                    score += 10 - abs(c_val - composition_wt)
            if temperature_K is not None and t_val is not None:
                if abs(t_val - temperature_K) < 5.0:  # Within 5 K tolerance
                    score += 10 - abs(t_val - temperature_K) / 5.0

            if score > best_score:
                best_score = score
                best_match = candidate_path

        if best_match is None:
            raise FileNotFoundError(
                f"No good match found for {substance} from {source} "
                f"at composition={composition_wt}, temperature={temperature_K}"
            )

        logger.debug(f"Loading RI from: {best_match}")
        ri = db.read(
            best_match,
            wavenumber=self.wavenumber,
            out_of_range='clip',
        )

        return ri

    @staticmethod
    def _parse_comp_temp_from_path(path: str) -> tuple[float | None, float | None]:
        """
        Extract composition (wt%) and temperature (K) from an ARIA file path.

        Parameters
        ----------
        path : str
            Relative path to a .ri file, e.g.,
            'Acids/Sulphuric/H2SO4_48_213K_Myhre_2003.ri' or
            'Acids/Nitric/HNO340_253_Biermann_2000.ri'

        Returns
        -------
        tuple[float | None, float | None]
            (composition_wt%, temperature_K), with None for values not found.
        """
        filename = Path(path).stem  # Remove .ri extension

        composition = None
        temperature = None

        # Extract temperature: look for patterns like "213K", "253_K", or bare "253"
        # First try with K suffix
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*K(?:_|$)', filename, re.IGNORECASE)
        if temp_match:
            temperature = float(temp_match.group(1))
        else:
            # For HNO3/H2SO4 files like "HNO340_253_Biermann_2000",
            # the temperature is the second number sequence after substance+comp
            # Extract all number sequences and check if any look like temperatures (100-400 K)
            all_numbers = re.findall(r'(\d+(?:\.\d+)?)', filename)
            for num_str in all_numbers:
                num = float(num_str)
                # Reasonable temperature range: 100-400 K
                if 100 <= num <= 400:
                    # Assume this is temperature if it appears after composition numbers
                    temperature = num
                    break

        # For composition, look for patterns like "H2SO4-72" or "HNO30_263"
        # Extract the number after the substance prefix
        comp_patterns = [
            r'H2SO4[-_]?(\d+(?:\.\d+)?)',
            r'HNO3(\d+(?:\.\d+)?)',  # HNO3 has composition directly appended
            r'(?:Sulphuric|Nitric)[-_]?(\d+(?:\.\d+)?)',
            r'H2O[-_]?(\d+(?:\.\d+)?)',
            r'(?:water)[-_]?(\d+(?:\.\d+)?)',
        ]
        for pattern in comp_patterns:
            m = re.search(pattern, filename, re.IGNORECASE)
            if m:
                composition = float(m.group(1))
                break

        return composition, temperature
