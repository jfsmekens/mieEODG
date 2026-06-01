"""
Refractive index reader for the ARIA (Aerosol Refractive Index Archive) database.

The ARIA database is a collection of ASCII ``.ri`` files organised in a
directory hierarchy:

    <root>/<Category>/<Substance>/<Entry>/filename.ri

Each file starts with ``#KEYWORD = value`` header lines, followed by a
``#FORMAT = col1 col2 ...`` line and then numeric data.  Columns are either
wavelength-indexed (``FORMAT = WAVL N K``, wavelength in μm) or
wavenumber-indexed (``FORMAT = WAVN N K``, wavenumber in cm⁻¹).

Adapted from:
    srfm/ARIA_module.py  (EODG, Oxford; see https://github.com/eodg-oxford/srfm)

The key differences from the srfm version:
  - Fully decoupled from the srfm package; database path is a user argument.
  - Returns Cm = n − 1j·k  (negative imaginary for absorbing particles),
    consistent with the Mie sign convention used throughout mieEODG.
  - Interpolation onto a caller-supplied wavenumber grid is optional.
  - All outputs are plain NumPy arrays / dataclasses; no xarray dependency.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled ARIA path resolution
# ---------------------------------------------------------------------------

def _bundled_aria_path() -> Path:
    """
    Return the path to the ARIA database.

    Resolution order:
    1. Environment variable ``MIEODG_ARIA_ROOT`` (allows user override).
    2. ``mieEODG/data/ARIA`` bundled with the package.

    Raises
    ------
    FileNotFoundError
        If neither location contains a valid directory.
    """
    env = os.environ.get('MIEODG_ARIA_ROOT')
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"MIEODG_ARIA_ROOT is set to '{env}' but the directory was not found."
        )
    bundled = Path(__file__).parent / 'data' / 'ARIA'
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        "Bundled ARIA database not found.  "
        "Set the MIEODG_ARIA_ROOT environment variable to your local ARIA path."
    )

# Conversion factor: 1 μm ↔ 10000 cm⁻¹
_UM_TO_WAVN = 1.0e4   # wavenumber (cm⁻¹) = 10000 / wavelength (μm)


@dataclass
class RefrIndex:
    """
    Refractive index data read from an ARIA ``.ri`` file.

    Attributes
    ----------
    wavenumber : np.ndarray, shape (N,)
        Wavenumber grid [cm⁻¹], ascending order.
    wavelength : np.ndarray, shape (N,)
        Wavelength grid [μm] = 10000 / wavenumber.
    n : np.ndarray, shape (N,)
        Real part of the complex refractive index.
    k : np.ndarray, shape (N,)
        Imaginary part (extinction coefficient, >= 0).
    Cm : np.ndarray, shape (N,), dtype complex128
        Complex refractive index = n − 1j·k.
        Sign convention: negative imaginary part for absorbing particles,
        consistent with mieEODG's Mie routines.
    header : dict[str, str]
        Header metadata parsed from ``#KEYWORD = value`` lines.
    filepath : str
        Absolute path to the source ``.ri`` file.
    category : str
        Normalised ARIA category key, e.g. ``'volcanic_ash'`` or
        ``'water_and_ice/water'``.  Populated automatically when the file is
        read via :class:`ARIADatabase`; empty string for files read directly
        with :func:`readRI`.  Suitable for direct use with :func:`get_cmap`
        and :func:`get_color`.
    substance_key : str
        Normalised ARIA substance name derived from the second path component
        (citation suffixes stripped), e.g. ``'hematite'``, ``'sodium_chloride'``.
        Distinct from :attr:`substance` (which is the ``SUBSTANCE`` header field).
        Empty string if not available.
    T_K : float or None
        Temperature [K] parsed from the ``TEMPERATURE`` header field, or
        *None* if absent or not parseable.  Used by
        :func:`~mieEODG.constants.aria_density` for ice density.
        Note: :attr:`temperature` is the raw header string; ``T_K`` is the
        parsed numeric value.
    composition : dict[str, float]
        Parsed composition data from the ``CONCENTRATION`` header field.
        Keys depend on the species type:

        ``'weight_fraction'``
            Primary solute weight fraction in [0, 1], for solution-phase
            species (e.g. H₂SO₄ in water, organic acids).  Range values
            (``'55.0 - 55.2 wt%'``) are stored as the midpoint.
        ``'SiO2'``
            Silica weight fraction in [0, 1], for volcanic ash and
            rocks where CONCENTRATION encodes mineralogy rather than a
            solute concentration.

        Empty dict if the field is absent or not parseable.
        Use the :attr:`weight_fraction` convenience property to access the
        ``'weight_fraction'`` entry directly.
    """
    wavenumber: np.ndarray
    wavelength: np.ndarray
    n:          np.ndarray
    k:          np.ndarray
    Cm:         np.ndarray
    header:      dict              = field(default_factory=dict)
    filepath:    str               = ""
    category:    str               = ""
    composition:   dict[str, float]  = field(default_factory=dict)
    substance_key: str               = ""        # normalised ARIA path component (2nd level)
    T_K:           float | None      = None      # parsed TEMPERATURE header value [K]

    # Convenience properties
    @property
    def substance(self) -> str:
        """``SUBSTANCE`` field from the file header, or empty string."""
        return self.header.get('SUBSTANCE', '').strip()

    @property
    def temperature(self) -> str:
        """``TEMPERATURE`` field from the file header, or empty string."""
        return self.header.get('TEMPERATURE', '').strip()

    @property
    def reference(self) -> str:
        """``REFERENCE`` field from the file header, or empty string."""
        return self.header.get('REFERENCE', '').strip()

    @property
    def weight_fraction(self) -> float | None:
        """
        Primary solute weight fraction [0, 1], or *None* if not available.

        Shortcut for ``composition.get('weight_fraction')``.  Populated for
        solution-phase species (sulphuric acid, organic acids, etc.) where
        the ARIA ``CONCENTRATION`` header encodes a wt% value.
        """
        return self.composition.get('weight_fraction')

    def __repr__(self) -> str:
        sub  = self.substance or Path(self.filepath).stem
        n_pt = len(self.wavenumber)
        wmin = self.wavenumber[0]
        wmax = self.wavenumber[-1]
        return (
            f"RefrIndex('{sub}', {n_pt} pts, "
            f"ν=[{wmin:.1f}, {wmax:.1f}] cm⁻¹)"
        )


def readRI(
    filepath: str | Path,
    wavenumber: np.ndarray | None = None,
    extrapolate: bool = False,
    out_of_range: Literal['error', 'warn', 'clip'] = 'error',
) -> RefrIndex:
    """
    Read a single ARIA ``.ri`` file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.ri`` file.
    wavenumber : np.ndarray or None
        If supplied, interpolate n and k onto this wavenumber grid [cm⁻¹].
        Grid must be within the file's native range unless ``extrapolate``
        or ``out_of_range`` permit otherwise.
    extrapolate : bool
        If True, allow linear extrapolation beyond the file's spectral range.
        Default False.
    out_of_range : {'error', 'warn', 'clip'}
        Action when ``wavenumber`` extends beyond the file's range and
        ``extrapolate`` is False:

        ``'error'``  (default) — raise ``ValueError``.
        ``'warn'``   — log a warning and clip to file bounds.
        ``'clip'``   — silently clip to file bounds.

    Returns
    -------
    RefrIndex
        Parsed (and optionally interpolated) refractive index data.

    Notes
    -----
    The file's native abscissa (wavelength μm or wavenumber cm⁻¹) is
    converted to wavenumber internally.  Data are sorted into ascending
    wavenumber order before any interpolation.
    """
    filepath = Path(filepath).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"ARIA file not found: {filepath}")

    header: dict[str, str] = {}
    fmt_cols: list[str]    = []
    data_rows: list[list[float]] = []

    with filepath.open('r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Strip leading '#' and split on first '='
                content = line[1:].strip()
                if content.upper().startswith('FORMAT'):
                    _, _, rest = content.partition('=')
                    fmt_cols = rest.split()
                elif '=' in content:
                    key, _, val = content.partition('=')
                    header[key.strip().upper()] = val.strip()
                # Lines like '#DISTRIBUTEDBY...' without '=' are skipped
            else:
                try:
                    data_rows.append([float(v) for v in line.split()])
                except ValueError:
                    logger.debug("Skipping non-numeric line in %s: %r", filepath.name, line)

    if not fmt_cols:
        raise ValueError(f"No #FORMAT line found in {filepath}")
    if not data_rows:
        raise ValueError(f"No numeric data found in {filepath}")

    data = np.array(data_rows, dtype=np.float64)

    fmt_upper = [c.upper() for c in fmt_cols]
    try:
        absc_col = fmt_upper.index('WAVL') if 'WAVL' in fmt_upper else fmt_upper.index('WAVN')
        n_col    = fmt_upper.index('N')
        k_col    = fmt_upper.index('K')
    except ValueError as exc:
        raise ValueError(
            f"Unexpected FORMAT columns {fmt_cols!r} in {filepath}. "
            "Expected columns N and K plus one of WAVL or WAVN."
        ) from exc

    absc_raw = data[:, absc_col]
    n_raw    = data[:, n_col]
    k_raw    = data[:, k_col]

    # Convert abscissa to wavenumber [cm⁻¹], ascending
    is_wavelength = 'WAVL' in fmt_upper
    if is_wavelength:
        # wavelength in μm → wavenumber in cm⁻¹; this reverses ordering
        wavn_raw = _UM_TO_WAVN / absc_raw
    else:
        wavn_raw = absc_raw.copy()

    # Sort into ascending wavenumber order
    sort_idx = np.argsort(wavn_raw)
    wavn_raw = wavn_raw[sort_idx]
    n_raw    = n_raw[sort_idx]
    k_raw    = k_raw[sort_idx]
    wavl_raw = _UM_TO_WAVN / wavn_raw

    if wavenumber is None:
        # Return native grid
        return RefrIndex(
            wavenumber = wavn_raw,
            wavelength = wavl_raw,
            n          = n_raw,
            k          = k_raw,
            Cm         = n_raw - 1j * k_raw,
            header     = header,
            filepath   = str(filepath),
        )

    # --- Interpolate onto caller-supplied grid ---
    wavenumber = np.asarray(wavenumber, dtype=np.float64)
    wmin_file  = wavn_raw[0]
    wmax_file  = wavn_raw[-1]
    wmin_req   = float(wavenumber.min())
    wmax_req   = float(wavenumber.max())

    out_low  = wmin_req < wmin_file
    out_high = wmax_req > wmax_file

    if (out_low or out_high) and not extrapolate:
        msg = (
            f"Requested wavenumber range [{wmin_req:.1f}, {wmax_req:.1f}] cm⁻¹ "
            f"exceeds file range [{wmin_file:.1f}, {wmax_file:.1f}] cm⁻¹ "
            f"in {filepath.name}."
        )
        if out_of_range == 'error':
            raise ValueError(msg)
        elif out_of_range == 'warn':
            logger.warning(msg + "  Clipping to file bounds.")
            wavenumber = np.clip(wavenumber, wmin_file, wmax_file)
        else:  # 'clip'
            wavenumber = np.clip(wavenumber, wmin_file, wmax_file)

    fill = 'extrapolate' if extrapolate else (wmin_file, wmax_file)
    interp_kind = 'linear'
    n_interp = interp1d(wavn_raw, n_raw, kind=interp_kind,
                        bounds_error=False, fill_value=fill)(wavenumber)
    k_interp = interp1d(wavn_raw, k_raw, kind=interp_kind,
                        bounds_error=False, fill_value=fill)(wavenumber)

    # k must not go negative due to interpolation artefacts
    k_interp = np.clip(k_interp, 0.0, None)

    wavl_interp = _UM_TO_WAVN / wavenumber

    return RefrIndex(
        wavenumber = wavenumber,
        wavelength = wavl_interp,
        n          = n_interp,
        k          = k_interp,
        Cm         = n_interp - 1j * k_interp,
        header     = header,
        filepath   = str(filepath),
    )


# ---------------------------------------------------------------------------
# CONCENTRATION header parser
# ---------------------------------------------------------------------------

# Matches:  "72 wt%"  "72wt%"  "72%"  "72.5 wt%"  "55.0 - 55.2 wt%"
_RE_WT_RANGE  = re.compile(
    r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*wt%', re.I
)
_RE_WT_SINGLE = re.compile(r'(\d+(?:\.\d+)?)\s*wt%', re.I)
_RE_PCT_BARE  = re.compile(r'(\d+(?:\.\d+)?)\s*%$')   # bare % with no qualifier

# Matches:  "SiO<sub>2</sub>: 49.1%"
_RE_SIO2 = re.compile(r'SiO\s*(?:<[^>]*>.*?</[^>]*>)?\s*2?\s*[:\s]\s*(\d+(?:\.\d+)?)\s*%', re.I)


def _parse_concentration(raw: str) -> dict[str, float]:
    """
    Parse the ARIA ``CONCENTRATION`` header value into a composition dict.

    Recognised formats
    ------------------
    ``"72 wt%"`` / ``"72wt%"`` / ``"72%"``
        → ``{'weight_fraction': 0.72}``

    ``"55.0 - 55.2 wt%"``
        → ``{'weight_fraction': 0.551}``  (midpoint)

    ``"SiO<sub>2</sub>: 49.1%"``
        → ``{'SiO2': 0.491}``

    Anything else (empty, HTML-only, non-numeric, density strings like
    ``"5.46 g/cm3"``) → ``{}``

    Parameters
    ----------
    raw : str
        The raw value string from the ``#CONCENTRATION = …`` header line
        (with the key and ``=`` already stripped).

    Returns
    -------
    dict[str, float]
        Parsed composition entries; empty if the value is not recognised.
    """
    s = raw.strip()
    if not s:
        return {}

    # SiO₂ mineralogy format (volcanic ash, rocks)
    m = _RE_SIO2.search(s)
    if m:
        return {'SiO2': float(m.group(1)) / 100.0}

    # Range wt%  e.g. "55.0 - 55.2 wt%"
    m = _RE_WT_RANGE.search(s)
    if m:
        midpoint = (float(m.group(1)) + float(m.group(2))) / 2.0
        return {'weight_fraction': midpoint / 100.0}

    # Single wt%  e.g. "72 wt%"
    m = _RE_WT_SINGLE.search(s)
    if m:
        return {'weight_fraction': float(m.group(1)) / 100.0}

    # Bare %  e.g. "72%"  — treated as wt% (only occurs in Acids category)
    m = _RE_PCT_BARE.search(s)
    if m:
        return {'weight_fraction': float(m.group(1)) / 100.0}

    return {}


# ---------------------------------------------------------------------------
# ARIA category key derivation
# ---------------------------------------------------------------------------

#: Categories where the depth-2 subcategory is meaningful and should be
#: included in the category key (e.g. ``'Water_and_Ice/Water'``,
#: ``'Acids/Sulphuric'``).
_DEPTH2_CATEGORIES: frozenset[str] = frozenset({'Water_and_Ice', 'Acids'})


_RE_CITATION = re.compile(r'_?\([^)]*\)$')   # strips trailing _(Author_year)
_RE_TEMP_K    = re.compile(r'(\d+(?:\.\d+)?)\s*K', re.I)


def _aria_substance(filepath: Path, aria_root: Path) -> str:
    """
    Return the normalised ARIA substance name (second path component).

    Trailing citation suffixes like ``_(Querry_1987)`` are stripped.
    Result is lowercase with spaces replaced by underscores.

    Returns empty string if the file is not deep enough under ``aria_root``.
    """
    try:
        parts = filepath.relative_to(aria_root).parts
    except ValueError:
        return ""
    if len(parts) < 2:
        return ""
    raw = parts[1]
    cleaned = _RE_CITATION.sub('', raw)
    return cleaned.lower().replace(' ', '_')


def _parse_temperature(raw: str) -> float | None:
    """
    Parse a temperature value in Kelvin from an ARIA ``TEMPERATURE`` header.

    Handles formats like ``'130 K'``, ``'106K'``, ``'298'`` (assumed K).

    Returns *None* if no numeric value is found.
    """
    s = raw.strip()
    if not s:
        return None
    m = _RE_TEMP_K.search(s)
    if m:
        return float(m.group(1))
    # bare number with no unit — assume Kelvin
    try:
        return float(s)
    except ValueError:
        return None


def _aria_category(filepath: Path, aria_root: Path) -> str:
    """
    Derive the normalised category key for a ``.ri`` file.

    For most categories the key is the top-level directory name
    (e.g. ``'volcanic_ash'``).  For categories in :data:`_DEPTH2_CATEGORIES`
    the first two path components are joined with ``'/'``
    (e.g. ``'water_and_ice/water'``).

    Parameters
    ----------
    filepath : Path
        Absolute path to the ``.ri`` file.
    aria_root : Path
        Absolute path to the ARIA database root.

    Returns
    -------
    str
        Normalised key suitable for :func:`get_cmap` / :func:`get_color`.
        Empty string if the file is not under ``aria_root``.
    """
    try:
        parts = filepath.relative_to(aria_root).parts
    except ValueError:
        return ""

    if not parts:
        return ""

    top = parts[0]
    if top in _DEPTH2_CATEGORIES and len(parts) >= 2:
        key = f"{top}/{parts[1]}"
    else:
        key = top

    return key.lower().replace(' ', '_')


class ARIADatabase:
    """
    Interface to a locally stored ARIA refractive index database.

    The database root is a directory containing category sub-directories
    (e.g. ``Minerals``, ``Water_and_Ice``), each of which contains substance
    sub-directories, which in turn contain entry sub-directories holding one
    or more ``.ri`` files.

    Parameters
    ----------
    path : str or Path or None
        Path to the local ARIA database root directory.  If *None* (default),
        the bundled database is used (location resolved by
        :func:`_bundled_aria_path`; can be overridden with the
        ``MIEODG_ARIA_ROOT`` environment variable).

    Examples
    --------
    >>> db = ARIADatabase()                          # bundled ARIA
    >>> db = ARIADatabase('/custom/path/to/ARIA')    # explicit override
    >>> entries = db.list(category='Minerals')
    >>> ri = db.read('Minerals/Calcite_(Querry_1987)/calcite_Querry_1987.ri')
    >>> ri = db.read('Water_and_Ice/Water/298_K_(Hale_and_Querry_1973)/water_Hale_1973.ri',
    ...              wavenumber=np.arange(500, 3000, 10.0))
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            self.root = _bundled_aria_path()
        else:
            self.root = Path(path).resolve()
            if not self.root.is_dir():
                raise FileNotFoundError(f"ARIA database root not found: {self.root}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list(
        self,
        category: str | None = None,
        substance: str | None = None,
        pattern: str | None = None,
    ) -> list[str]:
        """
        List available ``.ri`` files.

        The ARIA database has variable directory depth (2–5 levels), so this
        method traverses the full tree with ``rglob`` and returns the
        *relative path to each .ri file* from the database root.

        Parameters
        ----------
        category : str or None
            Restrict to a specific top-level category directory
            (e.g. ``'Minerals'``, ``'Volcanic_Ash'``).  Case-sensitive.
        substance : str or None
            Restrict to entries whose path contains this substring
            (case-insensitive).  Applied after ``category`` filter.
        pattern : str or None
            Additional substring filter applied to the full relative path
            (case-insensitive).

        Returns
        -------
        list[str]
            Sorted relative paths (``/``-separated) to every matching
            ``.ri`` file.
        """
        search_root = self.root / category if category is not None else self.root
        if category is not None and not search_root.is_dir():
            raise FileNotFoundError(f"ARIA category not found: {search_root}")

        paths = sorted(search_root.rglob('*.ri'))
        entries: list[str] = []
        for p in paths:
            rel = p.relative_to(self.root).as_posix()
            if substance is not None and substance.lower() not in rel.lower():
                continue
            if pattern is not None and pattern.lower() not in rel.lower():
                continue
            entries.append(rel)
        return entries

    def find(
        self,
        pattern: str,
        category: str | None = None,
    ) -> list[str]:
        """
        Search for ``.ri`` files whose path contains ``pattern``.

        Parameters
        ----------
        pattern : str
            Substring to match (case-insensitive).
        category : str or None
            Restrict search to a specific top-level category.

        Returns
        -------
        list[str]
            Matching relative paths to ``.ri`` files.
        """
        return self.list(category=category, pattern=pattern)

    def read(
        self,
        entry: str,
        wavenumber: np.ndarray | None = None,
        extrapolate: bool = False,
        out_of_range: Literal['error', 'warn', 'clip'] = 'error',
    ) -> RefrIndex:
        """
        Read the ``.ri`` file for a given database entry.

        Parameters
        ----------
        entry : str
            Relative path from the database root, e.g.
            ``'Minerals/Calcite/300_K,_e-ray_(Posch_et_al._2007)'``.
            Alternatively, an absolute path to any ``.ri`` file is accepted.
        wavenumber : np.ndarray or None
            Optional interpolation grid [cm⁻¹].  Passed to :func:`readRI`.
        extrapolate : bool
            Passed to :func:`readRI`.
        out_of_range : {'error', 'warn', 'clip'}
            Passed to :func:`readRI`.

        Returns
        -------
        RefrIndex

        Raises
        ------
        FileNotFoundError
            If the entry directory does not exist or contains no ``.ri`` file.
        ValueError
            If the entry directory contains more than one ``.ri`` file and
            the ambiguity cannot be resolved.
        """
        # Absolute path — pass through directly (no category derivation)
        entry_path = Path(entry)
        if entry_path.is_absolute():
            return readRI(entry_path, wavenumber=wavenumber,
                          extrapolate=extrapolate, out_of_range=out_of_range)

        candidate = self.root / entry

        # Relative path pointing directly at a .ri file
        if candidate.suffix == '.ri':
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"ARIA file not found: {candidate}\n"
                    f"Use db.find('{candidate.stem}') to search."
                )
            ri = readRI(candidate, wavenumber=wavenumber,
                        extrapolate=extrapolate, out_of_range=out_of_range)
            self._stamp(ri, candidate)
            return ri

        # Relative path pointing at a directory — find .ri file(s) inside
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"ARIA entry not found: {candidate}\n"
                f"Use db.find('{Path(entry).name}') to search."
            )

        ri_files = sorted(candidate.glob('*.ri'))
        if len(ri_files) == 0:
            raise FileNotFoundError(f"No .ri file in {candidate}")
        if len(ri_files) > 1:
            raise ValueError(
                f"Multiple .ri files in {candidate}:\n"
                + '\n'.join(f'  {f.name}' for f in ri_files)
                + "\nSpecify the full relative path including the filename."
            )

        ri = readRI(ri_files[0], wavenumber=wavenumber,
                    extrapolate=extrapolate, out_of_range=out_of_range)
        self._stamp(ri, ri_files[0])
        return ri

    def _stamp(self, ri: 'RefrIndex', filepath: Path) -> None:
        """Populate ARIA-derived metadata fields on a freshly read RefrIndex."""
        ri.category      = _aria_category(filepath, self.root)
        ri.substance_key = _aria_substance(filepath, self.root)
        ri.composition   = _parse_concentration(ri.header.get('CONCENTRATION', ''))
        ri.T_K           = _parse_temperature(ri.header.get('TEMPERATURE', ''))

    def categories(self) -> list[str]:
        """Return the list of top-level category names."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def substances(self, category: str) -> list[str]:
        """Return substance names within a category."""
        cat_dir = self.root / category
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"Category not found: {category}")
        return sorted(p.name for p in cat_dir.iterdir() if p.is_dir())

    def __repr__(self) -> str:
        n = len(self.list())
        return f"ARIADatabase('{self.root}', {n} files)"
