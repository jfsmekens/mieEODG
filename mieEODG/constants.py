"""
Physical constants and aerosol property tables for mieEODG.

All values are in CGS units (density in g/cm³) unless otherwise noted,
consistent with the cm-based conventions used throughout the package.

Density sources
---------------
H2SO4 solution  : Lund Myhre et al. (2003), Table 1; polynomial fit as a
                  function of weight fraction w [0, 1].  Approximation valid
                  at ~298 K; for other temperatures use a dedicated EOS.
HNO3 solution   : CRC Handbook, PCHIP interpolation over 0–100 wt% at 20 °C.
Water           : IAPWS-IF97 at 298 K.
Ice Ih          : Linear fit to CRC Handbook data, 100–273 K.
                  ρ(T) = 0.9440 − 1.0×10⁻⁴·T  (T in K)
Volcanic ash    : SiO₂-dependent linear model (Gill 2010 / Devine et al. 1984):
                  ρ(SiO₂) = 3.20 − 1.40·f_SiO₂ g/cm³, clipped to [2.30, 2.90].
                  Falls back to 2.60 g/cm³ if SiO₂ fraction is unknown.
Mineral dust    : Same silicate matrix as ash; central value 2.60.
Sea salt        : Dried NaCl crystal density.
Black carbon    : Amorphous soot; literature range 1.7–2.1 g/cm³.
Minerals        : Crystallographic values from RRUFF / Mindat / CRC Handbook.
Salts           : Crystallographic values from CRC Handbook.
Metals          : Pure metal densities from CRC Handbook.
Rocks           : Representative bulk densities; see substance_density table.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Colormap

# ---------------------------------------------------------------------------
# Scalar aerosol densities [g/cm³]
# ---------------------------------------------------------------------------

#: Liquid water at 298 K [g/cm³].
DENSITY_WATER: float = 1.000

#: Ice (Ih) at 273 K [g/cm³].
DENSITY_ICE: float = 0.917

#: Volcanic ash — central value for silicic glass [g/cm³].
DENSITY_ASH: float = 2.60

#: Mineral dust — typical silicate matrix [g/cm³].
DENSITY_DUST: float = 2.60

#: Sea salt (dried NaCl) [g/cm³].
DENSITY_SEA_SALT: float = 2.16

#: Black carbon / soot [g/cm³].
DENSITY_BLACK_CARBON: float = 1.80

#: Calcite (CaCO₃) [g/cm³].
DENSITY_CALCITE: float = 2.71

#: Quartz (SiO₂) [g/cm³].
DENSITY_QUARTZ: float = 2.65


# ---------------------------------------------------------------------------
# H₂SO₄ solution density as a function of weight fraction
# ---------------------------------------------------------------------------

def ice_density(T: float) -> float:
    """
    Density of ice Ih as a function of temperature.

    Parameters
    ----------
    T : float
        Temperature [K], valid range approximately 100–273 K.

    Returns
    -------
    float
        Density [g/cm³], clipped to the physically plausible range
        [0.910, 0.940].

    Notes
    -----
    Linear fit to CRC Handbook of Chemistry and Physics data for ice Ih:

        ρ(T) = 0.9440 − 1.0×10⁻⁴ · T

    Representative values:
      T = 273 K → 0.917 g/cm³
      T = 200 K → 0.924 g/cm³
      T = 100 K → 0.934 g/cm³
    """
    rho = 0.9440 - 1.0e-4 * float(T)
    return float(np.clip(rho, 0.910, 0.940))


def sio2_density(f_sio2: float) -> float:
    """
    Bulk density of volcanic ash / rock as a function of SiO₂ weight fraction.

    Parameters
    ----------
    f_sio2 : float
        SiO₂ weight fraction in [0, 1].

    Returns
    -------
    float
        Estimated bulk density [g/cm³], clipped to [2.30, 2.90].

    Notes
    -----
    Linear model reflecting the inverse relationship between silica content
    and density in volcanic materials:

        ρ = 3.20 − 1.40 · f_SiO₂

    Characteristic points:
      f_SiO₂ = 0.45 (mafic/basaltic ash)        → 2.83 g/cm³
      f_SiO₂ = 0.60 (intermediate/andesitic ash) → 2.36 → clipped 2.36...

    Bracket reference (Devine et al. 1984; Gill 2010):
      Mafic / basaltic   (45–55% SiO₂): 2.7–2.9 g/cm³
      Intermediate       (55–65% SiO₂): 2.5–2.7 g/cm³
      Silicic / rhyolitic (>65% SiO₂): 2.3–2.5 g/cm³
    """
    rho = 3.20 - 1.40 * float(f_sio2)
    return float(np.clip(rho, 2.30, 2.90))


def h2so4_density(w: float | np.ndarray) -> float | np.ndarray:
    """
    Density of aqueous H₂SO₄ solution at ~298 K.

    Parameters
    ----------
    w : float or np.ndarray
        Sulfuric acid weight fraction in [0, 1].  For example, 72 wt% → w=0.72.

    Returns
    -------
    float or np.ndarray
        Density [g/cm³].

    Notes
    -----
    Polynomial fit to data from Lund Myhre et al. (2003), Table 1,
    valid for w ∈ [0, 1] at approximately 298 K.

    At the endpoints the fit gives:
      w = 0   → ~1.000 g/cm³  (pure water)
      w = 1   → ~1.838 g/cm³  (100% H₂SO₄)

    Reference
    ---------
    C.E. Lund Myhre, D.H. Christensen, F.M. Nicolaisen, C.J. Nielsen,
    "Spectroscopic study of aqueous H₂SO₄ at different temperatures and
    compositions: variation in dissociation and optical properties",
    J. Phys. Chem. A 107, 1979–1991 (2003).
    """
    w = np.asarray(w, dtype=np.float64)
    # 4th-degree polynomial: rho = sum c_i * w^i
    # Coefficients fit to Lund Myhre Table 1 (concentrations 0–96 wt%)
    c = np.array([1.0000, 0.6986, 0.3075, -0.4263, 0.2458])
    return float(np.polyval(c[::-1], w)) if w.ndim == 0 else np.polyval(c[::-1], w)


def hno3_density(w: float | np.ndarray) -> float | np.ndarray:
    """
    Density of aqueous HNO₃ solution at ~20 °C (293 K).

    Parameters
    ----------
    w : float or np.ndarray
        Nitric acid weight fraction in [0, 1].  For example, 50 wt% → w=0.50.

    Returns
    -------
    float or np.ndarray
        Density [g/cm³].

    Notes
    -----
    PCHIP interpolation of tabulated data from the CRC Handbook of
    Chemistry and Physics (concentrations 0–100 wt% at 20 °C):

      0 %  → 0.9982 g/cm³  (pure water)
     10 %  → 1.0543
     20 %  → 1.1150
     30 %  → 1.1800
     40 %  → 1.2463
     50 %  → 1.3100
     60 %  → 1.3667
     70 %  → 1.4134
     80 %  → 1.4521
     90 %  → 1.4826
    100 %  → 1.5129

    Reference
    ---------
    CRC Handbook of Chemistry and Physics, 84th ed., Table of
    "Concentrative Properties of Aqueous Solutions: Density, Refractive
    Index, Freezing Point Depression, and Viscosity", Nitric Acid entry.
    """
    from scipy.interpolate import PchipInterpolator as _Pchip

    _w_pts   = np.array([0.00, 0.10, 0.20, 0.30, 0.40,
                          0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    _rho_pts = np.array([0.9982, 1.0543, 1.1150, 1.1800, 1.2463,
                          1.3100, 1.3667, 1.4134, 1.4521, 1.4826, 1.5129])
    _interp = _Pchip(_w_pts, _rho_pts, extrapolate=True)

    w = np.asarray(w, dtype=np.float64)
    result = _interp(np.clip(w, 0.0, 1.0))
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Convenience look-up table: canonical short name → density [g/cm³]
# ---------------------------------------------------------------------------

#: Map from short aerosol label (or normalised ARIA category key) to bulk
#: density [g/cm³].  For H₂SO₄ solutions use :func:`h2so4_density`; for
#: HNO₃ solutions use :func:`hno3_density`; for ice use :func:`ice_density`;
#: for SiO₂-bearing ash use :func:`sio2_density`.
DENSITY_TABLE: dict[str, float] = {
    # Legacy / short-name keys
    'water':               DENSITY_WATER,
    'ice':                 DENSITY_ICE,
    'ash':                 DENSITY_ASH,
    'volcanic_ash':        DENSITY_ASH,
    'dust':                DENSITY_DUST,
    'mineral_dust':        DENSITY_DUST,
    'sea_salt':            DENSITY_SEA_SALT,
    'black_carbon':        DENSITY_BLACK_CARBON,
    'soot':                DENSITY_BLACK_CARBON,
    'calcite':             DENSITY_CALCITE,
    'quartz':              DENSITY_QUARTZ,
    # ARIA depth-2 category keys
    'water_and_ice/water': DENSITY_WATER,
    'water_and_ice/ice':   DENSITY_ICE,    # reference; prefer ice_density(T)
    'sand_and_dust':       DENSITY_DUST,
}


#: Per-substance density table keyed by normalised ARIA substance name
#: (``'<category>/<substance>'``, lowercase, underscores, citation stripped).
#: Values are crystallographic or representative bulk densities [g/cm³].
#:
#: Sources
#: -------
#: Minerals   : RRUFF Project / Mindat.org / CRC Handbook 97th ed.
#: Salts      : CRC Handbook 97th ed.
#: Metals     : CRC Handbook 97th ed.
#: Rocks      : Representative bulk densities (Telford et al. 1990;
#:              Carmichael 1989).  Pumice is highly variable; value given
#:              is for dense glass fraction (vesicle-free).
SUBSTANCE_DENSITY: dict[str, float] = {
    # --- Minerals ---
    'minerals/anatase':          3.90,   # TiO₂ polymorph
    'minerals/anhydrite':        2.96,   # CaSO₄
    'minerals/brookite':         4.13,   # TiO₂ polymorph
    'minerals/calcite':          2.71,   # CaCO₃
    'minerals/dolomite':         2.85,   # CaMg(CO₃)₂
    'minerals/enstatite':        3.20,   # MgSiO₃
    'minerals/hematite':         5.26,   # Fe₂O₃
    'minerals/illite':           2.75,   # clay mineral
    'minerals/kaolinite':        2.63,   # Al₂Si₂O₅(OH)₄
    'minerals/meteoric_diamond': 3.51,   # cubic carbon
    'minerals/montmorillonite':  2.35,   # smectite clay
    'minerals/olivine':          3.32,   # (Mg,Fe)₂SiO₄  forsteritic end
    'minerals/olivine_fayalite': 4.39,   # Fe₂SiO₄  fayalite end
    'minerals/perovskite':       4.10,   # CaTiO₃
    'minerals/quartz':           2.65,   # SiO₂
    'minerals/rutile':           4.23,   # TiO₂ polymorph
    'minerals/wustite':          5.74,   # FeO
    'minerals/zinc_selenide':    5.27,   # ZnSe
    'minerals/zinc_sulphide':    4.09,   # ZnS (sphalerite)
    # --- Salts ---
    'salts/ammonium_bisulfate':  1.78,   # NH₄HSO₄
    'salts/ammonium_sulphate':   1.77,   # (NH₄)₂SO₄
    'salts/barium_fluoride':     4.89,   # BaF₂
    'salts/caesium_bromide':     4.44,   # CsBr
    'salts/caesium_iodide':      4.51,   # CsI
    'salts/potassium_chloride':  1.98,   # KCl
    'salts/sodium_chloride':     2.16,   # NaCl
    'salts/sodium_nitrate':      2.26,   # NaNO₃
    'salts/sodium_sulphate':     2.66,   # Na₂SO₄ (anhydrous)
    # --- Metals ---
    'metals/manganese':         7.21,    # Mn (α-phase)
    'metals/molybdenum':        10.22,   # Mo
    'metals/zirconium':          6.52,   # Zr (α-phase)
    # --- Rocks and conglomerates ---
    'rocks_and_conglomerates/andesite': 2.65,   # intermediate volcanic rock
    'rocks_and_conglomerates/basalt':   2.85,   # mafic volcanic rock
    'rocks_and_conglomerates/granite':  2.67,   # felsic intrusive rock
    'rocks_and_conglomerates/limonite': 3.60,   # hydrated iron oxide aggregate
    'rocks_and_conglomerates/obsidian': 2.40,   # rhyolitic volcanic glass
    'rocks_and_conglomerates/pumice':   2.35,   # vesicle-free glass fraction
}


# ---------------------------------------------------------------------------
# Spectral range presets
# ---------------------------------------------------------------------------

#: Named spectral-grid presets (wavenumber bounds + default step, cm⁻¹).
#: Single source of truth — the demo config and the GUI both build from this.
SPECTRAL_RANGE_PRESETS: dict[str, dict] = {
    'thermal_ir': {
        'wn_min': 650.0, 'wn_max': 1350.0, 'wn_step': 1.0,
        'description': 'Thermal IR (8–15 µm) — common for aerosol retrievals',
    },
    'mid_ir': {
        'wn_min': 500.0, 'wn_max': 5000.0, 'wn_step': 5.0,
        'description': 'Mid-IR (2–20 µm) — complete aerosol spectrum',
    },
    'solar': {
        'wn_min': 4000.0, 'wn_max': 50000.0, 'wn_step': 10.0,
        'description': 'Solar (0.2–2.5 µm) — visible to near-IR',
    },
    'visible': {
        'wn_min': 10000.0, 'wn_max': 25000.0, 'wn_step': 50.0,
        'description': 'Visible (0.4–1.0 µm) — VIS instruments',
    },
}


# ---------------------------------------------------------------------------
# Canonical ARIA default entries
# ---------------------------------------------------------------------------

#: Default ``.ri`` entry (relative ARIA path) for each category or common
#: species label.  Preference is given to room-temperature (~298 K) data
#: from a well-established reference.  These paths are suitable for passing
#: directly to :meth:`~mieEODG.refindex.ARIADatabase.read`.
ARIA_DEFAULTS: dict[str, str] = {
    # Acids
    'acids':               (
        'Acids/Sulphuric/70%_to_79%/'
        '72%,_298_K_(Lund_Myhre_et_al._2003)/'
        'H2SO4_72_298K_Myhre_2003.ri'
    ),
    'acids/sulphuric':     (
        'Acids/Sulphuric/70%_to_79%/'
        '72%,_298_K_(Lund_Myhre_et_al._2003)/'
        'H2SO4_72_298K_Myhre_2003.ri'
    ),
    'acids/nitric':        (
        'Acids/Nitric/40%_to_59%/'
        '50%,_293_K_(Biermann_et_al._2000)/'
        'interpolated/'
        'HNO350_293_Biermann_2000.ri'
    ),
    # Water and Ice
    'water_and_ice/water': (
        'Water_and_Ice/Water/'
        '298_K_(Hale_and_Querry_1973)/'
        'H2O_Hale_1973.ri'
    ),
    'water_and_ice/ice':   (
        'Water_and_Ice/Ice/'
        '200_K_to_300_K/'
        '266_K_(Warren_and_Brandt_2008)/'
        'ICE_Warren_2008.ri'
    ),
    # Sand and Dust
    'sand_and_dust':       (
        'Sand_and_Dust/Dust/'
        'Algeria_-_67%_clays,_17%_quartz_(Di_Biagio_et_al._2014)/'
        'N93_DiBiagio_2014.ri'
    ),
    # Volcanic Ash
    'volcanic_ash':        (
        'Volcanic_Ash/Etna/'
        'Dequine_2020/'
        'etna-ash_Deguine_2020.ri'
    ),
    # Minerals (most commonly needed)
    'minerals/calcite':    'Minerals/Calcite',
    'minerals/quartz':     'Minerals/Quartz',
    'minerals/hematite':   'Minerals/Hematite',
    'minerals/kaolinite':  'Minerals/Kaolinite_(Querry_1987)',
    # Salts
    'salts/nacl':          'Salts/Sodium_Chloride',
    'salts/nh4hso4':       'Salts/Ammonium_Bisulfate',
    'salts/nh4so4':        'Salts/Ammonium_Sulphate',
}


# ---------------------------------------------------------------------------
# Display labels (for plot axes / legends)
# ---------------------------------------------------------------------------

#: Human-readable labels for standard aerosol types.
DISPLAY_NAMES: dict[str, str] = {
    'water':         'Water',
    'ice':           'Ice',
    'ash':           'Volcanic ash',
    'volcanic_ash':  'Volcanic ash',
    'dust':          'Mineral dust',
    'mineral_dust':  'Mineral dust',
    'sea_salt':      'Sea salt',
    'black_carbon':  'Black carbon',
    'soot':          'Black carbon',
    'calcite':       'Calcite',
    'quartz':        'Quartz',
}


# ---------------------------------------------------------------------------
# Plot colours and colormaps
#
# Keys are canonical short names (case-insensitive via get_cmap / get_color).
# plot_colors  : single representative colour per species (for lines, markers)
# plot_cmaps   : sequential matplotlib colormap per species (for size sweeps,
#                heatmaps, colorbars)
#
# Colormap rationale
# ------------------
# H2SO4        Greens   — sulphate aerosol conventionally shown in green
# ash          Greys    — ash / silicate particles are visually grey
# water        Blues    — liquid water droplets
# ice          GnBu     — icy blue-green (distinguishable from liquid water)
# dust         YlOrBr   — desert dust warm orange-brown tones
# sea_salt     PuBu     — oceanic origin, blue-purple
# black_carbon Greys    — soot; same family as ash but distinct single colour
# calcite      BuPu     — carbonate mineral
# quartz       OrRd     — silica mineral
# <default>    plasma   — neutral high-contrast fallback for unknown species
# ---------------------------------------------------------------------------

#: Default colormap for species not listed in :data:`plot_cmaps`.
_DEFAULT_CMAP: str = 'cmc.imola_r'

#: Default single colour for species not listed in :data:`plot_colors`.
_DEFAULT_COLOR: str = 'tab:grey'

#: Sequential colormap name for each ARIA category (and depth-2 subcategory
#: where relevant).  Keys are lowercase, spaces replaced by underscores.
#: cmcrameri maps are prefixed ``'cmc.'``; plain names are matplotlib built-ins.
#: Use :func:`get_cmap` for a safe, case-insensitive lookup with fallback.
plot_cmaps: dict[str, str] = {
    # --- ARIA top-level categories ---
    # Convention: low norm value (small size / dilute) → light colour,
    #             high norm value (large size / concentrated) → dark colour.
    # All cmcrameri maps default to dark→light, so _r variants are used.
    'acids':                    'cmc.bamako_r',   # light yellow→dark blue-green (parent fallback)
    'acids/sulphuric':          'cmc.bamako_r',   # H₂SO₄ — light yellow→dark blue-green
    'acids/nitric':             'cmc.lajolla_r',    # HNO₃  — light cream→dark magenta-red
    'combustion_and_organics':  'cmc.tokyo_r',  # light cream→dark brown-black
    'metals':                   'cmc.davos_r',    # light→dark steel blue-grey
    'minerals':                 'cmc.acton_r',    # light→dark purple-grey
    'organic_acids':            'cmc.buda_r',     # light→dark warm orange
    'rocks_and_conglomerates':  'cmc.turku_r',    # light→dark earthy brown
    'salts':                    'cmc.lapaz_r',    # light cream→dark blue-purple
    'sand_and_dust':            'cmc.bilbao_r',     # light→dark neutral
    'volcanic_ash':             'cmc.grayC_r',    # light→dark perceptual grey
    # --- Water_and_Ice depth-2 subcategories ---
    'water_and_ice/water':      'cmc.devon_r',     # light→dark green-blue
    'water_and_ice/ice':        'cool',           # cyan→magenta (already light→dark)
    # fallback for the parent category (if subcategory is not resolved)
    'water_and_ice':            'cmc.devon_r',
}

#: Representative single colour per aerosol species (lines, markers, patches).
plot_colors: dict[str, str] = {
    'h2so4':         'tab:green',
    'hno3':          'tab:red',
    'ash':           'tab:grey',
    'volcanic_ash':  'tab:grey',
    'water':         'deepskyblue',
    'ice':           'tab:cyan',
    'dust':          'tab:orange',
    'mineral_dust':  'tab:orange',
    'sea_salt':      'steelblue',
    'black_carbon':  'k',
    'soot':          'k',
    'calcite':       'mediumpurple',
    'quartz':        'tomato',
}


# ---------------------------------------------------------------------------
# Composition-aware density resolver
# ---------------------------------------------------------------------------

#: ARIA categories (normalised) for which density depends on weight fraction
#: and should be resolved via a dedicated function rather than DENSITY_TABLE.
_SOLUTION_CATEGORIES: frozenset[str] = frozenset({
    'acids',
    'acids/sulphuric',
    'acids/nitric',
    'organic_acids',
})


def aria_density(
    category:    str,
    composition: dict[str, float],
    T_K:           float | None = None,
    substance_key: str | None   = None,
) -> float | None:
    """
    Derive bulk particle density [g/cm³] from ARIA metadata.

    Lookup hierarchy
    ----------------
    1. **Substance-level** — :data:`SUBSTANCE_DENSITY` keyed by
       ``'<category>/<substance>'`` (e.g. ``'minerals/hematite'``).
    2. **Composition-dependent** functions:

       - H₂SO₄ solutions → :func:`h2so4_density` (weight fraction required).
       - Ice Ih → :func:`ice_density` (uses ``temperature`` if supplied,
         otherwise falls back to 273 K reference value).
       - Volcanic ash / rocks with known SiO₂ → :func:`sio2_density`.

    3. **Category-level** — :data:`DENSITY_TABLE` keyed by normalised
       ARIA category.
    4. Returns *None* if no value can be determined (caller must supply
       ``density`` explicitly to :class:`~mieEODG.aerosol.AerosolXSC`).

    Parameters
    ----------
    category : str
        Normalised ARIA category key (``RefrIndex.category``).
    composition : dict[str, float]
        ``RefrIndex.composition`` — may contain ``'weight_fraction'``
        or ``'SiO2'``.
    T_K : float or None
        Temperature [K] from the ``TEMPERATURE`` header (``RefrIndex.T_K``),
        used for ice density.  If *None*, the 273 K reference is used.
    substance_key : str or None
        Normalised ARIA substance name (second path component after
        stripping citation suffixes), e.g. ``'hematite'``.  If supplied,
        a ``'<category>/<substance_key>'`` entry in :data:`SUBSTANCE_DENSITY`
        is tried first.
    """
    key = _normalise_key(category)

    # 1. Substance-level lookup
    if substance_key is not None:
        sub_key = f"{key}/{_normalise_key(substance_key)}"
        if sub_key in SUBSTANCE_DENSITY:
            return SUBSTANCE_DENSITY[sub_key]

    # 2a. Acid solutions (weight-fraction dependent)
    if key in _SOLUTION_CATEGORIES:
        w = composition.get('weight_fraction')
        if w is None:
            return None   # wt% unknown — caller must supply density
        if 'sulphuric' in key:
            return h2so4_density(w)
        if 'nitric' in key:
            return hno3_density(w)
        # Other solutions (e.g. organic_acids): fall through

    # 2b. Ice Ih — temperature-dependent
    if key == 'water_and_ice/ice':
        T = T_K if T_K is not None else 273.0
        return ice_density(T)

    # 2c. Volcanic ash / rocks — SiO₂-dependent
    if key in ('volcanic_ash', 'rocks_and_conglomerates'):
        f_sio2 = composition.get('SiO2')
        if f_sio2 is not None:
            return sio2_density(f_sio2)

    # 3. Category-level fallback
    return DENSITY_TABLE.get(key)


def _normalise_key(key: str) -> str:
    """Lowercase and replace spaces with underscores for dict lookup."""
    return key.strip().lower().replace(' ', '_')


def _resolve_cmap(name: str) -> Colormap:
    """
    Resolve a colormap name string to a :class:`matplotlib.colors.Colormap`.

    Names prefixed with ``'cmc.'`` are looked up in :mod:`cmcrameri.cm`
    (e.g. ``'cmc.bamako'`` → ``cmcrameri.cm.bamako``).  All other names are
    resolved via :mod:`matplotlib.pyplot`.

    Parameters
    ----------
    name : str
        Colormap identifier, e.g. ``'cmc.bamako'``, ``'plasma'``, ``'cool'``.

    Returns
    -------
    Colormap
        The resolved colormap object.

    Raises
    ------
    ImportError
        If a ``'cmc.*'`` name is requested but :mod:`cmcrameri` is not installed.
    KeyError
        If the name is not found in the expected source.
    """
    if name.startswith('cmc.'):
        try:
            import cmcrameri.cm as _cmc
        except ImportError as exc:
            raise ImportError(
                f"Colormap '{name}' requires the cmcrameri package: "
                "pip install cmcrameri"
            ) from exc
        attr = name[4:]   # strip 'cmc.'
        if not hasattr(_cmc, attr):
            raise KeyError(f"cmcrameri has no colormap '{attr}'")
        return getattr(_cmc, attr)  # includes _r variants natively

    import matplotlib.pyplot as _plt
    return _plt.colormaps[name]


def get_cmap(category: str) -> Colormap:
    """
    Return the colormap for a given ARIA category or species label.

    Lookup is case-insensitive; spaces and underscores are interchangeable.
    Categories not in :data:`plot_cmaps` return the default colormap
    (``'plasma'``).

    Parameters
    ----------
    category : str
        ARIA category key, e.g. ``'Volcanic_Ash'``, ``'Water_and_Ice/Water'``,
        or a legacy species label like ``'h2so4'``.

    Returns
    -------
    matplotlib.colors.Colormap
        The resolved colormap object, ready for direct use in plots.
    """
    name = plot_cmaps.get(_normalise_key(category), _DEFAULT_CMAP)
    return _resolve_cmap(name)


def get_color(category: str) -> str:
    """
    Return the representative plot colour for a given ARIA category or species.

    Lookup is case-insensitive; spaces and underscores are interchangeable.
    Categories not in :data:`plot_colors` return the default colour
    (``'tab:grey'``).

    Parameters
    ----------
    category : str
        ARIA category key or legacy species label, e.g. ``'volcanic_ash'``,
        ``'h2so4'``, ``'water'``.

    Returns
    -------
    str
        Matplotlib colour string.
    """
    return plot_colors.get(_normalise_key(category), _DEFAULT_COLOR)
