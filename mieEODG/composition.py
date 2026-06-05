"""
Volcanic ash composition utilities for Prata et al. (2019) RI parameterization.

Provides composition dataclass and NBO/T calculator for bulk silicate rock
oxide analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Middlemost (1989) Fe²⁺/Fe³⁺ f ratios by rock type
# ---------------------------------------------------------------------------
# f = Fe²⁺ / (Fe²⁺ + Fe³⁺), with f defined at constant total iron
MIDDLEMOST_F_RATIOS: dict[str, float] = {
    'basalt':   0.30,
    'andesite': 0.35,
    'dacite':   0.40,
    'rhyolite': 0.50,
}


# ---------------------------------------------------------------------------
# BulkComposition dataclass
# ---------------------------------------------------------------------------

@dataclass
class BulkComposition:
    """
    Bulk oxide composition of volcanic ash by weight percentage.

    Parameters
    ----------
    sio2 : float
        SiO₂ wt%, 0–100. Required.
    tio2 : float
        TiO₂ wt%, default 0.
    al2o3 : float
        Al₂O₃ wt%, default 0.
    fe2o3 : float
        Total iron as Fe₂O₃ wt% (always available from XRF), default 0.
    feo : float or None
        FeO wt% (if known from wet chemistry or Mössbauer), default None.
        If provided, takes precedence over Middlemost estimation.
    mno : float
        MnO wt%, default 0.
    mgo : float
        MgO wt%, default 0.
    cao : float
        CaO wt%, default 0.
    na2o : float
        Na₂O wt%, default 0.
    k2o : float
        K₂O wt%, default 0.
    p2o5 : float
        P₂O₅ wt%, default 0.

    Notes
    -----
    Fe₂O₃ represents total iron. If FeO (Fe²⁺) is not directly measured,
    it is estimated via Fe²⁺/Fe³⁺ split using Middlemost (1989) f ratios
    based on rock type (see `nbot_from_composition`).
    """

    sio2: float
    tio2: float = 0.0
    al2o3: float = 0.0
    fe2o3: float = 0.0
    feo: float | None = None
    mno: float = 0.0
    mgo: float = 0.0
    cao: float = 0.0
    na2o: float = 0.0
    k2o: float = 0.0
    p2o5: float = 0.0

    def __post_init__(self) -> None:
        """Validate oxide range."""
        if not 0 <= self.sio2 <= 100:
            raise ValueError(f"sio2 must be in [0, 100]; got {self.sio2}")

    def __repr__(self) -> str:
        """Return compact representation."""
        items = [f"SiO₂={self.sio2:.1f}"]
        if self.al2o3 > 0:
            items.append(f"Al₂O₃={self.al2o3:.1f}")
        if self.fe2o3 > 0:
            items.append(f"Fe_tot={self.fe2o3:.1f}")
        if self.mgo > 0:
            items.append(f"MgO={self.mgo:.1f}")
        if self.cao > 0:
            items.append(f"CaO={self.cao:.1f}")
        if self.na2o > 0 or self.k2o > 0:
            items.append(f"Na₂O+K₂O={self.na2o + self.k2o:.1f}")
        return f"BulkComposition({', '.join(items)})"


# ---------------------------------------------------------------------------
# NBO/T Calculator
# ---------------------------------------------------------------------------

def nbot_from_composition(
    comp: BulkComposition,
    *,
    f: float | None = None,
    rock_type: str | None = None,
) -> float:
    r"""
    Compute NBO/T (Non-Bridging Oxygen / Tetrahedra) from bulk composition.

    NBO/T quantifies the degree of depolymerization in silicate melts, ranging
    from 0 (fully polymerized silica) to ~0.8 (highly depolymerized basalt).

    Parameters
    ----------
    comp : BulkComposition
        Bulk oxide composition.
    f : float or None
        Explicit Fe²⁺/Fe³⁺ ratio (Fe²⁺ / (Fe²⁺ + Fe³⁺)).
        If provided, overrides Middlemost estimation.
    rock_type : str or None
        Rock type for Middlemost (1989) f lookup: 'basalt', 'andesite',
        'dacite', 'rhyolite'. Ignored if comp.feo is set or f is supplied.

    Returns
    -------
    float
        NBO/T value, typically in [0, 0.8] for geological samples.

    Raises
    ------
    ValueError
        If Fe²⁺/Fe³⁺ split cannot be resolved (comp.feo is None,
        f is None, and rock_type is not recognized or None).

    Notes
    -----
    NBO/T definition (Mysen & Richet 2005):

    .. math::

        \text{NBO/T} = \frac{2O - 4T}{T}

    where:
    - O = total oxygen in the melt (from oxide wt%)
    - T = tetrahedral cations = Si⁴⁺ + Ti⁴⁺ + P⁵⁺ + Al³⁺ (excess) + Fe³⁺ (excess)

    Al and Fe are partitioned into tetrahedral vs. octahedral coordination
    based on alkali availability. Excess Al beyond the Al needed to
    charge-balance alkalis is assumed octahedral.
    """
    # --- Resolve Fe²⁺/Fe³⁺ split ---
    if comp.feo is not None:
        # Direct measurement takes precedence
        feo = comp.feo
        logger.debug(f"Using measured FeO = {feo:.2f} wt%")
    elif f is not None:
        # Explicit f supplied
        feo = _split_iron(comp.fe2o3, f)
        logger.debug(f"Using explicit f = {f:.2f} → FeO = {feo:.2f} wt%")
    elif rock_type is not None:
        # Middlemost lookup
        if rock_type not in MIDDLEMOST_F_RATIOS:
            raise ValueError(
                f"rock_type {rock_type!r} not in {sorted(MIDDLEMOST_F_RATIOS.keys())}"
            )
        f = MIDDLEMOST_F_RATIOS[rock_type]
        feo = _split_iron(comp.fe2o3, f)
        logger.debug(
            f"Using Middlemost ({rock_type}) f = {f:.2f} → FeO = {feo:.2f} wt%"
        )
    else:
        raise ValueError(
            "Fe²⁺/Fe³⁺ split cannot be resolved: provide comp.feo, f parameter, "
            "or rock_type (one of 'basalt', 'andesite', 'dacite', 'rhyolite')"
        )

    # --- Convert wt% to molar amounts ---
    # Molecular weights (g/mol)
    mw = {
        'SiO2': 60.09,
        'TiO2': 79.87,
        'Al2O3': 101.96,
        'Fe2O3': 159.69,
        'FeO': 71.85,
        'MnO': 70.94,
        'MgO': 40.31,
        'CaO': 56.08,
        'Na2O': 61.98,
        'K2O': 94.20,
        'P2O5': 141.94,
    }

    # Moles of each oxide (normalized to 100g sample)
    moles = {
        'Si': comp.sio2 / mw['SiO2'],
        'Ti': comp.tio2 / mw['TiO2'],
        'Al': comp.al2o3 / (mw['Al2O3'] / 2),  # 2 Al per Al₂O₃
        'Fe3': (comp.fe2o3 - feo * 71.85 / 159.69) / (mw['Fe2O3'] / 2),  # Fe³⁺
        'Fe2': feo / mw['FeO'],  # Fe²⁺
        'Mn': comp.mno / mw['MnO'],
        'Mg': comp.mgo / mw['MgO'],
        'Ca': comp.cao / mw['CaO'],
        'Na': comp.na2o / (mw['Na2O'] / 2),  # 2 Na per Na₂O
        'K': comp.k2o / (mw['K2O'] / 2),  # 2 K per K₂O
        'P': comp.p2o5 / (mw['P2O5'] / 2),  # 2 P per P₂O₅
    }

    # --- Charge balance and cation partitioning ---
    # Cation charges
    alkali = moles['Na'] + moles['K']  # monovalent
    alk_earth = moles['Ca'] + moles['Mn'] + moles['Mg']  # divalent
    total_divalent = 2 * alk_earth + moles['Fe2']

    # Tetrahedral cations: Si + Ti + P (always tet), then Al and Fe³ if needed
    T_base = moles['Si'] + moles['Ti'] + moles['P']
    al_tet = min(moles['Al'], alkali)  # Al⁴⁺ to charge-balance monovalent alkalis
    fe3_tet = min(moles['Fe3'], max(0, total_divalent - alkali))  # excess divalent

    T = T_base + al_tet + fe3_tet

    # --- Compute oxygen balance ---
    # Oxygen from each oxide
    o_moles = (
        2 * moles['Si']  # SiO₂
        + 2 * moles['Ti']  # TiO₂
        + 3 * moles['Al']  # Al₂O₃
        + 3 * moles['Fe3']  # Fe₂O₃
        + moles['Fe2']  # FeO
        + moles['Mn']  # MnO
        + moles['Mg']  # MgO
        + moles['Ca']  # CaO
        + moles['Na']  # Na₂O
        + moles['K']  # K₂O
        + 5 * moles['P']  # P₂O₅
    )

    # --- NBO/T ---
    if T <= 0:
        raise ValueError("No tetrahedral cations (Si, Ti, P); cannot compute NBO/T")

    nbot = (2 * o_moles - 4 * T) / T
    logger.debug(f"O moles = {o_moles:.4f}, T moles = {T:.4f}, NBO/T = {nbot:.4f}")

    return float(nbot)


# ---------------------------------------------------------------------------
# Helper: Fe²⁺/Fe³⁺ Split
# ---------------------------------------------------------------------------

def _split_iron(fe2o3_wt: float, f: float) -> float:
    """
    Convert total iron (Fe₂O₃ wt%) to FeO using Fe²⁺/Fe³⁺ ratio f.

    Parameters
    ----------
    fe2o3_wt : float
        Total iron expressed as Fe₂O₃ wt%.
    f : float
        Fe²⁺ / (Fe²⁺ + Fe³⁺) ratio, typically in [0, 1].

    Returns
    -------
    float
        FeO wt%.

    Notes
    -----
    Given total iron as Fe₂O₃ and a split ratio f, solve:
    - Fe²⁺ = f · Fe_tot
    - Fe³⁺ = (1 - f) · Fe_tot

    where Fe_tot is moles of total iron. Convert back to wt% FeO.
    """
    mw_fe2o3 = 159.69
    mw_feo = 71.85

    # Total moles of Fe from Fe₂O₃
    fe_moles = (fe2o3_wt / mw_fe2o3) * 2

    # Split
    fe2_moles = f * fe_moles

    # Convert to FeO wt%
    feo_wt = fe2_moles * mw_feo

    return float(feo_wt)
