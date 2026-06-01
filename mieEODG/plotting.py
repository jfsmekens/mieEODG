"""
Plotting utilities for mieEODG.

Provides publication-ready figures for Mie scattering results with a
consistent style: thin grid lines, a secondary top axis showing wavelength,
and cmcrameri colormaps keyed to ARIA species categories.

Public functions
----------------
plot_spectrum      Plot one or more spectra (MEC / MSC / MAC vs wavenumber).
plot_lut_sweep     Plot a dense r0 sweep from a single AerosolLUT.
plot_lut           Multi-panel LUT survey from a dict or list of AerosolLUT objects.

Secondary-axis helpers (called internally, exposed for external use)
---------------------------------------------------------------------
_add_top_axis      Attach a wavelength axis on top of a wavenumber axes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global rcParams — match the demo script figure style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':                   10,
    'axes.labelsize':              9,
    'axes.titlesize':              10,
    'xtick.labelsize':             9,
    'ytick.labelsize':             9,
    'lines.linewidth':             0.9,
    'axes.grid':                   True,
    'grid.linewidth':              0.3,
    'grid.alpha':                  0.4,
    'grid.linestyle':              '-',
    'axes.axisbelow':              True,
    'figure.constrained_layout.use': True,
    'axes.formatter.limits':       (-4, 4),
})

# ---------------------------------------------------------------------------
# Wavelength tick candidates
# ---------------------------------------------------------------------------
# Primary axis: cm⁻¹, secondary axis: µm
_WL_TICKS_UM: list[float] = [
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,   # VNIR
    1.25, 1.5, 2.0, 2.5,                  # SWIR
    3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, # MWIR / TIR
    25, 30, 50, 75, 100, 150, 200,        # FIR
]
# Primary axis: cm⁻¹, secondary axis: nm
_WL_TICKS_NM: list[float] = [
    400, 450, 500, 550, 600, 650, 700,    # VNIR
    750, 800, 900, 1000, 1200,
    1500, 2000, 2500, 3000, 5000,
]
_WL_TICKS_MAX: int = 12


# ---------------------------------------------------------------------------
# Conversion helpers (self-inverse for secondary_xaxis)
# ---------------------------------------------------------------------------

def _fw(x: float) -> float:
    """Wavenumber (cm⁻¹) ↔ wavelength (µm).  Self-inverse."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1.0e4 / x, np.nan)


def _fn(x: float) -> float:
    """Wavenumber (cm⁻¹) ↔ wavelength (nm).  Self-inverse."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1.0e7 / x, np.nan)


# ---------------------------------------------------------------------------
# Secondary top-axis
# ---------------------------------------------------------------------------

def _add_top_axis(
    ax: plt.Axes,
    units: str = 'um',
) -> plt.Axes | None:
    """
    Attach a secondary x-axis on top showing wavelength.

    The primary axis is assumed to show wavenumber in cm⁻¹.  Tick positions
    are chosen from a pre-defined candidate list and thinned if too many would
    be shown.

    Parameters
    ----------
    ax : plt.Axes
        Primary (wavenumber) axes.
    units : {'um', 'nm'}
        Units for the secondary axis.  Default ``'um'`` (µm).

    Returns
    -------
    plt.Axes or None
        The secondary axes object, or ``None`` if the primary limits are not
        strictly positive (which would make 1/x undefined).  The caller should
        remove this object (``secax.remove()``) before redrawing in an
        interactive context.
    """
    lo, hi = ax.get_xlim()
    # Works for both ascending and descending wavenumber axes
    wn_lo, wn_hi = min(lo, hi), max(lo, hi)
    if not (wn_lo > 0 and wn_hi > 0):
        return None

    if units == 'nm':
        conv = _fn
        ticks_cand = _WL_TICKS_NM
        xlabel = 'Wavelength (nm)'
    else:
        conv = _fw
        ticks_cand = _WL_TICKS_UM
        xlabel = 'Wavelength (µm)'

    secax = ax.secondary_xaxis('top', functions=(conv, conv))

    # Filter candidate ticks to those within the visible window
    wl_lo, wl_hi = sorted([conv(wn_lo), conv(wn_hi)])
    visible = [t for t in ticks_cand if wl_lo <= t <= wl_hi]
    if len(visible) > _WL_TICKS_MAX:
        step = (len(visible) + _WL_TICKS_MAX - 1) // _WL_TICKS_MAX
        visible = visible[::step]
    if visible:
        secax.xaxis.set_ticks(visible)

    secax.set_xlabel(xlabel, fontsize=9)
    return secax


# ---------------------------------------------------------------------------
# Quantity metadata
# ---------------------------------------------------------------------------

_QUANTITY_LABELS: dict[str, str] = {
    'MEC': r'MEC  (cm$^2$ g$^{-1}$)',
    'MSC': r'MSC  (cm$^2$ g$^{-1}$)',
    'MAC': r'MAC  (cm$^2$ g$^{-1}$)',
}
_VALID_QUANTITIES = frozenset(_QUANTITY_LABELS)


def _check_quantity(quantity: str) -> None:
    if quantity not in _VALID_QUANTITIES:
        raise ValueError(
            f"quantity must be one of {sorted(_VALID_QUANTITIES)!r}; "
            f"got {quantity!r}."
        )


# ---------------------------------------------------------------------------
# plot_spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    wn: np.ndarray,
    data: np.ndarray,
    *,
    labels: list[str] | None = None,
    quantity: str = 'MEC',
    title: str | None = None,
    top_axis: bool = True,
    top_units: str = 'um',
    colors: list | None = None,
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot one or more mass coefficient spectra vs wavenumber.

    Parameters
    ----------
    wn : np.ndarray, shape (N_wn,)
        Wavenumber grid [cm⁻¹].
    data : np.ndarray, shape (N_wn,) or (N_spectra, N_wn)
        Spectrum or stack of spectra.
    labels : list[str] or None
        Legend labels, one per spectrum.  Omitted if ``None``.
    quantity : {'MEC', 'MSC', 'MAC'}
        Y-axis label selector.  Default ``'MEC'``.
    title : str or None
        Figure title.
    top_axis : bool
        If True (default), attach a wavelength axis on top.
    top_units : {'um', 'nm'}
        Units for the secondary top axis.
    colors : list or None
        Matplotlib colour specifications, one per spectrum.  Cycles through
        the default colour cycle if ``None``.
    fig : plt.Figure or None
        Existing figure to draw into (for GUI embedding).  A new figure is
        created when ``None``.
    show : bool
        Call ``plt.show()`` on exit.  Set to ``False`` for embedding.
    save_path : str or Path or None
        Save path.  Skipped if ``None``.

    Returns
    -------
    plt.Figure
    """
    _check_quantity(quantity)

    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n_spectra = data.shape[0]

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.gca()

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(n_spectra):
        color = colors[i] if colors is not None else prop_cycle[i % len(prop_cycle)]
        label = labels[i] if labels is not None else None
        ax.plot(wn, data[i], color=color, label=label, alpha=0.9)

    ax.set_xlim(float(wn.max()), float(wn.min()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax.set_ylabel(_QUANTITY_LABELS[quantity], fontsize=9)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if labels is not None:
        ax.legend(fontsize=8)

    if top_axis:
        _add_top_axis(ax, units=top_units)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# plot_lut_sweep
# ---------------------------------------------------------------------------

def plot_lut_sweep(
    lut: 'AerosolLUT',  # noqa: F821 — forward ref avoids circular import
    *,
    quantity: str = 'MEC',
    r0_um: np.ndarray | None = None,
    n_plot: int = 200,
    title: str | None = None,
    top_axis: bool = True,
    top_units: str = 'um',
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a dense r0 sweep from a single :class:`~mieEODG.AerosolLUT`.

    Spectra are coloured by r0 using the category-specific colormap from
    :func:`~mieEODG.constants.get_cmap`.

    Parameters
    ----------
    lut : AerosolLUT
        Pre-computed LUT.
    quantity : {'MEC', 'MSC', 'MAC'}
        Which mass coefficient to plot.
    r0_um : np.ndarray or None
        r0 values [µm] to query.  A log-spaced grid of *n_plot* points
        spanning the LUT bounds is used when ``None``.
    n_plot : int
        Number of r0 samples when *r0_um* is ``None``.
    title : str or None
        Figure title.  Defaults to ``lut.substance`` or ``lut.category``.
    top_axis : bool
        Attach a wavelength top axis.
    top_units : {'um', 'nm'}
        Units for the secondary top axis.
    fig : plt.Figure or None
        Existing figure for GUI embedding.
    show : bool
        Call ``plt.show()`` on exit.
    save_path : str or Path or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    from .constants import get_cmap

    _check_quantity(quantity)

    if r0_um is None:
        r0_um = np.logspace(
            np.log10(lut.r0_um[0]),
            np.log10(lut.r0_um[-1]),
            n_plot,
        )

    norm  = mcolors.LogNorm(vmin=float(r0_um[0]), vmax=float(r0_um[-1]))
    cmap  = get_cmap(lut.category or '')

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.gca()

    wn = lut.wavenumber
    for r in r0_um:
        fwd = lut.forward(r)
        ax.plot(wn, fwd[quantity], color=cmap(norm(r)), lw=0.7, alpha=0.8)

    ax.set_xlim(float(wn.max()), float(wn.min()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax.set_ylabel(_QUANTITY_LABELS[quantity], fontsize=9)

    _title = title if title is not None else (lut.substance or lut.category or '')
    if _title:
        ax.set_title(_title, fontsize=10)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04, aspect=20)
    cbar.set_label('$r_0$ (µm)', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.2g}')
    )

    if top_axis:
        _add_top_axis(ax, units=top_units)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# plot_lut
# ---------------------------------------------------------------------------

def plot_lut(
    luts: Union[dict, list],
    *,
    quantity: str = 'MEC',
    r0_um: np.ndarray | None = None,
    n_plot: int = 200,
    n_cols: int = 3,
    suptitle: str | None = None,
    top_axis: bool = True,
    top_units: str = 'um',
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Multi-panel LUT mass coefficient survey.

    Replicates the layout of Part 2 in ``tests/demo_LUT.py``, but accepts
    any number of LUTs and layouts automatically.

    Parameters
    ----------
    luts : dict[str, AerosolLUT] or list[AerosolLUT]
        When a **dict**, keys are used as panel titles.
        When a **list**, ``lut.substance`` (falling back to ``lut.category``)
        is used as the panel title.
    quantity : {'MEC', 'MSC', 'MAC'}
        Which mass coefficient to plot.  Default ``'MEC'``.
    r0_um : np.ndarray or None
        r0 query grid [µm].  When ``None``, each panel uses a log-spaced
        grid of *n_plot* points spanning that LUT's own bounds.
    n_plot : int
        Number of r0 samples when *r0_um* is ``None``.
    n_cols : int
        Number of columns in the panel grid.  Default 3.
    suptitle : str or None
        Overall figure title.
    top_axis : bool
        Attach wavelength top axis to each panel.
    top_units : {'um', 'nm'}
        Units for the secondary top axes.
    fig : plt.Figure or None
        Existing figure for GUI embedding.
    show : bool
        Call ``plt.show()`` on exit.
    save_path : str or Path or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    from .constants import get_cmap

    _check_quantity(quantity)

    # Unpack into (title, lut) pairs
    if isinstance(luts, dict):
        items: list[tuple[str, 'AerosolLUT']] = list(luts.items())
    else:
        items = [
            (lut.substance or lut.category or f'LUT {i}', lut)
            for i, lut in enumerate(luts)
        ]

    n_panels = len(items)
    if n_panels == 0:
        raise ValueError("luts must contain at least one entry.")

    n_cols   = min(n_cols, n_panels)
    n_rows   = int(np.ceil(n_panels / n_cols))
    figsize  = (5.5 * n_cols, 4.5 * n_rows)

    if fig is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    else:
        axes = np.asarray(fig.axes).reshape(n_rows, n_cols)

    axes_flat = np.asarray(axes).ravel()

    # Hide unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for idx, (panel_title, lut) in enumerate(items):
        ax   = axes_flat[idx]
        wn   = lut.wavenumber
        cmap = get_cmap(lut.category or '')

        if r0_um is None:
            r0_sweep = np.logspace(
                np.log10(lut.r0_um[0]),
                np.log10(lut.r0_um[-1]),
                n_plot,
            )
        else:
            r0_sweep = np.asarray(r0_um, dtype=np.float64)

        norm = mcolors.LogNorm(
            vmin=float(r0_sweep[0]),
            vmax=float(r0_sweep[-1]),
        )

        for r in r0_sweep:
            fwd = lut.forward(r)
            ax.plot(wn, fwd[quantity], color=cmap(norm(r)), lw=0.7, alpha=0.8)

        ax.set_xlim(float(wn.max()), float(wn.min()))
        ax.set_ylim(bottom=0)
        ax.set_title(panel_title, fontsize=10)
        ax.set_ylabel(_QUANTITY_LABELS[quantity], fontsize=9)
        ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)

        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.05, aspect=20)
        cbar.set_label('$r_0$ (µm)', fontsize=9)
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2g}')
        )

        if top_axis:
            _add_top_axis(ax, units=top_units)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=10)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()
    return fig
