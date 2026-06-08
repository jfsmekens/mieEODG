"""
Plotting utilities for mieEODG.

Provides publication-ready figures for Mie scattering results with a
consistent style: thin grid lines, a secondary top axis showing wavelength,
and cmcrameri colormaps keyed to ARIA species categories.

Public functions
----------------
plot_spectrum           Plot one or more spectra (MEC / MSC / MAC vs wavenumber).
plot_lut_sweep         Plot a dense r0 sweep from a single AerosolLUT.
plot_lut               Multi-panel LUT survey from a dict or list of AerosolLUT objects.
plot_lut_slice         Plot a single slice (composable) through 1-D or N-D LUTs.
plot_lut_heatmap            Plot optical properties as 2-D heatmap (parameter trade-offs).
plot_size_dist              Plot particle size distribution PDF (dN/d ln r vs radius).
plot_phase_function         Plot phase function F(θ) and asymmetry parameter g(λ) side-by-side.
plot_phase_function_heatmap Plot phase function as 2-D heatmap (angle vs wavelength).

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
    'font.size':                   11,
    'axes.labelsize':              9,
    'axes.titlesize':              12,
    'xtick.labelsize':             9,
    'xtick.direction':             'in',
    'ytick.labelsize':             9,
    'ytick.direction':             'in',
    'lines.linewidth':             0.9,
    'axes.grid':                   True,
    'grid.linewidth':              0.5,
    'grid.alpha':                  0.5,
    'grid.linestyle':              '--',
    'axes.axisbelow':              True,
    'figure.constrained_layout.use': True,
    'axes.formatter.limits':       (-4, 4),
    'errorbar.capsize':             2,
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

# Radius tick candidates for the size distribution x-axis (µm)
_R_TICKS_UM: list[float] = [
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.5,
    1, 2, 5,
    10, 20, 50,
    100, 200, 500,
]
_R_TICKS_MAX: int = 10

_QUANTITY_LABELS: dict[str, str] = {
    # Mass coefficients [cm²/g]
    'MEC': r'MEC  (cm$^2$ g$^{-1}$)',
    'MSC': r'MSC  (cm$^2$ g$^{-1}$)',
    'MAC': r'MAC  (cm$^2$ g$^{-1}$)',
    # Efficiency factors [dimensionless]
    'Q_ext': r'$Q_{ext}$', 'Q_sca': r'$Q_{sca}$', 'Q_abs': r'$Q_{abs}$',
    # Mean per-particle cross-sections [cm²]
    'C_ext': r'$C_{ext}$  (cm$^2$)', 'C_sca': r'$C_{sca}$  (cm$^2$)',
    'C_abs': r'$C_{abs}$  (cm$^2$)',
    # Volume coefficients [cm⁻¹]  (LUT built at Nd=1; scale by your Nd)
    'B_ext': r'$B_{ext}$  (cm$^{-1}$)', 'B_sca': r'$B_{sca}$  (cm$^{-1}$)',
    'B_abs': r'$B_{abs}$  (cm$^{-1}$)',
    # Asymmetry parameter [dimensionless]
    'g': r'$g$',
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


# ---------------------------------------------------------------------------
# plot_size_dist
# ---------------------------------------------------------------------------

def plot_size_dist(
    dists,
    *,
    r_range_um:  tuple[float, float] | None = None,
    n_pts:       int = 500,
    normalize:   bool = True,
    category:    str | None = None,
    labels:      list[str] | None = None,
    yscale:      str = 'linear',
    xscale:      str = 'log',
    style:       str = 'line',
    color:       str | None = None,
    title:       str | None = None,
    fig:         plt.Figure | None = None,
    show:        bool = True,
    save_path:   str | Path | None = None,
) -> plt.Figure:
    """
    Plot particle size distribution(s) as dN/d ln r vs radius.

    The y-axis shows ``r · n(r)`` — the number size distribution in
    logarithmic radius space — which gives the familiar bell-shaped curve
    on a log-radius axis.  Each curve is optionally normalised to its own
    peak so shapes can be compared regardless of total number density.

    Parameters
    ----------
    dists : SizeDistribution or list or dict
        A single distribution object, a list of distributions, or a
        ``dict`` mapping string labels to distributions.  When a list
        contains distributions with an ``r0_um`` attribute (e.g. a sweep
        of :class:`~mieEODG.LogNormalDist` or
        :class:`~mieEODG.size_dist.GaussianDist`), curves are coloured by
        r0 using the category colormap.
    r_range_um : (float, float) or None
        Explicit ``(r_min, r_max)`` in µm.  When ``None``, the range is
        derived automatically from the ``bounds()`` of each distribution
        with 20 % log-space padding on each side.
    n_pts : int
        Number of radius points for evaluation.  Default 500.
    normalize : bool
        If True (default), normalise each curve to its own peak so that
        the maximum is 1.  Useful for comparing shapes across distributions
        with different total number densities.
    category : str or None
        ARIA category key for colormap selection (e.g. ``'volcanic_ash'``).
        Used when ``dists`` is a list and distributions do not carry
        category metadata.  Falls back to the default colour cycle if not
        supplied.
    labels : list[str] or None
        Legend labels, one per distribution.  When ``None``, labels are
        derived from dict keys (if a dict) or ``repr(dist)`` (if a list).
        Pass an empty list to suppress the legend entirely.
    yscale : {'linear', 'log'}
        Y-axis scale.  Default ``'linear'``.
    style : {'line', 'shaded', 'hist'}
        Plot style:
        - ``'line'`` (default): line plot
        - ``'shaded'``: filled area (semi-transparent)
        - ``'hist'``: histogram bars
    title : str or None
        Figure title.
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
    from .size_dist import SizeDistribution

    # --- Normalise input to list of (label, dist) pairs ---
    if isinstance(dists, SizeDistribution):
        items = [(repr(dists), dists)]
    elif isinstance(dists, dict):
        items = list(dists.items())
    else:
        items = [(repr(d), d) for d in dists]

    if labels is not None and len(labels) > 0:
        items = [(lab, d) for (_, d), lab in zip(items, labels)]

    n_dists = len(items)
    if n_dists == 0:
        raise ValueError("dists must contain at least one distribution.")

    # --- Determine global radius range ---
    if r_range_um is not None:
        r_lo, r_hi = float(r_range_um[0]), float(r_range_um[1])
    else:
        # For monodisperse (delta) dists, bounds() is a tight ±1 % bracket;
        # widen to a half-decade window so the stem sits mid-axis.
        bounds_cm = []
        for _, d in items:
            if getattr(d, 'dist_name', None) == 'monodisperse':
                bounds_cm.append((d.r0 * 0.3, d.r0 * 3.0))
            else:
                bounds_cm.append(d.bounds())
        r_lo = min(b[0] for b in bounds_cm) * 1e4    # cm → µm
        r_hi = max(b[1] for b in bounds_cm) * 1e4
        # 20 % log-space padding
        log_pad = 0.2 * (np.log10(r_hi) - np.log10(r_lo))
        r_lo = 10 ** (np.log10(r_lo) - log_pad)
        r_hi = 10 ** (np.log10(r_hi) + log_pad)

    r_um = np.logspace(np.log10(r_lo), np.log10(r_hi), n_pts)
    r_cm = r_um * 1e-4

    # --- Color scheme ---
    # An explicit `color` overrides everything (all curves drawn in it).
    # Otherwise: category colormap keyed by r0_um for multi-curve sweeps, else
    # the default property cycle.
    has_r0 = all(hasattr(d, 'r0_um') for _, d in items)
    if color is not None:
        colors  = [color] * n_dists
        norm_r0 = None
        cmap    = None
    elif has_r0 and n_dists > 1:
        r0_vals = np.array([d.r0_um for _, d in items])
        norm_r0 = mcolors.LogNorm(vmin=r0_vals.min(), vmax=r0_vals.max())
        cmap    = get_cmap(category or '')
        colors  = [cmap(norm_r0(r0)) for r0 in r0_vals]
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors     = [prop_cycle[i % len(prop_cycle)] for i in range(n_dists)]
        norm_r0    = None
        cmap       = None

    # --- Validate parameters ---
    if yscale not in ('linear', 'log'):
        raise ValueError(f"yscale must be 'linear' or 'log'; got {yscale!r}")
    if style not in ('line', 'shaded', 'hist'):
        raise ValueError(f"style must be 'line', 'shaded', or 'hist'; got {style!r}")

    # --- Figure ---
    if fig is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        ax = fig.gca()

    for i, (label, dist) in enumerate(items):
        # Monodisperse (delta): draw a vertical stem at r₀ rather than a curve.
        if getattr(dist, 'dist_name', None) == 'monodisperse':
            height = 1.0 if normalize else dist.N
            ax.vlines(dist.r0_um, 0.0, height, color=colors[i], lw=1.5, label=label)
            ax.plot(dist.r0_um, height, 'o', color=colors[i], ms=5)
            continue

        n_r    = dist.pdf(r_cm)          # n(r) in cm-based units
        dndlnr = n_r * r_cm              # dN/d ln r = r · n(r)
        if normalize:
            peak = dndlnr.max()
            if peak > 0:
                dndlnr = dndlnr / peak

        # Plot with selected style
        if style == 'line':
            ax.plot(r_um, dndlnr, color=colors[i], label=label, lw=1.0, alpha=0.9)
        elif style == 'shaded':
            ax.fill_between(r_um, dndlnr, color=colors[i], label=label, alpha=0.5)
            ax.plot(r_um, dndlnr, color=colors[i], lw=0.8, alpha=0.9)
        elif style == 'hist':
            ax.bar(r_um, dndlnr, width=np.diff(r_um).mean() * 0.8,
                   color=colors[i], label=label, alpha=0.7, edgecolor='none')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(r_lo, r_hi)
    if normalize and yscale == 'linear':
        ax.set_ylim(bottom=0, top=1.05)

    # Curated log-spaced x-axis ticks (only meaningful on a log x-axis).
    if xscale == 'log':
        visible_ticks = [t for t in _R_TICKS_UM if r_lo <= t <= r_hi]
        if len(visible_ticks) > _R_TICKS_MAX:
            step = (len(visible_ticks) + _R_TICKS_MAX - 1) // _R_TICKS_MAX
            visible_ticks = visible_ticks[::step]
        if visible_ticks:
            ax.xaxis.set_ticks(visible_ticks)
            ax.xaxis.set_ticklabels([f'{t:g}' for t in visible_ticks])

    ax.set_xlabel('Radius (µm)', fontsize=9)
    y_label = 'dN / d ln r' + (' (normalised)' if normalize else '')
    ax.set_ylabel(y_label, fontsize=9)

    if title is not None:
        ax.set_title(title, fontsize=10)

    # Colorbar for r0 sweeps; legend for explicitly labelled dists
    if has_r0 and n_dists > 1 and cmap is not None:
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm_r0)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04, aspect=20)
        cbar.set_label('$r_0$ (µm)', fontsize=9)
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2g}')
        )
    elif not (labels is not None and len(labels) == 0):
        ax.legend(fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# plot_lut_slice
# ---------------------------------------------------------------------------

def plot_lut_slice(
    lut: Union['AerosolLUT', 'MultiDimLUT'],  # noqa: F821
    *,
    sweep_axis: str = 'r0',
    fixed_axes: dict | None = None,
    quantity: str = 'MEC',
    n_points: int = 200,
    title: str | None = None,
    top_axis: bool = True,
    top_units: str = 'um',
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a single slice through a 1-D or N-D LUT.

    This is a **composable** function that plots one spectral panel across a
    single dimension (e.g., r₀ sweep, composition sweep, temperature sweep).
    For multi-panel figures, call this function multiple times with different
    sweep axes or fixed_axes values, and arrange the panels in your own layout.

    **For 1-D LUT** (``AerosolLUT``):
        *sweep_axis* must be ``'r0'`` (the only dimension).
        *fixed_axes* is ignored.

    **For N-D LUT** (``MultiDimLUT``):
        *sweep_axis* specifies which dimension to vary (e.g., ``'r0'``,
        ``'composition'``, ``'temperature_K'``).
        *fixed_axes* specifies the values to hold constant for all other
        dimensions (e.g., ``{'composition': 65, 'temperature_K': 250}``).

    Parameters
    ----------
    lut : AerosolLUT or MultiDimLUT
        Look-up table to query.
    sweep_axis : str
        Axis to sweep. Default ``'r0'``.
        For ``AerosolLUT``: must be ``'r0'``.
        For ``MultiDimLUT``: any axis name (e.g., ``'composition'``,
        ``'temperature_K'``).
    fixed_axes : dict or None
        For ``MultiDimLUT``, specifies values to hold constant on all axes
        except *sweep_axis*.  Ignored for ``AerosolLUT``.
        Example: ``{'composition': 65, 'temperature_K': 250}``.
    quantity : {'MEC', 'MSC', 'MAC'}
        Which mass coefficient to plot.  Default ``'MEC'``.
    n_points : int
        Number of samples along the sweep axis.  Default 200.
    title : str or None
        Panel title.  If ``None``, a title is generated from the sweep
        parameters.
    top_axis : bool
        Attach a wavelength top axis.
    top_units : {'um', 'nm'}
        Units for the secondary top axis.
    fig : plt.Figure or None
        Existing figure for embedding.  A new figure is created if ``None``.
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

    # Detect LUT type
    is_multidim = hasattr(lut, 'forward') and hasattr(lut, 'ri_constraint')
    is_1d = hasattr(lut, 'forward') and hasattr(lut, 'r0_um')

    if not (is_multidim or is_1d):
        raise TypeError(
            f"lut must be AerosolLUT or MultiDimLUT; got {type(lut).__name__}"
        )

    # Prepare fixed_axes for query construction
    if fixed_axes is None:
        fixed_axes = {}

    wn = lut.wavenumber
    cmap = get_cmap(getattr(lut, 'category', '') or '')

    # --- 1-D LUT (AerosolLUT) ---
    if is_1d:
        if sweep_axis != 'r0':
            raise ValueError(
                f"For 1-D LUT, sweep_axis must be 'r0'; got {sweep_axis!r}"
            )

        r0_min, r0_max = float(lut.r0_um[0]), float(lut.r0_um[-1])
        r0_sweep = np.logspace(np.log10(r0_min), np.log10(r0_max), n_points)
        norm = mcolors.LogNorm(vmin=r0_min, vmax=r0_max)

        # Generate title if not provided
        if title is None:
            substance = getattr(lut, 'substance', None) or getattr(lut, 'category', '')
            title = f"{substance} – r₀ sweep" if substance else "r₀ sweep"

        # Plot
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            ax = fig.gca()

        for r0 in r0_sweep:
            fwd = lut.forward(r0)
            ax.plot(wn, fwd[quantity], color=cmap(norm(r0)), lw=0.7, alpha=0.8)

        # Colorbar
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04, aspect=20)
        cbar.set_label('$r_0$ (µm)', fontsize=9)
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2g}')
        )

    # --- N-D LUT (MultiDimLUT) ---
    else:
        # Build the sweep range
        sweep_axis_idx = None
        sweep_min, sweep_max = None, None

        for i, axis in enumerate(lut.axes):
            if axis.name == sweep_axis:
                sweep_axis_idx = i
                sweep_min = float(axis.values[0])
                sweep_max = float(axis.values[-1])
                break

        if sweep_axis_idx is None:
            raise ValueError(
                f"sweep_axis '{sweep_axis}' not found. Available: "
                f"{[ax.name for ax in lut.axes]}"
            )

        # Generate sweep points
        if lut.axes[sweep_axis_idx].log_spacing:
            sweep_vals = np.logspace(
                np.log10(sweep_min), np.log10(sweep_max), n_points
            )
            norm = mcolors.LogNorm(vmin=sweep_min, vmax=sweep_max)
        else:
            sweep_vals = np.linspace(sweep_min, sweep_max, n_points)
            norm = mcolors.Normalize(vmin=sweep_min, vmax=sweep_max)

        # Generate title if not provided
        if title is None:
            fixed_str = ', '.join(
                f"{k}={v:.4g}" for k, v in fixed_axes.items()
            ) if fixed_axes else "–"
            substance = getattr(lut, 'substance', None) or ''
            title = f"{substance} – {sweep_axis} sweep ({fixed_str})"

        # Plot
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            ax = fig.gca()

        for val in sweep_vals:
            query = {sweep_axis: val}
            query.update(fixed_axes)
            fwd = lut.forward(**query)
            ax.plot(wn, fwd[quantity], color=cmap(norm(val)), lw=0.7, alpha=0.8)

        # Colorbar
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04, aspect=20)
        cbar.set_label(
            f'{sweep_axis} ({lut.axes[sweep_axis_idx].units})',
            fontsize=9
        )
        if lut.axes[sweep_axis_idx].log_spacing:
            cbar.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.2g}')
            )

    # Common axis setup
    ax.set_xlim(float(wn.max()), float(wn.min()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=9)
    ax.set_ylabel(_QUANTITY_LABELS[quantity], fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)

    if top_axis:
        _add_top_axis(ax, units=top_units)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# plot_phase_function
# ---------------------------------------------------------------------------

def plot_phase_function(
    xsc: 'AerosolXSC',  # noqa: F821
    *,
    n_wavelengths: int = 5,
    wavelength_indices: list[int] | None = None,
    quantity: str = 'F11',
    cmap: str | None = None,
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot phase function and asymmetry parameter side-by-side.

    The left panel shows F11 (or other phase matrix component) vs scattering
    angle θ for multiple wavelengths, colored by wavelength. The right panel
    shows the asymmetry parameter g(λ) across the full spectrum, with markers
    at the selected wavelengths for cross-reference.

    The asymmetry parameter g is derived from the phase function:
    g = <cos(θ)> = weighted average scattering angle, where:
      - g ≈ +1: forward scattering
      - g ≈  0: isotropic scattering
      - g ≈ -1: backscattering

    Parameters
    ----------
    xsc : AerosolXSC
        Computed AerosolXSC object with phase function data.
        Must have been computed with ``theta_pts`` and phase data.
    n_wavelengths : int
        Number of representative wavelengths to plot in the left panel.
        Evenly spaced across the spectrum if ``wavelength_indices`` is ``None``.
        Default 5.
    wavelength_indices : list[int] or None
        Explicit wavenumber indices to plot. If ``None``, evenly spaced
        indices are selected. Example: ``[0, 100, 200, 300, 400]``.
    quantity : str
        Phase matrix component to plot: 'F11', 'F33', 'F12', or 'F34'.
        Default 'F11' (scattering matrix element).
    cmap : str or None
        Colormap for wavelength colors. If ``None`` (default), uses
        species-specific colormap if available, else 'viridis'.
        Pass an explicit name (e.g., 'hot', 'cool') to override.
    fig : plt.Figure or None
        Existing figure for embedding. A new figure is created if ``None``.
    show : bool
        Call ``plt.show()`` on exit.
    save_path : str or Path or None
        Save path.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If phase function data is not available in ``xsc``.
    """
    from .constants import get_cmap as get_category_cmap

    # Validate phase data
    if not hasattr(xsc, 'phase_matrix') or xsc.phase_matrix is None:
        raise ValueError(
            "Phase function data not available. "
            "Compute with: xsc.compute(theta_pts=..., n_legendre=...)"
        )

    if quantity not in ('F11', 'F33', 'F12', 'F34'):
        raise ValueError(
            f"quantity must be 'F11', 'F33', 'F12', or 'F34'; got {quantity!r}"
        )

    # Map quantity to phase_matrix column
    quantity_idx = {'F11': 0, 'F33': 1, 'F12': 2, 'F34': 3}[quantity]

    # Extract data
    wn = xsc.wn          # shape (n_wn,)
    theta = xsc.theta    # shape (n_theta,)
    phase_matrix = xsc.phase_matrix  # shape (n_wn, n_theta, 4)
    g = xsc.g            # shape (n_wn,)

    n_wn = len(wn)

    # Select wavelength indices
    if wavelength_indices is None:
        # Evenly spaced indices
        wavelength_indices = np.linspace(0, n_wn - 1, n_wavelengths, dtype=int)
    else:
        wavelength_indices = np.asarray(wavelength_indices, dtype=int)

    # Convert wavenumber to wavelength for colormap
    wl_um = 1e4 / wn  # cm⁻¹ to µm
    wl_selected = wl_um[wavelength_indices]
    wl_min, wl_max = wl_um.min(), wl_um.max()
    norm_wl = mcolors.Normalize(vmin=wl_min, vmax=wl_max)

    # Use species-specific colormap if not explicitly provided
    if cmap is None:
        category = getattr(xsc.ri, 'category', None)
        if category:
            cmap = get_category_cmap(category)
        else:
            cmap = 'viridis'

    cmap = plt.get_cmap(cmap)

    # Create figure
    if fig is None:
        fig, (ax_phase, ax_g) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        axes = fig.axes
        if len(axes) < 2:
            raise ValueError("Figure must have at least 2 axes")
        ax_phase, ax_g = axes[0], axes[1]

    # --- Left panel: Phase function curves ---
    for idx in wavelength_indices:
        f_theta = phase_matrix[idx, :, quantity_idx]
        color = cmap(norm_wl(wl_um[idx]))
        ax_phase.semilogy(
            theta, f_theta,
            color=color,
            lw=1.5,
            alpha=0.9,
            label=f'{wl_um[idx]:.2f} µm'
        )

    ax_phase.set_xlabel('Scattering angle θ (°)', fontsize=10)
    ax_phase.set_ylabel(f'{quantity}(θ)', fontsize=10)
    ax_phase.set_title(f'Phase Function ({quantity})', fontsize=11)
    ax_phase.grid(True, alpha=0.3, which='both', lw=0.5)
    ax_phase.legend(fontsize=8, loc='best', title='Wavelength')
    ax_phase.set_xlim(0, 180)

    # --- Right panel: Asymmetry parameter g(λ) ---
    wl_all = 1e4 / wn
    ax_g.plot(wl_all, g, color='steelblue', lw=1.2, alpha=0.8, label='g(λ)')

    # Mark selected wavelengths
    g_selected = g[wavelength_indices]
    colors_selected = [cmap(norm_wl(wl)) for wl in wl_selected]
    ax_g.scatter(wl_selected, g_selected, c=colors_selected, s=100, zorder=5,
                 edgecolors='black', linewidths=0.5)

    ax_g.axhline(0, color='gray', linestyle='--', lw=0.7, alpha=0.5)
    ax_g.set_xlabel('Wavelength (µm)', fontsize=10)
    ax_g.set_ylabel('Asymmetry parameter g', fontsize=10)
    ax_g.set_title('Asymmetry Parameter g(λ)', fontsize=11)
    ax_g.grid(True, alpha=0.3, lw=0.5)
    ax_g.set_ylim(-1.1, 1.1)
    ax_g.set_xlim(wl_min, wl_max)

    # Overall title
    substance = getattr(xsc, 'substance', None) or getattr(xsc.ri, 'substance', '')
    fig.suptitle(
        f'{substance} – Phase Function and Asymmetry Parameter',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# plot_lut_heatmap
# ---------------------------------------------------------------------------

def plot_lut_heatmap(
    lut: Union['AerosolLUT', 'MultiDimLUT'],  # noqa: F821
    *,
    x_axis: str = 'wavelength',
    y_axis: str = 'radius',
    fixed_axes: dict | None = None,
    quantity: str = 'MEC',
    wavelength_um: float | None = None,
    n_points_x: int = 100,
    n_points_y: int = 50,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = True,
    contours: bool = False,
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot optical properties as a 2-D heatmap across two parameter dimensions.

    Creates a 2-D grid of optical property values (e.g., MEC) as a function
    of two axes (e.g., wavelength and particle radius). Reveals trade-offs
    and interactions between parameters at a glance.

    **For 1-D LUT** (``AerosolLUT``):
        Axes can be 'wavelength' and 'radius' (the two natural dimensions).

    **For N-D LUT** (``MultiDimLUT``):
        Axes can be any combination (e.g., 'composition', 'temperature_K', 'radius').
        Other dimensions are fixed via *fixed_axes*.

    Parameters
    ----------
    lut : AerosolLUT or MultiDimLUT
        Look-up table to query.
    x_axis : str
        Horizontal axis: 'wavelength', 'radius', 'composition', 'temperature_K', etc.
        Default 'wavelength'.
    y_axis : str
        Vertical axis. Default 'radius'.
    fixed_axes : dict or None
        For ``MultiDimLUT``, values to hold constant on all axes except
        *x_axis* and *y_axis*. Example: ``{'temperature_K': 250}``.
    quantity : {'MEC', 'MSC', 'MAC'}
        Which optical property to plot. Default 'MEC'.
    n_points_x : int
        Number of points along x-axis. Default 100.
    n_points_y : int
        Number of points along y-axis. Default 50.
    cmap : str or None
        Colormap name (e.g., 'viridis', 'hot', 'RdYlBu', 'cmc.davos').
        If ``None`` (default), uses species-specific colormap if available
        (e.g., 'cmc.grayC_r' for volcanic ash), else 'viridis'.
        Pass an explicit name to override the species default.
    vmin : float or None
        Colorbar minimum. If ``None``, uses data minimum.
    vmax : float or None
        Colorbar maximum. If ``None``, uses data maximum.
    log_scale : bool
        Apply log scaling to both axes (if logarithmic). Default True.
    contours : bool
        Overlay contour lines on the heatmap. Default False.
    fig : plt.Figure or None
        Existing figure for embedding. A new figure is created if ``None``.
    show : bool
        Call ``plt.show()`` on exit.
    save_path : str or Path or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    from .constants import get_cmap as get_category_cmap

    _check_quantity(quantity)

    if fixed_axes is None:
        fixed_axes = {}

    # Detect LUT type
    is_1d = hasattr(lut, 'r0_um') and not hasattr(lut, 'ri_constraint')
    is_multidim = hasattr(lut, 'ri_constraint') and hasattr(lut, 'axes')

    if not (is_1d or is_multidim):
        raise TypeError(f"lut must be AerosolLUT or MultiDimLUT; got {type(lut).__name__}")

    # Axis name mapping and defaults
    axis_info = {
        'wavelength': {'log': True, 'units': 'µm'},
        'radius': {'log': True, 'units': 'µm'},
        'composition': {'log': False, 'units': 'wt%'},
        'temperature_K': {'log': False, 'units': 'K'},
    }

    if x_axis not in axis_info or y_axis not in axis_info:
        raise ValueError(
            f"Axes must be from {list(axis_info.keys())}; "
            f"got x_axis={x_axis!r}, y_axis={y_axis!r}"
        )

    # --- Build query grids ---
    wn = lut.wn if hasattr(lut, 'wn') else lut.wavenumber
    wl_um = 1e4 / wn

    # Map axis names to values
    axis_vals = {}

    if is_1d:
        axis_vals['radius'] = lut.r0_um
        axis_vals['wavelength'] = wl_um
    else:  # multidim
        axis_vals['wavelength'] = wl_um   # spectral dim is always available
        for axis in lut.axes:
            if axis.name == 'r0':
                axis_vals['radius'] = axis.values
            else:
                axis_vals[axis.name] = axis.values

    # Generate grid points
    if axis_info[x_axis]['log']:
        x_vals = np.logspace(
            np.log10(axis_vals[x_axis].min()),
            np.log10(axis_vals[x_axis].max()),
            n_points_x
        )
    else:
        x_vals = np.linspace(
            axis_vals[x_axis].min(),
            axis_vals[x_axis].max(),
            n_points_x
        )

    if axis_info[y_axis]['log']:
        y_vals = np.logspace(
            np.log10(axis_vals[y_axis].min()),
            np.log10(axis_vals[y_axis].max()),
            n_points_y
        )
    else:
        y_vals = np.linspace(
            axis_vals[y_axis].min(),
            axis_vals[y_axis].max(),
            n_points_y
        )

    # Create 2-D grid
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Evaluate LUT at each grid point
    if is_1d:
        # For 1-D LUT: forward() takes r0_um, so we need to manually index wavelengths
        for i in range(n_points_y):
            for j in range(n_points_x):
                r0_val = Y[i, j] if y_axis == 'radius' else X[i, j]
                result = lut.forward(r0_val)
                # Extract value at the desired wavelength
                if x_axis == 'wavelength':
                    wl_query = X[i, j]
                    # Find nearest wavelength index
                    wl_idx = np.argmin(np.abs(wl_um - wl_query))
                elif y_axis == 'wavelength':
                    wl_query = Y[i, j]
                    wl_idx = np.argmin(np.abs(wl_um - wl_query))
                else:
                    wl_idx = 0
                Z[i, j] = result[quantity][wl_idx]
    else:
        # For N-D LUT: forward() takes keyword arguments and returns the full
        # spectrum, so we must (a) translate the display axis name 'radius'→'r0',
        # (b) when an axis IS wavelength, pick that spectral index, and (c) when
        # NEITHER axis is wavelength, evaluate at a fixed wavelength.
        def _qname(disp_axis):
            return 'r0' if disp_axis == 'radius' else disp_axis

        # Fixed-wavelength index for the no-wavelength-axis case.
        if wavelength_um is not None:
            fixed_wl_idx = int(np.argmin(np.abs(wl_um - wavelength_um)))
        else:
            fixed_wl_idx = len(wl_um) // 2   # middle of the band by default

        # Translate any 'radius' key in fixed_axes to 'r0'.
        fixed_q = {_qname(k): v for k, v in fixed_axes.items()}

        for i in range(n_points_y):
            for j in range(n_points_x):
                query = dict(fixed_q)
                wl_idx = None
                if x_axis == 'wavelength':
                    wl_idx = int(np.argmin(np.abs(wl_um - X[i, j])))
                else:
                    query[_qname(x_axis)] = X[i, j]
                if y_axis == 'wavelength':
                    wl_idx = int(np.argmin(np.abs(wl_um - Y[i, j])))
                else:
                    query[_qname(y_axis)] = Y[i, j]
                spectrum = lut.forward(**query)[quantity]   # (n_wn,)
                Z[i, j] = spectrum[wl_idx if wl_idx is not None else fixed_wl_idx]

    # Create figure
    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=False)
    else:
        ax = fig.gca()

    # Use species-specific colormap if not explicitly provided
    if cmap is None:
        # Get category-specific colormap
        category = getattr(lut, 'category', None) or getattr(
            getattr(lut, 'ri_constraint', None), 'category', None
        )
        if category:
            cmap = get_category_cmap(category)
        else:
            cmap = 'viridis'

    # Plot heatmap
    im = ax.pcolormesh(
        X, Y, Z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

    # Axis scaling
    if axis_info[x_axis]['log']:
        ax.set_xscale('log')
    if axis_info[y_axis]['log']:
        ax.set_yscale('log')

    # Labels and formatting
    ax.set_xlabel(f'{x_axis.title()} ({axis_info[x_axis]["units"]})', fontsize=11)
    ax.set_ylabel(f'{y_axis.title()} ({axis_info[y_axis]["units"]})', fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(_QUANTITY_LABELS.get(quantity, quantity), fontsize=10)

    # Contours (optional)
    if contours:
        contour_levels = np.linspace(Z.min(), Z.max(), 8)
        ax.contour(X, Y, Z, levels=contour_levels, colors='white',
                  linewidths=0.5, alpha=0.4)

    # Title
    substance = getattr(lut, 'substance', None) or getattr(lut, 'category', '')
    if fixed_axes:
        fixed_str = ', '.join(f'{k}={v:.4g}' for k, v in fixed_axes.items())
        title = f'{substance} – {quantity}({x_axis}, {y_axis}) | Fixed: {fixed_str}'
    else:
        title = f'{substance} – {quantity}({x_axis}, {y_axis})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    # The colorbar installs a layout engine; subplots_adjust would conflict
    # with it (emitting a UserWarning).  Only apply manual spacing when no
    # layout engine is active.
    if fig.get_layout_engine() is None:
        fig.subplots_adjust(bottom=0.12, left=0.12, right=0.92)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# plot_phase_function_heatmap
# ---------------------------------------------------------------------------

def plot_phase_function_heatmap(
    xsc: 'AerosolXSC',  # noqa: F821
    *,
    quantity: str = 'F11',
    cmap: str | None = None,
    log_scale: bool = True,
    theta_smooth: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    fig: plt.Figure | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot phase function as a 2-D heatmap: scattering angle vs wavelength.

    Shows how the phase matrix component (F11, F33, F12, or F34) evolves
    across both scattering angle and wavelength simultaneously. Reveals
    spectral dependence of phase function shape and magnitudes.

    Parameters
    ----------
    xsc : AerosolXSC
        Computed AerosolXSC object with phase function data.
        Must have been computed with ``theta_pts`` and phase data.
    quantity : {'F11', 'F33', 'F12', 'F34'}
        Phase matrix component to plot. Default 'F11' (intensity).
        - F11: total intensity (most commonly used)
        - F33: perpendicular polarization component
        - F12: linear polarization mixing term
        - F34: circular polarization term
    cmap : str or None
        Colormap name (e.g., 'viridis', 'hot', 'RdYlBu').
        If ``None``, uses species-specific colormap if available.
    log_scale : bool
        Apply log scaling to color values. Useful for revealing both
        forward-scattering peaks and backscatter details. Default True.
    theta_smooth : int or None
        Number of interpolation points for smoothing scattering angle axis.
        If ``None`` (default), uses original angle grid. If specified,
        interpolates to this many points for a smoother appearance.
        Example: ``theta_smooth=180`` gives 1-degree resolution.
    vmin : float or None
        Colorbar minimum value. If ``None``, uses data minimum.
        Applied after log scaling if ``log_scale=True``.
    vmax : float or None
        Colorbar maximum value. If ``None``, uses data maximum.
        Applied after log scaling if ``log_scale=True``.
    fig : plt.Figure or None
        Existing figure for embedding. A new figure is created if ``None``.
    show : bool
        Call ``plt.show()`` on exit.
    save_path : str or Path or None
        Save path.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If phase function data is not available or invalid quantity.
    """
    from .constants import get_cmap as get_category_cmap

    # Validate phase data
    if not hasattr(xsc, 'phase_matrix') or xsc.phase_matrix is None:
        raise ValueError(
            "Phase function data not available. "
            "Compute with: xsc.compute(theta_pts=..., n_legendre=...)"
        )

    if quantity not in ('F11', 'F33', 'F12', 'F34'):
        raise ValueError(
            f"quantity must be 'F11', 'F33', 'F12', or 'F34'; got {quantity!r}"
        )

    # Extract data
    wn = xsc.wn          # shape (n_wn,)
    theta = xsc.theta    # shape (n_theta,)
    phase_matrix = xsc.phase_matrix  # shape (n_wn, n_theta, 4)

    # Map quantity to phase_matrix column
    quantity_idx = {'F11': 0, 'F33': 1, 'F12': 2, 'F34': 3}[quantity]

    # Extract the requested component
    f_component = phase_matrix[:, :, quantity_idx]  # shape (n_wn, n_theta)

    # Convert wavenumber to wavelength for display
    wl_um = 1e4 / wn  # cm⁻¹ to µm

    # Optionally interpolate scattering angle axis for smoother appearance
    if theta_smooth is not None and theta_smooth > len(theta):
        from scipy.interpolate import interp1d
        # Interpolate along angle axis for each wavelength
        f_interp = np.zeros((len(wl_um), theta_smooth))
        for i in range(len(wl_um)):
            interp_func = interp1d(theta, f_component[i, :], kind='cubic')
            theta_fine = np.linspace(theta[0], theta[-1], theta_smooth)
            f_interp[i, :] = interp_func(theta_fine)
        f_component = f_interp
        theta = np.linspace(theta[0], theta[-1], theta_smooth)

    # Create 2-D grids for pcolormesh
    Theta, Wl = np.meshgrid(theta, wl_um)

    # Apply log scaling if requested (useful for wide dynamic range)
    if log_scale:
        # Avoid log(0) by clipping to small positive value
        f_plot = np.where(f_component > 0, f_component, 1e-6)
        f_plot = np.log10(f_plot)
    else:
        f_plot = f_component

    # Create figure
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    else:
        ax = fig.gca()

    # Use species-specific colormap if not explicitly provided
    if cmap is None:
        category = getattr(xsc.ri, 'category', None)
        if category:
            cmap = get_category_cmap(category)
        else:
            cmap = 'viridis'

    # Plot heatmap
    im = ax.pcolormesh(
        Theta, Wl, f_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

    # Labels and formatting
    ax.set_xlabel('Scattering angle θ (°)', fontsize=11)
    ax.set_ylabel('Wavelength (µm)', fontsize=11)
    ax.set_xlim(0, 180)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    if log_scale:
        cbar.set_label(f'log₁₀({quantity}(θ, λ))', fontsize=10)
    else:
        cbar.set_label(f'{quantity}(θ, λ)', fontsize=10)

    # Title
    substance = getattr(xsc.ri, 'substance', '')
    r0_um = getattr(xsc.dist, 'r0_um', None)
    if r0_um is not None:
        title = f'{substance} – Phase Function {quantity}(θ, λ) | r₀ = {r0_um:.2f} µm'
    else:
        title = f'{substance} – Phase Function {quantity}(θ, λ)'
    ax.set_title(title, fontsize=12, fontweight='bold')

    # The colorbar installs a layout engine; subplots_adjust would conflict
    # with it (emitting a UserWarning).  Only apply manual spacing when no
    # layout engine is active.
    if fig.get_layout_engine() is None:
        fig.subplots_adjust(bottom=0.12, left=0.12, right=0.92)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved → %s", save_path)
    if show:
        plt.show()

    return fig
