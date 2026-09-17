"""
mieEODG.functions
=================
Utility I/O helpers for the package and GUI.

Currently provides save/load of particle size-distribution PDFs as simple CSV
files, so a distribution previewed in the viewer can be exported, and measured
or custom PDFs can be loaded back in for computation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def savePDF(
    path: str | Path,
    r_um: np.ndarray,
    dndlnr: np.ndarray,
    *,
    substance: str | None = None,
    normalized: bool = False,
) -> None:
    """
    Save a size-distribution PDF to CSV.

    The file stores the plotted quantity ``dN/d ln r`` against radius in µm,
    with a small comment header.  Round-trips with :func:`loadPDF`.

    Parameters
    ----------
    path : str or Path
        Output ``.csv`` path.
    r_um : array-like
        Radii [µm].
    dndlnr : array-like
        ``dN/d ln r`` at each radius (same length as ``r_um``).
    substance : str or None
        Optional label written into the header comment.
    normalized : bool
        Recorded in the header to note whether the curve is peak-normalised.
    """
    r_um = np.asarray(r_um, dtype=np.float64)
    dndlnr = np.asarray(dndlnr, dtype=np.float64)
    if r_um.shape != dndlnr.shape:
        raise ValueError("r_um and dndlnr must have the same shape.")

    path = Path(path)
    header_lines = ["mieEODG size-distribution PDF"]
    if substance:
        header_lines.append(f"substance: {substance}")
    header_lines.append(f"normalized: {bool(normalized)}")
    header_lines.append("columns: radius_um, dNdlnr")
    header = "\n".join(header_lines)

    np.savetxt(
        path, np.column_stack([r_um, dndlnr]),
        delimiter=",", header=header, comments="# ",
        fmt=("%.8g", "%.8g"),
    )


def loadPDF(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a size-distribution PDF from CSV (the format written by :func:`savePDF`).

    Comment lines (``#``) and an optional ``radius_um,dNdlnr`` text header row
    are skipped.  Returns ``(r_um, dndlnr)`` sorted by ascending radius.

    Parameters
    ----------
    path : str or Path
        Input ``.csv`` path.  Must have two numeric columns: radius [µm] and
        ``dN/d ln r``.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``r_um`` and ``dndlnr``.
    """
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        # Possibly whitespace-delimited; retry without a delimiter.
        data = np.atleast_2d(np.genfromtxt(path, comments="#"))
    if data.shape[1] < 2:
        raise ValueError(f"Could not parse two numeric columns from {path}.")

    r_um = data[:, 0]
    dndlnr = data[:, 1]
    good = np.isfinite(r_um) & np.isfinite(dndlnr) & (r_um > 0)
    r_um, dndlnr = r_um[good], dndlnr[good]
    order = np.argsort(r_um)
    return r_um[order], dndlnr[order]
