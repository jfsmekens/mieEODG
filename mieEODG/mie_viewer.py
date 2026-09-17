"""
MieViewer — Interactive GUI for Mie scattering calculations.

Provides a desktop interface to explore aerosol optical properties:
- Select refractive indices from ARIA database
- Compute single-particle Mie properties (AerosolXSC)
- Generate and query lookup tables (1-D and N-D)
- Visualize phase functions and optical coefficients
- Export results and figures

Launch via:
    python -m mieEODG.mie_viewer
    mie-viewer  (console script)
"""

from __future__ import annotations

import logging
import logging.handlers
import threading
import traceback
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt

# Benign matplotlib notice emitted whenever tight_layout() is applied to an
# embedded figure — suppress it so the GUI console/stderr stays clean.
warnings.filterwarnings(
    "ignore",
    message="The figure layout has changed to tight",
    category=UserWarning,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

# Import from submodules (not the package top) so the GUI does not depend on
# mieEODG/__init__.py import order — MieViewer is itself imported by __init__,
# so a top-level `from mieEODG import …` would be a fragile circular import.
from mieEODG.refindex import ARIADatabase, RefrIndex
from mieEODG.aerosol import AerosolXSC
from mieEODG.lut import AerosolLUT
from mieEODG.size_dist import (
    LogNormalDist, GaussianDist, ModifiedGammaDist,
    MonodisperseDist, TabulatedDist,
)
from mieEODG.constants import (
    ARIA_DEFAULTS, SPECTRAL_RANGE_PRESETS, get_cmap, aria_density,
)
from mieEODG.plotting import plot_size_dist
from mieEODG.functions import savePDF, loadPDF
from mieEODG.lut_multi import (
    MultiDimLUT, LUTAxis,
    registry_substances, registry_sources, registry_axes,
)


# ============================================================================
# Logging Handler for GUI
# ============================================================================

class _GuiLogHandler(logging.Handler):
    """Log handler that writes to a Tkinter Text widget with syntax coloring."""

    _LEVEL_TAG = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, text_widget: tk.Text) -> None:
        super().__init__()
        self._text = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to text widget."""
        try:
            msg = self.format(record)
            tag = self._LEVEL_TAG.get(record.levelno, "info")
            self._text.after(0, self._append, msg, tag)
        except Exception:
            self.handleError(record)

    def _append(self, msg: str, tag: str) -> None:
        """Append message to text widget in main thread."""
        try:
            self._text.config(state=tk.NORMAL)
            self._text.insert(tk.END, msg + "\n", tag)
            self._text.see(tk.END)
            self._text.config(state=tk.DISABLED)
        except tk.TclError:
            pass


# ============================================================================
# MieViewer Main Class
# ============================================================================

class MieViewer(tk.Toplevel):
    """Interactive GUI for Mie scattering calculations."""

    def __init__(self, parent=None):
        """Initialize MieViewer window."""
        super().__init__(parent)
        try:
            from mieEODG import __version__ as _pkg_version
        except Exception:
            _pkg_version = ""
        _ver = f" v{_pkg_version}" if _pkg_version else ""
        self.title(f"MieViewer{_ver} — Mie Scattering Calculator")

        # Store parent for later cleanup
        self.parent = parent

        # Cache screen width for proportional sizing
        self.update_idletasks()
        self._sw = self.winfo_screenwidth()
        self._sh = self.winfo_screenheight()

        # Set initial window size: 85% of screen, min 1200x700
        w = max(1200, int(self._sw * 0.85))
        h = max(700, int(self._sh * 0.85))
        x = (self._sw - w) // 2
        y = (self._sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1200, 700)

        # State variables
        self.ri: RefrIndex | None = None          # interpolated onto the spectral grid
        self.ri_full: RefrIndex | None = None     # native file range (for Full-range plot)
        self.ri_path: str | None = None           # source path, for grid re-interpolation
        self.category_selected: str = "Volcanic_Ash"
        self.substance_selected: str = ""
        self.wavenumber: np.ndarray = np.arange(650, 1351, 1.0)
        self.dist: LogNormalDist | GaussianDist | ModifiedGammaDist | None = None
        self.xsc: AerosolXSC | None = None
        # Unified result for the XSC tab (works for both Single and distributions)
        self.xsc_result = None            # AerosolResult
        self.xsc_density: float | None = None
        self.xsc_kind: str = ""           # 'Single' | 'LogNormal' | ...
        self.xsc_phase: AerosolXSC | None = None   # phase-computed object
        # Persistent size-distribution preview popup (created on first Visualize)
        self._sd_win: tk.Toplevel | None = None
        self._sd_fig = None
        self._sd_canvas = None
        self.lut: AerosolLUT | None = None
        self.mdlut = None                 # MultiDimLUT (N-D), built via Build…→N-D
        self.backend: str = "python"
        self._op_message: str = ""

        # ARIA database
        self.aria_db = ARIADatabase()

        # Setup logging
        self._setup_logging()

        # Initialize matplotlib figures (before building UI)
        self.ri_fig = plt.Figure(figsize=(10, 5), dpi=100)

        # Build UI (toolbar first, then layout)
        self._build_toolbar()
        self._build_layout()

        # Initial RI load (AFTER layout so ri_canvas exists)
        self._load_default_ri()

        logger = logging.getLogger("mieEODG.gui")
        logger.info("MieViewer initialized")

    def _setup_logging(self) -> None:
        """Configure logging to GUI console and file."""
        # File handler
        log_dir = Path.home() / ".mieEODG"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "mieviewer.log"

        # Create logger. Clear any handlers left by a previous MieViewer in
        # this process so repeated launches don't stack handlers (which would
        # duplicate every log line). The 'mieEODG.gui' logger is dedicated to
        # the GUI, so clearing it has no side effects elsewhere.
        self.logger = logging.getLogger("mieEODG.gui")
        self.logger.setLevel(logging.DEBUG)
        for _h in list(self.logger.handlers):
            self.logger.removeHandler(_h)

        # File handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2 * 1024 * 1024, backupCount=3
        )
        self.file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
        )
        self.file_handler.setFormatter(file_fmt)
        self.logger.addHandler(self.file_handler)

        # GUI handler (added after log console is created)
        self.gui_handler = None
        self.gui_fmt = logging.Formatter("%(asctime)s  [%(levelname)-7s]  %(message)s",
                                         datefmt="%H:%M:%S")

    def _build_toolbar(self) -> None:
        """Build toolbar with action buttons."""
        outer = ttk.Frame(self)
        outer.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        # Row 1: compute + LUT sections
        row1 = ttk.Frame(outer)
        row1.pack(side=tk.TOP, fill=tk.X)

        # --- Compute section ---
        ttk.Button(row1, text="Compute XSC", command=self._on_compute_single).pack(
            side=tk.LEFT, padx=2)
        # Phase is an option of AerosolXSC.compute(theta_pts=...) — fold it into
        # the same pass rather than a separate (recomputing) button.
        self._phase_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="+ Phase", variable=self._phase_var).pack(
            side=tk.LEFT, padx=(0, 2))

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # --- LUT section ---
        ttk.Label(row1, text="LUT:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row1, text="Build…", command=self._on_build_lut).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Load…", command=self._on_load_lut).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Save…", command=self._on_save_lut).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # Backend selector on right side (keep self.backend in sync)
        self.backend_var = tk.StringVar(value=self.backend)
        self.backend_var.trace_add(
            "write", lambda *_a: setattr(self, "backend", self.backend_var.get()))
        ttk.Label(row1, text="Backend:").pack(side=tk.RIGHT, padx=(0, 2))
        ttk.Combobox(
            row1,
            textvariable=self.backend_var,
            values=["python", "fortran"],
            state="readonly",
            width=10,
        ).pack(side=tk.RIGHT, padx=(0, 6))

        # Row 2: status text + progress bar
        row2 = ttk.Frame(outer)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(row2, textvariable=self._status_var, foreground="gray").pack(
            side=tk.LEFT, padx=(14, 8))
        # Progress bar fills whatever horizontal space the status text leaves.
        self.progress_bar = ttk.Progressbar(row2, mode="indeterminate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    def _build_layout(self) -> None:
        """Build main window layout with dynamic sizing."""
        # Main content container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # PanedWindow with proportional sizing (left panel ~24% of screen width, frozen)
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL — fixed width (~24% of screen), doesn't resize
        left_width = int(self._sw * 0.24)
        left_frame = ttk.Frame(paned, width=left_width)
        left_frame.pack_propagate(False)  # Freeze width
        paned.add(left_frame, weight=0)   # Don't get extra space on resize

        # Build left panel controls
        self._build_left_panel_controls(left_frame)

        # RIGHT PANEL CONTAINER — expands to fill remaining space
        right_container = ttk.Frame(paned)
        paned.add(right_container, weight=1)  # Gets all extra space

        # Build right panel components.  Pack the log console FIRST (side=BOTTOM,
        # fixed height, non-expanding) so it is always reserved and visible; the
        # notebook then fills whatever space remains and resizes with the window.
        self._build_log_console(right_container)
        self._build_right_panel(right_container)

    def _build_left_panel_controls(self, parent: tk.Widget) -> None:
        """Build left control panel."""
        frame = parent

        # Scrollable container for controls.  Match the canvas background to the
        # ttk theme so the unoccupied area below the controls doesn't show up as
        # a white block (notably in dark mode).
        theme_bg = ttk.Style().lookup("TFrame", "background") or None
        canvas = tk.Canvas(frame, highlightthickness=0, bg=theme_bg)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 1. Spectral Grid
        self._build_spectral_grid(scrollable_frame)

        # 2. Refractive Index
        self._build_ri_frame(scrollable_frame)

        # 3. Size Distribution
        self._build_size_dist_frame(scrollable_frame)

        # 4. Phase Function options
        self._build_phase_options_frame(scrollable_frame)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_spectral_grid(self, parent: tk.Widget) -> None:
        """Build spectral grid control frame."""
        lf = ttk.LabelFrame(parent, text="Spectral Grid", padding=8)
        lf.pack(fill=tk.X, padx=4, pady=4)

        # WN range on a single line:  WN range: [start] – [stop] cm⁻¹
        ttk.Label(lf, text="WN range:").grid(row=0, column=0, sticky="w")
        range_frame = ttk.Frame(lf)
        range_frame.grid(row=0, column=1, sticky="w")
        self.wn_min_var = tk.StringVar(value="650")
        self.wn_max_var = tk.StringVar(value="1350")
        ttk.Entry(range_frame, textvariable=self.wn_min_var, width=7).pack(side=tk.LEFT)
        ttk.Label(range_frame, text="–").pack(side=tk.LEFT, padx=3)
        ttk.Entry(range_frame, textvariable=self.wn_max_var, width=7).pack(side=tk.LEFT)
        ttk.Label(range_frame, text="cm⁻¹").pack(side=tk.LEFT, padx=(4, 0))

        # Step on its own line
        ttk.Label(lf, text="Step:").grid(row=1, column=0, sticky="w")
        step_frame = ttk.Frame(lf)
        step_frame.grid(row=1, column=1, sticky="w")
        self.wn_step_var = tk.StringVar(value="1.0")
        ttk.Entry(step_frame, textvariable=self.wn_step_var, width=7).pack(side=tk.LEFT)
        ttk.Label(step_frame, text="cm⁻¹").pack(side=tk.LEFT, padx=(4, 0))

        # Apply button (commits the grid: parses fields → self.wavenumber,
        # re-interpolates the RI, replots).  No auto-refresh on these fields.
        btn_frame = ttk.Frame(lf)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(btn_frame, text="Apply", command=self._on_apply_spectral_grid).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Presets…", command=self._on_spectral_presets).pack(
            side=tk.LEFT, padx=2
        )

    def _build_ri_frame(self, parent: tk.Widget) -> None:
        """Build refractive index selection frame."""
        lf = ttk.LabelFrame(parent, text="Refractive Index", padding=8)
        lf.pack(fill=tk.X, padx=4, pady=4)

        # Category
        ttk.Label(lf, text="Category:").grid(row=0, column=0, sticky="w")
        self.category_var = tk.StringVar(value=self.category_selected)
        categories = sorted(self.aria_db.categories())
        self.category_combo = ttk.Combobox(
            lf, textvariable=self.category_var, values=categories, width=20, state="readonly"
        )
        self.category_combo.grid(row=0, column=1, sticky="ew")
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_changed)

        # Substance
        ttk.Label(lf, text="Substance:").grid(row=1, column=0, sticky="w")
        self.substance_var = tk.StringVar()
        self.substance_combo = ttk.Combobox(
            lf, textvariable=self.substance_var, width=20, state="readonly"
        )
        self.substance_combo.grid(row=1, column=1, sticky="ew")
        self.substance_combo.bind("<<ComboboxSelected>>", self._on_substance_changed)

        # Reference tree — drill-down navigator of the (heterogeneous) folder
        # structure under the substance (reference / composition / temperature
        # depending on the substance).  Leaves are the actual .ri files.
        # Spans the full frame width.
        tree_frame = ttk.Frame(lf)
        tree_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.ri_tree = ttk.Treeview(tree_frame, show="tree", height=7, selectmode="browse")
        ri_tree_sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.ri_tree.yview)
        self.ri_tree.configure(yscrollcommand=ri_tree_sb.set)
        ri_tree_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ri_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ri_tree.bind("<<TreeviewSelect>>", self._on_ri_tree_select)
        self._ri_tree_leaves: dict[str, str] = {}   # leaf iid → relative .ri path

        # Temperature — read-only display of the loaded file's measured T
        # (each .ri is at a fixed T, chosen via the tree; not user-editable).
        ttk.Label(lf, text="Temperature:").grid(row=3, column=0, sticky="w")
        self.temp_label = ttk.Label(lf, text="—", foreground="gray")
        self.temp_label.grid(row=3, column=1, sticky="w")

        # Density — editable: auto-filled when resolvable, manual entry otherwise.
        ttk.Label(lf, text="Density:").grid(row=4, column=0, sticky="w")
        dens_frame = ttk.Frame(lf)
        dens_frame.grid(row=4, column=1, sticky="w")
        self.density_var = tk.StringVar()
        self.density_entry = ttk.Entry(dens_frame, textvariable=self.density_var, width=8)
        self.density_entry.pack(side=tk.LEFT)
        ttk.Label(dens_frame, text="g/cm³").pack(side=tk.LEFT, padx=(4, 0))
        self.density_status = ttk.Label(dens_frame, text="", foreground="gray")
        self.density_status.pack(side=tk.LEFT, padx=(6, 0))

        # Buttons (left-aligned, natural width)
        btn_frame = ttk.Frame(lf)
        btn_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(btn_frame, text="Load Custom RI…", command=self._on_load_custom_ri).pack(
            side=tk.LEFT, padx=2
        )

        lf.columnconfigure(1, weight=1)
        self._update_substance_list()

    def _build_size_dist_frame(self, parent: tk.Widget) -> None:
        """Build particle size control frame."""
        lf = ttk.LabelFrame(parent, text="Particle Size", padding=8)
        lf.pack(fill=tk.X, padx=4, pady=4)

        # Type selector (Single, an analytic distribution, or a loaded Custom PDF)
        ttk.Label(lf, text="Type:").grid(row=0, column=0, sticky="w")
        self.dist_type_var = tk.StringVar(value="LogNormal")
        particle_types = ["Single", "LogNormal", "Gaussian", "ModifiedGamma", "Custom"]
        type_combo = ttk.Combobox(
            lf, textvariable=self.dist_type_var, values=particle_types, width=20, state="readonly"
        )
        type_combo.grid(row=0, column=1, sticky="ew")
        type_combo.bind("<<ComboboxSelected>>", self._on_particle_type_changed)
        self._loaded_pdf: tuple[np.ndarray, np.ndarray] | None = None  # (r_um, dndlnr)

        # r0 (always shown; disabled for Custom)
        self.r0_label = ttk.Label(lf, text="r₀ (µm):")
        self.r0_label.grid(row=1, column=0, sticky="w")
        self.r0_var = tk.StringVar(value="1.0")
        self.r0_entry = ttk.Entry(lf, textvariable=self.r0_var, width=12)
        self.r0_entry.grid(row=1, column=1, sticky="w")

        # sigma / s (hidden for Single)
        self.sigma_label = ttk.Label(lf, text="σ / s:")
        self.sigma_label.grid(row=2, column=0, sticky="w")
        self.sigma_var = tk.StringVar(value="1.5")
        self.sigma_entry = ttk.Entry(lf, textvariable=self.sigma_var, width=12)
        self.sigma_entry.grid(row=2, column=1, sticky="w")

        # N_d — number density [cm⁻³] (identical particles for Single)
        self.ntotal_label = ttk.Label(lf, text="N_d (cm⁻³):")
        self.ntotal_label.grid(row=3, column=0, sticky="w")
        self.ntotal_var = tk.StringVar(value="1.0E6")
        self.ntotal_entry = ttk.Entry(lf, textvariable=self.ntotal_var, width=12)
        self.ntotal_entry.grid(row=3, column=1, sticky="w")

        # Live-update the preview popup (if open) as parameters change
        for ent in (self.r0_entry, self.sigma_entry, self.ntotal_entry):
            ent.bind("<KeyRelease>", self._refresh_sizedist_popup)

        # Buttons (left-aligned, natural width)
        btns = ttk.Frame(lf)
        btns.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Visualize", command=self._on_visualize_particle).pack(
            side=tk.LEFT, padx=(0, 2))
        ttk.Button(btns, text="Load PDF…", command=self._on_load_pdf).pack(
            side=tk.LEFT, padx=2)

        lf.columnconfigure(1, weight=1)

    def _build_phase_options_frame(self, parent: tk.Widget) -> None:
        """Build phase-function options (used when '+ Phase' is ticked)."""
        lf = ttk.LabelFrame(parent, text="Phase Function Options", padding=8)
        lf.pack(fill=tk.X, padx=4, pady=4)

        # Number of scattering angles (θ grid, 0–180°)
        ttk.Label(lf, text="N angles (θ):").grid(row=0, column=0, sticky="w")
        self.phase_nangles_var = tk.StringVar(value="181")
        ttk.Entry(lf, textvariable=self.phase_nangles_var, width=10).grid(
            row=0, column=1, sticky="w")
        ttk.Label(lf, text="0–180°").grid(row=0, column=2, sticky="w", padx=(4, 0))

        # Legendre expansion order
        ttk.Label(lf, text="N Legendre:").grid(row=1, column=0, sticky="w")
        self.phase_nlegendre_var = tk.StringVar(value="32")
        ttk.Entry(lf, textvariable=self.phase_nlegendre_var, width=10).grid(
            row=1, column=1, sticky="w")

        ttk.Label(
            lf, text="Applied when '+ Phase' is ticked.\nPhase uses the Python backend.",
            foreground="gray", font=("TkDefaultFont", 8), justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        lf.columnconfigure(1, weight=1)

    def _build_right_panel(self, parent: tk.Widget) -> None:
        """Build right panel with notebook tabs into parent container."""
        # Notebook (4 tabs: RI, XSC, Phase Function, LUT)
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Tab 1: RI
        self.ri_tab = self._build_ri_tab()
        self.notebook.add(self.ri_tab, text="RI")

        # Tab 2: XSC (combined Single + Size Distribution results)
        self.xsc_tab = self._build_xsc_tab()
        self.notebook.add(self.xsc_tab, text="XSC")

        # Tab 3: Phase Function
        self.phase_tab = self._build_phase_tab()
        self.notebook.add(self.phase_tab, text="Phase Function")

        # Tab 4: LUT (placeholder)
        self.lut_tab = self._build_lut_tab()
        self.notebook.add(self.lut_tab, text="LUT")

    def _build_ri_tab(self) -> tk.Frame:
        """Build RI tab with matplotlib figure."""
        frame = ttk.Frame(self.notebook)

        # Display-mode control row (above the canvas)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Range:").pack(side=tk.LEFT, padx=(0, 6))
        self.ri_plot_mode_var = tk.StringVar(value="grid")
        ttk.Radiobutton(
            ctrl, text="Spectral Grid", variable=self.ri_plot_mode_var,
            value="grid", command=self._plot_ri,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            ctrl, text="Full range", variable=self.ri_plot_mode_var,
            value="full", command=self._plot_ri,
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Matplotlib canvas for RI figure (figure already initialized in __init__)
        self.ri_canvas = FigureCanvasTkAgg(self.ri_fig, master=frame)
        self.ri_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        return frame

    def _build_log_console(self, parent: tk.Widget) -> None:
        """Build the always-visible log console at the bottom of the right panel.

        Packed with a fixed row height and ``expand=False`` so it is always
        reserved; the notebook above it absorbs window resizing instead.
        """
        log_frame = ttk.LabelFrame(parent, text="Log Console", padding=4)
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=(2, 0), expand=False)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget (read-only) — fixed height (rows), no vertical expansion
        self.log_text = tk.Text(
            log_frame,
            height=9,
            state=tk.DISABLED,
            font=("TkFixedFont", 11),
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD,
        )
        # expand=True so the text fills the full panel width; the fixed row
        # height (above) still bounds the vertical size.
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Configure tags
        self.log_text.tag_config("debug", foreground="#999999")
        self.log_text.tag_config("info", foreground="#555555")
        self.log_text.tag_config("warning", foreground="#bb7700")
        self.log_text.tag_config("error", foreground="#cc2200")

        # Attach GUI handler
        self.gui_handler = _GuiLogHandler(self.log_text)
        self.gui_handler.setLevel(logging.DEBUG)
        self.gui_handler.setFormatter(self.gui_fmt)
        self.logger.addHandler(self.gui_handler)

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def _on_apply_spectral_grid(self) -> None:
        """Apply spectral grid changes."""
        try:
            wn_min = float(self.wn_min_var.get())
            wn_max = float(self.wn_max_var.get())
            wn_step = float(self.wn_step_var.get())

            if wn_min <= 0 or wn_max <= 0 or wn_step <= 0:
                raise ValueError("All values must be positive")
            if wn_min >= wn_max:
                raise ValueError("WN Min must be < WN Max")

            self.wavenumber = np.arange(wn_min, wn_max + wn_step, wn_step)
            self.logger.info(f"Spectral grid: {wn_min}–{wn_max} cm⁻¹ @ {wn_step} cm⁻¹ ({len(self.wavenumber)} points)")

            # Re-interpolate the loaded RI onto the new grid and refresh the plot
            # (the grid window also drives the Full-range mode's overlay span).
            if self.ri_path is not None:
                self.ri = self.aria_db.read(
                    self.ri_path, wavenumber=self.wavenumber, out_of_range="clip"
                )
                self._plot_ri()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Spectral grid error: {e}")
            self.logger.error(f"Spectral grid validation failed: {e}")

    def _on_spectral_presets(self) -> None:
        """Popup to pick a named spectral-grid preset and apply it."""
        dlg = tk.Toplevel(self)
        dlg.title("Spectral Grid Presets")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Choose a spectral range:").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 6))

        def _apply(name: str, spec: dict) -> None:
            self.wn_min_var.set(f"{spec['wn_min']:g}")
            self.wn_max_var.set(f"{spec['wn_max']:g}")
            self.wn_step_var.set(f"{spec['wn_step']:g}")
            dlg.destroy()
            self.logger.info(f"Applied spectral preset '{name}'")
            self._on_apply_spectral_grid()

        for i, (name, spec) in enumerate(SPECTRAL_RANGE_PRESETS.items(), start=1):
            ttk.Button(
                dlg, text=name, width=12,
                command=lambda n=name, s=spec: _apply(n, s),
            ).grid(row=i, column=0, sticky="w", padx=(10, 6), pady=2)
            ttk.Label(
                dlg,
                text=(f"{spec['wn_min']:g}–{spec['wn_max']:g} cm⁻¹ "
                      f"@ {spec['wn_step']:g}   ·   {spec['description']}"),
                foreground="gray",
            ).grid(row=i, column=1, sticky="w", padx=(0, 10), pady=2)

        ttk.Button(dlg, text="Cancel", command=dlg.destroy).grid(
            row=len(SPECTRAL_RANGE_PRESETS) + 1, column=0, columnspan=2, pady=(8, 10))

    def _on_particle_type_changed(self, event=None) -> None:
        """Enable/disable distribution parameters based on particle type.

        Single        → σ/s disabled (one size; N_d = identical-particle count).
        Custom        → r₀ and σ/s disabled (the shape comes from the loaded
                        PDF); N_d rescales it.
        Distributions → all enabled.
        Disabled fields are greyed rather than hidden so the panel height
        stays constant.  N_d is always active.
        """
        ptype = self.dist_type_var.get()
        is_single = ptype == "Single"
        is_custom = ptype == "Custom"

        def _set(entry, lbl, enabled):
            entry.config(state="normal" if enabled else "disabled")
            lbl.config(foreground="black" if enabled else "gray")

        _set(self.r0_entry, self.r0_label, not is_custom)
        _set(self.sigma_entry, self.sigma_label, not (is_single or is_custom))
        # N_d remains enabled for all particle types
        self.ntotal_entry.config(state="normal")
        self.ntotal_label.config(foreground="black")
        # Keep the preview popup (if open) in sync
        self._refresh_sizedist_popup()

    def _on_load_pdf(self) -> None:
        """Load a measured/custom size-distribution PDF (CSV) → 'Custom' type."""
        path = filedialog.askopenfilename(
            title="Load size-distribution PDF",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            r_um, dndlnr = loadPDF(path)
            if r_um.size < 2:
                raise ValueError("PDF must have at least 2 points")
            self._loaded_pdf = (r_um, dndlnr)
            self.dist_type_var.set("Custom")
            self._on_particle_type_changed()
            self.logger.info(
                f"Loaded custom PDF from {Path(path).name} ({r_um.size} points, "
                f"{r_um.min():.3g}–{r_um.max():.3g} µm)")
            self._refresh_sizedist_popup()
        except Exception as e:
            messagebox.showerror("Load PDF Error", str(e))
            self.logger.error(f"Failed to load PDF: {e}")

    def _on_category_changed(self, event=None) -> None:
        """Update substance list when category changes."""
        self._update_substance_list()

    def _update_substance_list(self) -> None:
        """Populate substance dropdown based on category."""
        category = self.category_var.get()
        if category:
            substances = sorted(self.aria_db.substances(category))
            self.substance_combo.config(values=substances)
            if substances:
                self.substance_combo.current(0)
                self._on_substance_changed()

    def _on_substance_changed(self, event=None) -> None:
        """Rebuild the reference tree for the chosen substance and load the first file."""
        self._populate_ri_tree(select_path=None)

    def _populate_ri_tree(self, select_path: str | None = None) -> None:
        """Build the drill-down folder tree under category/substance.

        Mirrors the actual (heterogeneous) ARIA folder layout: intermediate
        nodes are folders (reference / composition / temperature, per substance)
        and leaves are the ``.ri`` files.  Selecting a leaf loads it.

        Parameters
        ----------
        select_path : str or None
            Relative ``.ri`` path to pre-select; if None, the first leaf is used.
        """
        from pathlib import PurePosixPath

        self.ri_tree.delete(*self.ri_tree.get_children())
        self._ri_tree_leaves = {}

        category = self.category_var.get()
        substance = self.substance_var.get()
        if not (category and substance):
            return
        try:
            relpaths = self.aria_db.references(category, substance)
        except Exception as e:
            self.logger.error(f"Failed to list references: {e}")
            return

        prefix_len = len(PurePosixPath(category, substance).parts)
        node_by_key: dict[tuple, str] = {}   # (parent_iid, name) → iid
        first_leaf: str | None = None
        target_leaf: str | None = None

        for rel in relpaths:
            parts = PurePosixPath(rel).parts[prefix_len:]   # below substance
            parent = ""
            # Intermediate folders
            for name in parts[:-1]:
                key = (parent, name)
                iid = node_by_key.get(key)
                if iid is None:
                    iid = self.ri_tree.insert(parent, tk.END, text=name, open=True)
                    node_by_key[key] = iid
                parent = iid
            # Leaf file (.ri)
            leaf_name = parts[-1] if parts else PurePosixPath(rel).name
            leaf_iid = self.ri_tree.insert(parent, tk.END, text=leaf_name)
            self._ri_tree_leaves[leaf_iid] = rel
            if first_leaf is None:
                first_leaf = leaf_iid
            if select_path is not None and rel == select_path:
                target_leaf = leaf_iid

        chosen = target_leaf or first_leaf
        if chosen is not None:
            self.ri_tree.see(chosen)
            self.ri_tree.selection_set(chosen)
            # Load directly — programmatic selection_set does not reliably fire
            # <<TreeviewSelect>>.  The binding still handles user clicks; the
            # reload guard below prevents a double-load if it does fire.
            self._load_ri_from_tree(chosen)

    def _on_ri_tree_select(self, event=None) -> None:
        """Load the .ri file when a leaf node is selected (folders are ignored)."""
        sel = self.ri_tree.selection()
        if sel:
            self._load_ri_from_tree(sel[0])

    def _load_ri_from_tree(self, iid: str) -> None:
        """Load the .ri file for a tree leaf; no-op for folders or re-selection."""
        relpath = self._ri_tree_leaves.get(iid)
        if relpath is None or relpath == self.ri_path:
            return   # folder node, or already loaded
        try:
            self._load_ri(relpath)
        except Exception as e:
            self.logger.error(f"Failed to load RI: {e}")
            messagebox.showerror("RI Load Error", str(e))

    def _on_load_custom_ri(self) -> None:
        """Load custom RI file."""
        path = filedialog.askopenfilename(
            title="Load Custom RI File",
            filetypes=[("RI files", "*.ri"), ("All files", "*.*")],
        )
        if path:
            try:
                self._load_ri(path)
            except Exception as e:
                messagebox.showerror("RI Load Error", str(e))
                self.logger.error(f"Failed to load custom RI: {e}")

    def _load_default_ri(self) -> None:
        """Load the default RI on startup, keeping the comboboxes in sync.

        The default is driven THROUGH the category/substance selectors so the
        displayed selection always matches ``self.ri`` (otherwise the panel
        would show the first-alphabetical substance while computations use a
        different default).
        """
        try:
            default_path = ARIA_DEFAULTS["volcanic_ash"]
            parts = Path(default_path).parts
            category, substance = parts[0], parts[1]   # e.g. 'Volcanic_Ash', 'Etna'

            self.category_var.set(category)
            substances = sorted(self.aria_db.substances(category))
            self.substance_combo.config(values=substances)
            self.substance_var.set(substance if substance in substances
                                   else (substances[0] if substances else ""))
            # Build the reference tree and pre-select the exact default file.
            self._populate_ri_tree(select_path=default_path)
        except Exception as e:
            self.logger.warning(f"Failed to load default RI: {e}")

    def _load_ri(self, path: str) -> None:
        """Load RI and update displays."""
        self.ri_path = path
        # Native (full file range) — used by the Full-range plot mode.
        self.ri_full = self.aria_db.read(path)
        # Interpolated onto the active spectral grid — used by compute + grid plot.
        self.ri = self.aria_db.read(path, wavenumber=self.wavenumber, out_of_range="clip")
        self.logger.info(f"Loaded RI: {self.ri.substance} [{self.ri.category}]")

        # Show the file's measured temperature (from the header), if any.
        self.temp_label.config(
            text=(f"{self.ri.T_K:g} K" if self.ri.T_K is not None else "—"))

        # Resolve density using the same hierarchy AerosolXSC uses, so the
        # displayed value matches what the computation will actually use.
        rho = aria_density(
            category      = self.ri.category,
            composition   = self.ri.composition,
            T_K           = self.ri.T_K,
            substance_key = self.ri.substance_key,
        )
        if rho is not None:
            self.density_var.set(f"{rho:.4f}")
            self.density_status.config(text="(auto)", foreground="gray")
        else:
            # No automatic density (e.g. salts/custom .ri) — the user must enter
            # one; compute/LUT will use whatever is in the entry.
            self.logger.warning(
                f"Density not auto-resolvable for category='{self.ri.category}', "
                f"substance_key='{self.ri.substance_key}'. Enter one manually."
            )
            self.density_var.set("")
            self.density_status.config(text="enter manually", foreground="orange")

        # Update RI plot
        self._plot_ri()

    def _plot_ri(self) -> None:
        """Plot refractive index on RI tab (Spectral Grid or Full range mode)."""
        if self.ri is None or not hasattr(self, 'ri_canvas'):
            return

        from mieEODG.plotting import _add_top_axis

        mode = self.ri_plot_mode_var.get() if hasattr(self, 'ri_plot_mode_var') else "grid"

        if mode == "full" and self.ri_full is not None:
            wn, n, k = self.ri_full.wavenumber, self.ri_full.n, self.ri_full.k
        else:
            wn, n, k = self.wavenumber, self.ri.n, self.ri.k

        self.ri_fig.clear()

        # Top: real part
        ax1 = self.ri_fig.add_subplot(2, 1, 1)
        ax1.plot(wn, n, color="steelblue", lw=1.5, label="n")
        ax1.set_ylabel("Real Part (n)")
        ax1.set_xlim(wn.max(), wn.min())  # descending wavenumber
        ax1.grid(True, alpha=0.3)
        _add_top_axis(ax1, units='um')  # wavelength top axis only on top panel
        ax1.tick_params(labelbottom=False)

        # Bottom: imaginary part (shares x-axis with top)
        ax2 = self.ri_fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(wn, k, color="coral", lw=1.5, label="k")
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("Imag. Part (k)")
        ax2.grid(True, alpha=0.3)

        # In Full-range mode, overlay the selected spectral grid window.
        if mode == "full":
            g_lo, g_hi = float(self.wavenumber.min()), float(self.wavenumber.max())
            for ax in (ax1, ax2):
                ax.axvspan(g_lo, g_hi, color="0.8", alpha=0.5, zorder=0,
                           label="Spectral grid")
            ax1.legend(loc="best", fontsize=8)

        self.ri_fig.tight_layout()
        self.ri_canvas.draw()

        self.logger.debug(f"Plotted RI: n range {self.ri.n.min():.3f}–{self.ri.n.max():.3f}")

    # ------------------------------------------------------------------
    # XSC tab
    # ------------------------------------------------------------------

    # Quantity families: key → (radio label, [(result-attr, panel-label)], unit,
    #                            scale-from-cm-internal, peak-format)
    _XSC_FAMILIES = {
        "Q": ("Q (efficiency)",
              [("Q_ext", "Extinction"), ("Q_sca", "Scattering"), ("Q_abs", "Absorption")],
              "", 1.0),
        "C": ("C (cross-section)",
              [("C_ext", "Extinction"), ("C_sca", "Scattering"), ("C_abs", "Absorption")],
              "µm²", 1.0e8),     # cm² → µm²
        "B": ("B (volume coeff.)",
              [("B_ext", "Extinction"), ("B_sca", "Scattering"), ("B_abs", "Absorption")],
              "cm⁻¹", 1.0),
        "M": ("M (mass coeff.)",
              [("MEC", "Extinction"), ("MSC", "Scattering"), ("MAC", "Absorption")],
              "cm²/g", 1.0),
    }
    # Per-panel (ext / sca / abs) differentiation: same species hue, varied
    # linestyle + progressive lightening.  (linestyle, lighten-fraction)
    _XSC_PANEL_STYLE = (("-", 0.0), ("--", 0.30), (":", 0.55))

    @staticmethod
    def _lighten(color, amount: float):
        """Blend *color* toward white by *amount* (0 = unchanged, 1 = white)."""
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def _build_xsc_tab(self) -> tk.Frame:
        """Build XSC tab: Q/C/B/M spectra with a quantity toggle + info box."""
        frame = ttk.Frame(self.notebook)

        # Quantity-family toggle row (top)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Quantity:").pack(side=tk.LEFT, padx=(0, 6))
        self.xsc_family_var = tk.StringVar(value="M")
        for key, (label, *_rest) in self._XSC_FAMILIES.items():
            ttk.Radiobutton(
                ctrl, text=label, variable=self.xsc_family_var,
                value=key, command=self._plot_xsc,
            ).pack(side=tk.LEFT, padx=(0, 4))

        # Info box at the bottom
        info_lf = ttk.LabelFrame(frame, text="Result Info", padding=6)
        info_lf.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))
        self.xsc_info_label = ttk.Label(
            info_lf, text="No result — click Compute XSC.",
            foreground="gray", justify=tk.LEFT,
        )
        self.xsc_info_label.pack(anchor=tk.W)

        # Figure: 3 stacked panels (ext/sca/abs), shared x-axis
        self.xsc_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.xsc_canvas = FigureCanvasTkAgg(self.xsc_fig, master=frame)
        self.xsc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        return frame

    def _plot_xsc(self) -> None:
        """Plot the selected quantity family (Q/C/B/M) for the current result."""
        if self.xsc_result is None or not hasattr(self, "xsc_canvas"):
            return

        from mieEODG.plotting import _add_top_axis
        from mieEODG.constants import get_color

        family = self.xsc_family_var.get()
        label, panels, unit, scale = self._XSC_FAMILIES[family]
        wn = self.xsc_result.wavenumber
        unit_sfx = f"\n({unit})" if unit else ""
        base_color = get_color(self.ri.category or "")

        self.xsc_fig.clear()
        axes = []
        for i, (attr, panel_label) in enumerate(panels):
            data = getattr(self.xsc_result, attr) * scale
            ls, lighten = self._XSC_PANEL_STYLE[i]
            color = self._lighten(base_color, lighten)
            ax = self.xsc_fig.add_subplot(3, 1, i + 1, sharex=axes[0] if axes else None)
            ax.plot(wn, data, color=color, lw=1.6, linestyle=ls)
            ax.set_ylabel(f"{panel_label}{unit_sfx}")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_xlim(wn.max(), wn.min())  # descending wavenumber
                _add_top_axis(ax, units="um")
            if i < len(panels) - 1:
                ax.tick_params(labelbottom=False)
            axes.append(ax)
        axes[-1].set_xlabel("Wavenumber (cm⁻¹)")

        self.xsc_fig.tight_layout()
        self.xsc_canvas.draw()

        # Info box
        r = self.xsc_result
        r_eff_um = 0.75 * (r.V_avg / r.G_avg) * 1e4  # cm → µm
        rho_str = f"{self.xsc_density:.4f} g/cm³" if self.xsc_density else "n/a"
        ext_attr = panels[0][0]
        peak = float(getattr(r, ext_attr).max()) * scale
        info = (
            f"{self.ri.substance}  [{self.ri.category}]   ρ = {rho_str}\n"
            f"{self.xsc_kind}   r_eff = {r_eff_um:.3f} µm   "
            f"N_d = {r.N_avg:.3g} cm⁻³   backend = {self.backend}\n"
            f"{label} peak (ext) = {peak:.3g} {unit}".rstrip()
        )
        self.xsc_info_label.config(text=info, foreground="black")

    # ------------------------------------------------------------------
    # Phase Function tab
    # ------------------------------------------------------------------

    def _build_phase_tab(self) -> tk.Frame:
        """Build the Phase Function tab: mode/quantity controls + figure."""
        frame = ttk.Frame(self.notebook)

        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))

        ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
        self.phase_mode_var = tk.StringVar(value="curves")
        ttk.Radiobutton(ctrl, text="Curves", variable=self.phase_mode_var,
                        value="curves", command=self._plot_phase).pack(side=tk.LEFT)
        ttk.Radiobutton(ctrl, text="Heatmap", variable=self.phase_mode_var,
                        value="heatmap", command=self._plot_phase).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(ctrl, text="Quantity:").pack(side=tk.LEFT)
        self.phase_quantity_var = tk.StringVar(value="F11")
        qcombo = ttk.Combobox(ctrl, textvariable=self.phase_quantity_var, width=6,
                              state="readonly", values=["F11", "F33", "F12", "F34"])
        qcombo.pack(side=tk.LEFT, padx=(2, 0))
        qcombo.bind("<<ComboboxSelected>>", lambda _e: self._plot_phase())

        self.phase_fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.phase_canvas = FigureCanvasTkAgg(self.phase_fig, master=frame)
        self.phase_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Placeholder text until phase is computed
        ax = self.phase_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Tick '+ Phase' next to Compute XSC to include "
                "the phase matrix.",
                ha="center", va="center", color="gray", transform=ax.transAxes)
        ax.set_axis_off()

        return frame

    def _plot_phase(self) -> None:
        """Render the phase function (curves or heatmap) for the computed phase."""
        if not hasattr(self, "phase_canvas"):
            return

        # No phase computed → show the placeholder prompt.
        if self.xsc_phase is None:
            self.phase_fig.clear()
            ax = self.phase_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Tick '+ Phase' next to Compute XSC to include "
                    "the phase matrix.", ha="center", va="center", color="gray",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.phase_canvas.draw()
            return

        from mieEODG.plotting import plot_phase_function, plot_phase_function_heatmap

        quantity = self.phase_quantity_var.get()
        mode = self.phase_mode_var.get()
        self.phase_fig.clear()
        if mode == "heatmap":
            # heatmap uses fig.gca() — an empty figure is fine.
            plot_phase_function_heatmap(
                self.xsc_phase, quantity=quantity, log_scale=True,
                fig=self.phase_fig, show=False,
            )
        else:
            # curves expects the figure to already have 2 axes (left/right).
            self.phase_fig.subplots(1, 2)
            plot_phase_function(
                self.xsc_phase, quantity=quantity,
                fig=self.phase_fig, show=False,
            )
        self.phase_canvas.draw()

    # ------------------------------------------------------------------
    # LUT tab
    # ------------------------------------------------------------------

    def _build_lut_tab(self) -> tk.Frame:
        """Build the LUT tab — a unified, slice-only viewer for 1-D and N-D LUTs.

        The single display mode is ``plot_lut_slice`` (a family of spectra
        coloured by the swept axis).  For a 1-D ``AerosolLUT`` the swept axis is
        always r₀; for an N-D ``MultiDimLUT`` any axis may be swept with the
        remainder held fixed.  A point query (``forward``) overlays a single
        interpolated spectrum.  Controls are laid out on two rows so they fit
        regardless of LUT dimensionality.
        """
        frame = ttk.Frame(self.notebook)

        # --- Row 1: quantity + swept axis + fixed-value entries ---
        row1 = ttk.Frame(frame)
        row1.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(row1, text="Quantity:").pack(side=tk.LEFT)
        self.lut_quantity_var = tk.StringVar(value="MEC")
        self.lut_quantity_combo = ttk.Combobox(
            row1, textvariable=self.lut_quantity_var, width=7, state="readonly")
        self.lut_quantity_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.lut_quantity_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._plot_lut_slice())
        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(row1, text="Sweep:").pack(side=tk.LEFT)
        self.lut_sweep_var = tk.StringVar(value="r₀")
        self.lut_sweep_combo = ttk.Combobox(
            row1, textvariable=self.lut_sweep_var, width=14, state="readonly")
        self.lut_sweep_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.lut_sweep_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._rebuild_fixed_entries())
        ttk.Label(row1, text="Fixed:").pack(side=tk.LEFT)
        self._lut_fixed_frame = ttk.Frame(row1)
        self._lut_fixed_frame.pack(side=tk.LEFT, padx=(2, 0))
        self._lut_fixed_vars: dict[str, tk.StringVar] = {}

        # --- Row 2: point query (one entry per axis) ---
        row2 = ttk.Frame(frame)
        row2.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(row2, text="Query at:").pack(side=tk.LEFT)
        self._lut_query_frame = ttk.Frame(row2)
        self._lut_query_frame.pack(side=tk.LEFT, padx=(2, 0))
        self._lut_query_vars: dict[str, tk.StringVar] = {}
        ttk.Button(row2, text="Query", command=self._on_lut_query).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Button(row2, text="Clear", command=self._on_lut_query_clear).pack(
            side=tk.LEFT, padx=(2, 0))

        # Info line
        self.lut_info_label = ttk.Label(frame, text="", foreground="gray", justify=tk.LEFT)
        self.lut_info_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))

        self.lut_fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.lut_canvas = FigureCanvasTkAgg(self.lut_fig, master=frame)
        self.lut_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Active-display state.
        self._active_lut_kind = "1d"
        self._active_lut_obj = None               # self.lut or self.mdlut
        self._lut_axinfo: list[dict] = []         # per-axis metadata for the active LUT
        self._lut_disp2name: dict[str, str] = {}  # display label → forward kwarg name
        self._lut_query_point: dict[str, float] | None = None
        return frame

    # ------------------------------------------------------------------
    # Unified slice display (1-D AerosolLUT or N-D MultiDimLUT)
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_default(info: dict) -> float:
        """Representative value for an axis: geometric mean (log) or median."""
        vals = np.asarray(info["values"], dtype=float)
        if info.get("log"):
            return float(np.exp(np.mean(np.log(vals))))
        return float(np.median(vals))

    def _activate_lut_display(self, kind: str) -> None:
        """Configure the LUT tab controls for the active 1-D or N-D LUT and draw.

        Parameters
        ----------
        kind : str
            ``'1d'`` for the current ``self.lut`` (AerosolLUT) or ``'nd'`` for
            ``self.mdlut`` (MultiDimLUT).
        """
        self._active_lut_kind = kind
        if kind == "1d":
            lut = self.lut
            if lut is None:
                return
            self._active_lut_obj = lut
            self._lut_axinfo = [{
                "name": "r0", "disp": "r₀", "units": "µm",
                "values": np.asarray(lut.r0_um, dtype=float), "log": True,
            }]
            quantities = ["MEC", "MSC", "MAC"]
        else:
            lut = self.mdlut
            if lut is None:
                return
            self._active_lut_obj = lut
            self._lut_axinfo = [{
                "name": a.name,
                "disp": "r₀" if a.name == "r0" else a.name,
                "units": a.units,
                "values": np.asarray(a.values, dtype=float),
                "log": bool(a.log_spacing),
            } for a in lut.axes]
            quantities = list(lut.quantities)

        self._lut_disp2name = {ax["disp"]: ax["name"] for ax in self._lut_axinfo}

        # Quantity selector.
        self.lut_quantity_combo.config(values=quantities)
        if self.lut_quantity_var.get() not in quantities:
            self.lut_quantity_var.set(quantities[0])

        # Swept-axis selector (locked for a 1-D LUT — r₀ is the only option).
        disp_axes = [ax["disp"] for ax in self._lut_axinfo]
        self.lut_sweep_combo.config(
            values=disp_axes,
            state="disabled" if len(disp_axes) == 1 else "readonly")
        if self.lut_sweep_var.get() not in disp_axes:
            self.lut_sweep_var.set("r₀" if "r₀" in disp_axes else disp_axes[0])

        # Query entries — one per axis (the full point passed to forward()).
        for w in self._lut_query_frame.winfo_children():
            w.destroy()
        self._lut_query_vars = {}
        for i, ax in enumerate(self._lut_axinfo):
            ttk.Label(self._lut_query_frame,
                      text=f"{ax['disp']} ({ax['units']}):").grid(
                          row=0, column=2 * i, sticky="e")
            var = tk.StringVar(value=f"{self._axis_default(ax):.4g}")
            self._lut_query_vars[ax["name"]] = var
            ttk.Entry(self._lut_query_frame, textvariable=var, width=7).grid(
                row=0, column=2 * i + 1, sticky="w", padx=(2, 8))

        self._lut_query_point = None
        self.lut_info_label.config(text="", foreground="gray")
        self._rebuild_fixed_entries()   # populates fixed entries and draws

    def _rebuild_fixed_entries(self) -> None:
        """Rebuild the fixed-value entries (every axis except the swept one) and draw."""
        sweep = self._lut_disp2name.get(self.lut_sweep_var.get(), "r0")
        for w in self._lut_fixed_frame.winfo_children():
            w.destroy()
        self._lut_fixed_vars = {}
        fixed_axes = [ax for ax in self._lut_axinfo if ax["name"] != sweep]
        if not fixed_axes:
            ttk.Label(self._lut_fixed_frame, text="(none)",
                      foreground="gray").grid(row=0, column=0)
        else:
            for i, ax in enumerate(fixed_axes):
                ttk.Label(self._lut_fixed_frame, text=f"{ax['disp']}=").grid(
                    row=0, column=2 * i, sticky="e")
                var = tk.StringVar(value=f"{self._axis_default(ax):.4g}")
                self._lut_fixed_vars[ax["name"]] = var
                ent = ttk.Entry(self._lut_fixed_frame, textvariable=var, width=7)
                ent.grid(row=0, column=2 * i + 1, sticky="w", padx=(2, 6))
                ent.bind("<Return>", lambda _e: self._plot_lut_slice())
        self._plot_lut_slice()

    def _lut_forward(self, point: dict[str, float]) -> dict:
        """Interpolate the active LUT at a full axis point → {quantity: spectrum}."""
        if self._active_lut_kind == "1d":
            return self.lut.forward(point["r0"])
        return self.mdlut.forward(**point)

    def _plot_lut_slice(self) -> None:
        """Draw the swept-axis slice for the active LUT, with optional query overlay."""
        if not hasattr(self, "lut_canvas"):
            return
        from mieEODG.plotting import plot_lut_slice

        active = self._active_lut_obj
        if active is None:
            self.lut_fig.clear()
            ax = self.lut_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Build or load a LUT to view the slice sweep.",
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_axis_off()
            self.lut_canvas.draw()
            return

        quantity = self.lut_quantity_var.get()
        sweep = self._lut_disp2name.get(self.lut_sweep_var.get(), "r0")
        fixed = {}
        for ax in self._lut_axinfo:
            if ax["name"] == sweep:
                continue
            try:
                fixed[ax["name"]] = float(self._lut_fixed_vars[ax["name"]].get())
            except (ValueError, KeyError):
                pass

        self.lut_fig.clear()
        try:
            self.lut_fig.add_subplot(111)
            if self._active_lut_kind == "1d":
                plot_lut_slice(self.lut, sweep_axis="r0", quantity=quantity,
                               fig=self.lut_fig, show=False)
            else:
                plot_lut_slice(self.mdlut, sweep_axis=sweep, fixed_axes=fixed,
                               quantity=quantity, fig=self.lut_fig, show=False)

            # Overlay the queried spectrum (bold red) if a query is active.
            if self._lut_query_point is not None:
                fwd = self._lut_forward(self._lut_query_point)
                qax = self.lut_fig.axes[0]
                label = ", ".join(
                    f"{ax['disp']}={self._lut_query_point[ax['name']]:g}"
                    for ax in self._lut_axinfo)
                qax.plot(active.wavenumber, fwd[quantity], color="red", lw=2.0,
                         label=label, zorder=5)
                qax.legend(loc="best", fontsize=8)
        except Exception as e:
            self.lut_fig.clear()
            ax = self.lut_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Cannot render: {e}", ha="center", va="center",
                    color="crimson", transform=ax.transAxes, wrap=True)
            ax.set_axis_off()

        # Note: plot_lut_slice manages its own colorbar + layout engine; don't
        # call tight_layout()/subplots_adjust() here (it would conflict).
        self.lut_canvas.draw()

    def _on_lut_query(self) -> None:
        """Interpolate the active LUT at the entered point and overlay the spectrum."""
        active = self._active_lut_obj
        if active is None:
            messagebox.showwarning("No LUT", "Build or load a LUT first.")
            return
        point: dict[str, float] = {}
        try:
            for ax in self._lut_axinfo:
                point[ax["name"]] = float(self._lut_query_vars[ax["name"]].get())
        except (ValueError, KeyError) as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        # Flag any axis value outside the tabulated range.
        out_of_bounds = [
            ax for ax in self._lut_axinfo
            if not (float(ax["values"].min()) <= point[ax["name"]]
                    <= float(ax["values"].max()))
        ]
        ptstr = ", ".join(
            f"{ax['disp']}={point[ax['name']]:g}" for ax in self._lut_axinfo)

        q = self.lut_quantity_var.get()
        fwd = self._lut_forward(point)

        # N-D RegularGridInterpolator returns all-NaN outside the grid (it does
        # not extrapolate, unlike the 1-D PCHIP). Catch that before reducing.
        if not np.any(np.isfinite(fwd[q])):
            names = ", ".join(ax["disp"] for ax in out_of_bounds) or "the grid"
            messagebox.showwarning(
                "Outside LUT range",
                f"Query point [{ptstr}] is outside the LUT grid "
                f"({names}); the N-D interpolator does not extrapolate.\n\n"
                "Pick values within the tabulated axis ranges.")
            self._lut_query_point = None
            self.lut_info_label.config(
                text=f"{ptstr}   |   outside LUT range — no value",
                foreground="crimson")
            self._plot_lut_slice()
            self.logger.warning(f"LUT query [{ptstr}] outside grid — all-NaN")
            return

        self._lut_query_point = point
        self._plot_lut_slice()

        peak = float(np.nanmax(fwd[q]))
        wn_peak = float(active.wavenumber[int(np.nanargmax(fwd[q]))])
        extrap = "  (extrapolated — outside LUT range)" if out_of_bounds else ""
        self.lut_info_label.config(
            text=f"{ptstr}   |   {q} peak {peak:.3g} at {wn_peak:.0f} cm⁻¹{extrap}",
            foreground="black")
        self.logger.info(f"LUT query [{ptstr}] → {q} peak {peak:.3g}")

    def _on_lut_query_clear(self) -> None:
        """Remove the query overlay."""
        self._lut_query_point = None
        self.lut_info_label.config(text="", foreground="gray")
        self._plot_lut_slice()

    def _on_visualize_particle(self) -> None:
        """Open (or focus) the persistent size-distribution preview popup."""
        if self._sd_win is not None and self._sd_win.winfo_exists():
            # Already open — just refresh and bring to front.
            self._refresh_sizedist_popup()
            self._sd_win.lift()
            return
        self._build_sizedist_popup()
        self._refresh_sizedist_popup()

    def _build_sizedist_popup(self) -> None:
        """Create the single always-on-top size-distribution preview window."""
        win = tk.Toplevel(self)
        win.title("Size Distribution Preview")
        win.geometry("820x520")
        win.attributes("-topmost", True)        # always on top
        win.protocol("WM_DELETE_WINDOW", self._close_sizedist_popup)

        # Control row mapping to plot_size_dist keywords
        ctrl = ttk.Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(ctrl, text="Style:").pack(side=tk.LEFT)
        self._sd_style_var = tk.StringVar(value="line")
        sd_style = ttk.Combobox(
            ctrl, textvariable=self._sd_style_var, width=8, state="readonly",
            values=["line", "shaded", "hist"],
        )
        sd_style.pack(side=tk.LEFT, padx=(2, 10))
        sd_style.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sizedist_popup())

        ttk.Label(ctrl, text="X:").pack(side=tk.LEFT)
        self._sd_xscale_var = tk.StringVar(value="log")
        sd_x = ttk.Combobox(
            ctrl, textvariable=self._sd_xscale_var, width=7, state="readonly",
            values=["log", "linear"],
        )
        sd_x.pack(side=tk.LEFT, padx=(2, 8))
        sd_x.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sizedist_popup())

        ttk.Label(ctrl, text="Y:").pack(side=tk.LEFT)
        self._sd_yscale_var = tk.StringVar(value="linear")
        sd_y = ttk.Combobox(
            ctrl, textvariable=self._sd_yscale_var, width=7, state="readonly",
            values=["linear", "log"],
        )
        sd_y.pack(side=tk.LEFT, padx=(2, 8))
        sd_y.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sizedist_popup())

        ttk.Label(ctrl, text="Color:").pack(side=tk.LEFT)
        self._sd_color_var = tk.StringVar(value="tab:blue")
        sd_c = ttk.Combobox(
            ctrl, textvariable=self._sd_color_var, width=10, state="readonly",
            values=["tab:blue", "tab:red", "tab:green", "goldenrod"],
        )
        sd_c.pack(side=tk.LEFT, padx=(2, 8))
        sd_c.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sizedist_popup())

        self._sd_normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl, text="Normalize", variable=self._sd_normalize_var,
            command=self._refresh_sizedist_popup,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(ctrl, text="Export…", command=self._on_export_pdf).pack(side=tk.RIGHT)

        # Figure + navigation toolbar (Save figure, zoom, pan)
        self._sd_fig = plt.Figure(figsize=(6.4, 4.2), dpi=100)
        self._sd_canvas = FigureCanvasTkAgg(self._sd_fig, master=win)
        toolbar = NavigationToolbar2Tk(self._sd_canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._sd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._sd_win = win

    def _close_sizedist_popup(self) -> None:
        """Destroy the preview popup and clear references."""
        if self._sd_win is not None:
            self._sd_win.destroy()
        self._sd_win = None
        self._sd_fig = None
        self._sd_canvas = None

    def _refresh_sizedist_popup(self, *_args) -> None:
        """Rebuild the distribution from current panel params and redraw.

        No-op if the popup is closed.  Called on Visualize, on any popup
        option change, and whenever the particle parameters change.
        """
        if self._sd_win is None or not self._sd_win.winfo_exists():
            return
        try:
            r0_um = float(self.r0_var.get())
            n_d = float(self.ntotal_var.get())
            if r0_um <= 0 or n_d <= 0:
                raise ValueError
            ptype = self.dist_type_var.get()
            sigma = 1.0
            if ptype != "Single":
                sigma = float(self.sigma_var.get())
                if sigma <= 0:
                    raise ValueError
            dist = self._build_size_dist(ptype, r0_um, sigma, n_d)
        except ValueError:
            return  # incomplete/invalid params — leave the last good plot

        self._sd_dist = dist   # remember for Export
        self._sd_fig.clear()
        plot_size_dist(
            dist,
            style=self._sd_style_var.get(),
            yscale=self._sd_yscale_var.get(),
            xscale=self._sd_xscale_var.get(),
            color=self._sd_color_var.get(),
            normalize=self._sd_normalize_var.get(),
            title=f"{ptype}",
            fig=self._sd_fig,
            show=False,
        )
        self._sd_fig.tight_layout()
        self._sd_canvas.draw()

    def _on_export_pdf(self) -> None:
        """Export the previewed size distribution as a CSV PDF (savePDF)."""
        dist = getattr(self, "_sd_dist", None)
        if dist is None:
            messagebox.showwarning("No distribution", "Nothing to export yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Export PDF as CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv",
        )
        if not path:
            return
        try:
            # Sample the distribution as dN/d ln r over its radius range.
            r_lo, r_hi = dist.bounds()
            r_um = np.logspace(np.log10(r_lo * 1e4), np.log10(r_hi * 1e4), 500)
            n_r = dist.pdf(r_um * 1e-4)
            dndlnr = n_r * (r_um * 1e-4)
            savePDF(path, r_um, dndlnr,
                    substance=self.dist_type_var.get(),
                    normalized=self._sd_normalize_var.get())
            self.logger.info(f"Exported size-distribution PDF → {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self.logger.error(f"PDF export failed: {e}")

    def _build_size_dist(self, particle_type: str, r0_um: float,
                         sigma: float, n_d: float):
        """Construct a size-distribution object from the panel parameters.

        Single → MonodisperseDist (delta).  ModifiedGamma needs explicit
        radius cutoffs, which the panel does not expose; they are bracketed
        generously around the effective radius r₀.
        """
        if particle_type == "Single":
            return MonodisperseDist.from_um(r0_um, n=n_d)
        if particle_type == "LogNormal":
            return LogNormalDist.from_um(r0_um, sigma, n=n_d)
        if particle_type == "Gaussian":
            return GaussianDist.from_um(r0_um, sigma, n=n_d)
        if particle_type == "ModifiedGamma":
            # a = effective radius (r₀), b = effective variance (σ/s field).
            return ModifiedGammaDist.from_um(
                r0_um, sigma, r0_um * 0.05, r0_um * 15.0, n=n_d
            )
        if particle_type == "Custom":
            if self._loaded_pdf is None:
                raise ValueError("No PDF loaded — use 'Load PDF…' first.")
            r_um, dndlnr = self._loaded_pdf
            return TabulatedDist.from_dndlnr(r_um, dndlnr, n=n_d)
        raise ValueError(f"Unknown particle type: {particle_type!r}")

    def _read_density(self) -> float | None:
        """Parse the density entry → positive float, or None if blank/invalid.

        None means "let AerosolXSC auto-resolve" (which raises for materials
        with no density mapping — those require a manual entry).
        """
        raw = self.density_var.get().strip()
        if not raw:
            return None
        val = float(raw)            # may raise ValueError → caught by caller
        if val <= 0:
            raise ValueError("Density must be positive")
        return val

    def _on_compute_single(self) -> None:
        """Compute single particle or size distribution properties (threaded)."""
        if self.ri is None:
            messagebox.showwarning("Missing RI", "Please select a refractive index first")
            return

        try:
            r0_um = float(self.r0_var.get())
            if r0_um <= 0:
                raise ValueError("r₀ must be positive")

            # Density: blank → auto-resolve; otherwise the manual value.
            density = self._read_density()

            # N_d — number density [cm⁻³].  Used by both paths: it scales the
            # extensive B_* coefficients; Q/C/M are independent of it.
            n_d = float(self.ntotal_var.get())
            if n_d <= 0:
                raise ValueError("N_d must be positive")

            particle_type = self.dist_type_var.get()

            # All particle types — including Single (MonodisperseDist) — go
            # through one path: AerosolXSC.  A monodisperse delta is handled
            # exactly (single-node evaluation, no quadrature) inside
            # mieSizeDist, so Single yields the same Q/B/C/M *and* phase as a
            # distribution, with no separate code path.
            sigma = 1.0
            if particle_type != "Single":
                sigma = float(self.sigma_var.get())
                if sigma <= 0:
                    raise ValueError("σ/s must be positive")
            self.dist = self._build_size_dist(particle_type, r0_um, sigma, n_d)
            self.xsc_kind = particle_type

            with_phase = self._phase_var.get()
            n_angles = n_legendre = None
            if with_phase:
                n_angles = int(self.phase_nangles_var.get())
                n_legendre = int(self.phase_nlegendre_var.get())
                if n_angles < 2 or n_legendre < 1:
                    raise ValueError("Phase: N angles ≥ 2 and N Legendre ≥ 1")

            kind_msg = (f"single particle: r₀={r0_um:.2f} µm, N_d={n_d:.3g} cm⁻³"
                        if particle_type == "Single"
                        else (f"{particle_type}, r₀={r0_um:.2f} µm, "
                              f"σ/s={sigma:.2f}, N_d={n_d:.3g} cm⁻³"))
            self.logger.info(f"Computing {kind_msg}" + ("  (+ phase)" if with_phase else ""))
            if with_phase and self.backend == "fortran":
                self.logger.info("  phase requires the Python backend — using Python for this pass")
            self._begin_op("Computing XSC…" + (" + phase" if with_phase else ""))
            threading.Thread(
                target=self._compute_dist_worker,
                args=(with_phase, n_angles, n_legendre, density), daemon=False,
            ).start()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.logger.error(f"Computation validation failed: {e}")

    def _begin_op(self, message: str, total: int | None = None) -> None:
        """Begin a long operation's progress display.

        ``total=None`` → indeterminate marquee (unknown duration).
        ``total=N``    → determinate 0..N bar, advanced via _progress_step.
        """
        self._status_var.set(message)
        self._op_message = message
        try:
            self.progress_bar.stop()
        except tk.TclError:
            pass
        if total is None:
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(12)   # marquee animation (ms per step)
        else:
            self.progress_bar.config(mode="determinate", maximum=total, value=0)

    def _progress_step(self, current: int, total: int) -> None:
        """Advance the determinate progress bar (main thread)."""
        try:
            self.progress_bar.config(value=current)
            self._status_var.set(f"{self._op_message} ({current}/{total})")
        except tk.TclError:
            pass

    def _post_progress(self, current: int, total: int) -> None:
        """Thread-safe progress callback for worker threads."""
        self._safe_after(lambda: self._progress_step(current, total))

    def _end_op(self, message: str = "Ready") -> None:
        """Stop the progress bar and reset the status line."""
        try:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=0)
        except tk.TclError:
            pass
        self._status_var.set(message)

    def _safe_after(self, callback) -> None:
        """Schedule a main-thread callback, ignoring a closed window.

        Worker threads call this to post results; if the user closed the
        window mid-computation the Tk interpreter is gone, so we swallow the
        resulting error instead of dumping a traceback.
        """
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _compute_dist_worker(self, with_phase: bool = False,
                             n_angles: int | None = None,
                             n_legendre: int | None = None,
                             density: float | None = None) -> None:
        """Worker thread: XSC computation via AerosolXSC, optionally with phase."""
        try:
            # Phase is Python-only, so force the Python backend when requested.
            backend = "python" if with_phase else self.backend
            xsc = AerosolXSC(ri=self.ri, dist=self.dist, wn=self.wavenumber,
                             backend=backend, density=density)
            if with_phase:
                theta = np.linspace(0.0, 180.0, int(n_angles))
                xsc.compute(theta_pts=theta, n_legendre=int(n_legendre), progress=False)
            else:
                xsc.compute(progress=False)
            self.xsc = xsc
            self.xsc_result = xsc.result
            self.xsc_density = xsc.density
            self.xsc_phase = xsc if with_phase else None
            self._safe_after(self._on_xsc_done)
        except Exception as e:
            msg = f"Computation failed: {e}\n{traceback.format_exc()}"
            self._safe_after(lambda: (self.logger.error(msg), self._end_op("Error")))

    def _on_xsc_done(self) -> None:
        """Main-thread callback after a computation completes."""
        self._end_op("XSC done")
        self.logger.info(f"XSC done: MEC peak {self.xsc_result.MEC.max():.3g} cm²/g")
        self._plot_xsc()
        self.notebook.select(self.xsc_tab)
        # Phase tab: populate if phase was computed in this pass, else reset.
        self._plot_phase()
        if self.xsc_phase is not None:
            self.logger.info("Phase matrix ready (see Phase Function tab).")

    def _lut_build_dialog(self) -> dict | None:
        """Modal popup collecting 1-D or N-D LUT build parameters.

        Returns a params dict (keyed by ``mode`` = '1d' | 'nd'), or None if
        cancelled.

        1-D: a log-spaced r₀ sweep for the currently-selected RI (AerosolLUT).
        N-D: r₀ × audit-grid sweep axes (composition/temperature) for a
        registry substance/source (MultiDimLUT); swept axes use ALL audit
        points; σ/s is fixed at 1.5 in the N-D engine.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Build LUT")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        result: dict = {}

        outer = ttk.Frame(dlg, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # --- Mode toggle ---
        mode_var = tk.StringVar(value="1d")
        mrow = ttk.Frame(outer)
        mrow.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(mrow, text="Type:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(mrow, text="1-D (r₀)", variable=mode_var, value="1d",
                        command=lambda: _switch()).pack(side=tk.LEFT)
        ttk.Radiobutton(mrow, text="N-D (composition / temperature)",
                        variable=mode_var, value="nd",
                        command=lambda: _switch()).pack(side=tk.LEFT, padx=(6, 0))

        # --- 1-D frame ---
        f1 = ttk.LabelFrame(outer, text="1-D r₀ sweep", padding=8)
        disttype_v = tk.StringVar(value="log_normal")
        r0min_v = tk.StringVar(value="0.1")
        r0max_v = tk.StringVar(value="10.0")
        nnodes_v = tk.StringVar(value="20")
        sigma_v = tk.StringVar(value=self.sigma_var.get() if hasattr(self, "sigma_var") else "1.5")

        ttk.Label(f1, text="Dist type:").grid(row=0, column=0, sticky="w", pady=2)
        dt_combo = ttk.Combobox(f1, textvariable=disttype_v, width=12, state="readonly",
                                values=["log_normal", "gaussian", "monodisperse"])
        dt_combo.grid(row=0, column=1, sticky="w")
        sigma_lbl = ttk.Label(f1, text="σ / s:")
        sigma_ent = None
        for i, (lab, v) in enumerate([("r₀ min (µm):", r0min_v), ("r₀ max (µm):", r0max_v),
                                      ("N nodes:", nnodes_v)], start=1):
            ttk.Label(f1, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f1, textvariable=v, width=12).grid(row=i, column=1, sticky="w")
        sigma_lbl.grid(row=4, column=0, sticky="w", pady=2)
        sigma_ent = ttk.Entry(f1, textvariable=sigma_v, width=12)
        sigma_ent.grid(row=4, column=1, sticky="w")

        def _on_dt(*_a):
            mono = disttype_v.get() == "monodisperse"
            sigma_ent.config(state="disabled" if mono else "normal")
            sigma_lbl.config(foreground="gray" if mono else "black")
        dt_combo.bind("<<ComboboxSelected>>", _on_dt)
        _on_dt()

        # --- N-D frame ---
        f2 = ttk.LabelFrame(outer, text="N-D sweep (registry substances)", padding=8)
        subs = registry_substances()
        substance_v = tk.StringVar(value=subs[0] if subs else "")
        source_v = tk.StringVar()
        nd_r0min_v = tk.StringVar(value="0.1")
        nd_r0max_v = tk.StringVar(value="10.0")
        nd_nnodes_v = tk.StringVar(value="15")

        ttk.Label(f2, text="Substance:").grid(row=0, column=0, sticky="w", pady=2)
        sub_combo = ttk.Combobox(f2, textvariable=substance_v, values=subs,
                                 width=18, state="readonly")
        sub_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(f2, text="Source:").grid(row=1, column=0, sticky="w", pady=2)
        src_combo = ttk.Combobox(f2, textvariable=source_v, width=18, state="readonly")
        src_combo.grid(row=1, column=1, sticky="w")
        ttk.Label(f2, text="(stores Q/C/B/M + g)", foreground="gray").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2)
        for i, (lab, v) in enumerate([("r₀ min (µm):", nd_r0min_v),
                                      ("r₀ max (µm):", nd_r0max_v),
                                      ("N r₀ nodes:", nd_nnodes_v)], start=3):
            ttk.Label(f2, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f2, textvariable=v, width=12).grid(row=i, column=1, sticky="w")

        # Distribution family + spread (r₀ sweep), mirroring the 1-D LUT.
        nd_disttype_v = tk.StringVar(value="log_normal")
        nd_sigma_v = tk.StringVar(value="1.5")
        ttk.Label(f2, text="Dist type:").grid(row=6, column=0, sticky="w", pady=2)
        nd_dt_combo = ttk.Combobox(f2, textvariable=nd_disttype_v, width=12, state="readonly",
                                   values=["log_normal", "gaussian", "monodisperse"])
        nd_dt_combo.grid(row=6, column=1, sticky="w")
        nd_sigma_lbl = ttk.Label(f2, text="σ / s:")
        nd_sigma_lbl.grid(row=7, column=0, sticky="w", pady=2)
        nd_sigma_ent = ttk.Entry(f2, textvariable=nd_sigma_v, width=12)
        nd_sigma_ent.grid(row=7, column=1, sticky="w")

        def _on_nd_dt(*_a):
            mono = nd_disttype_v.get() == "monodisperse"
            nd_sigma_ent.config(state="disabled" if mono else "normal")
            nd_sigma_lbl.config(foreground="gray" if mono else "black")
        nd_dt_combo.bind("<<ComboboxSelected>>", _on_nd_dt)
        _on_nd_dt()

        axis_summary = ttk.Label(f2, text="", foreground="gray", justify=tk.LEFT)
        axis_summary.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Grid preview — visual of the swept-axis nodes that will be built.
        prev_fig = plt.Figure(figsize=(3.2, 2.7), dpi=90)
        prev_canvas = FigureCanvasTkAgg(prev_fig, master=f2)
        prev_canvas.get_tk_widget().grid(row=0, column=2, rowspan=9,
                                         padx=(14, 0), sticky="n")

        def _draw_preview(axes: dict, total: int) -> None:
            prev_fig.clear()
            ax = prev_fig.add_subplot(111)
            items = list(axes.items())
            if len(items) >= 2:
                (n1, i1), (n2, i2) = items[0], items[1]
                X, Y = np.meshgrid(i1["points"], i2["points"])
                ax.scatter(X, Y, s=14, color="tab:blue", edgecolor="white", linewidth=0.4)
                ax.set_xlabel(f"{n1} ({i1['units']})", fontsize=8)
                ax.set_ylabel(f"{n2} ({i2['units']})", fontsize=8)
            elif len(items) == 1:
                n1, i1 = items[0]
                pts = i1["points"]
                ax.scatter(pts, np.zeros(len(pts)), s=18, color="tab:blue",
                           edgecolor="white", linewidth=0.4)
                ax.set_xlabel(f"{n1} ({i1['units']})", fontsize=8)
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "(no swept axes)", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_axis_off()
            swept = 1
            for _, info in items:
                swept *= len(info["points"])
            ax.set_title(f"{swept} RI nodes (loaded per r₀)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            prev_fig.tight_layout()
            prev_canvas.draw()

        def _refresh_nd(*_a):
            sub = substance_v.get()
            srcs = registry_sources(sub)
            src_combo.config(values=srcs)
            if source_v.get() not in srcs:
                source_v.set(srcs[0] if srcs else "")
            axes = registry_axes(sub)
            try:
                n_r0 = int(nd_nnodes_v.get())
            except ValueError:
                n_r0 = 0
            total = n_r0 if n_r0 > 0 else 1
            lines = [f"r₀: {nd_r0min_v.get()}–{nd_r0max_v.get()} µm ({n_r0} pts)"]
            for name, info in axes.items():
                pts = info["points"]
                lines.append(f"{name}: {min(pts):g}–{max(pts):g} {info['units']} "
                             f"({len(pts)} pts, all audit)")
                total *= len(pts)
            lines.append(f"→ grid: {total} nodes × {len(self.wavenumber)} wn")
            axis_summary.config(text="\n".join(lines))
            _draw_preview(axes, total)

        sub_combo.bind("<<ComboboxSelected>>", _refresh_nd)
        for _v in (nd_r0min_v, nd_r0max_v, nd_nnodes_v):
            _v.trace_add("write", _refresh_nd)
        _refresh_nd()

        def _switch():
            if mode_var.get() == "1d":
                f2.grid_remove(); f1.grid(row=1, column=0, sticky="ew")
            else:
                f1.grid_remove(); f2.grid(row=1, column=0, sticky="ew")
        _switch()

        def _ok():
            mode = mode_var.get()
            try:
                if mode == "1d":
                    dist_type = disttype_v.get()
                    mono = dist_type == "monodisperse"
                    r0_min = float(r0min_v.get()); r0_max = float(r0max_v.get())
                    n_nodes = int(nnodes_v.get())
                    sigma = 1.5 if mono else float(sigma_v.get())
                    if r0_min <= 0 or r0_max <= 0 or n_nodes < 2 or r0_min >= r0_max:
                        raise ValueError
                    if not mono and sigma <= 0:
                        raise ValueError
                    result.update(mode="1d", dist_type=dist_type, r0_min=r0_min,
                                  r0_max=r0_max, n_nodes=n_nodes, sigma=sigma)
                else:
                    r0_min = float(nd_r0min_v.get()); r0_max = float(nd_r0max_v.get())
                    n_nodes = int(nd_nnodes_v.get())
                    nd_dist = nd_disttype_v.get()
                    nd_mono = nd_dist == "monodisperse"
                    nd_s = 1.5 if nd_mono else float(nd_sigma_v.get())
                    if r0_min <= 0 or r0_max <= 0 or n_nodes < 2 or r0_min >= r0_max:
                        raise ValueError
                    if not substance_v.get() or not source_v.get():
                        raise ValueError
                    if not nd_mono and nd_s <= 0:
                        raise ValueError
                    result.update(mode="nd", substance=substance_v.get(),
                                  source=source_v.get(),
                                  r0_min=r0_min, r0_max=r0_max, n_nodes=n_nodes,
                                  dist_type=nd_dist, s=nd_s)
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Need r₀ min < r₀ max (both > 0), N nodes ≥ 2"
                    + (", σ/s > 0." if mode == "1d" else ", and a substance/source."),
                    parent=dlg)
                return
            dlg.destroy()

        btns = ttk.Frame(outer)
        btns.grid(row=2, column=0, pady=(10, 0))
        ttk.Button(btns, text="Build", command=_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        dlg.wait_window()
        return result or None

    def _on_build_lut(self) -> None:
        """Open the build-params popup, then build a 1-D or N-D LUT (threaded)."""
        params = self._lut_build_dialog()
        if params is None:
            return

        if params["mode"] == "1d":
            if self.ri is None:
                messagebox.showwarning("Missing RI", "Select a refractive index first.")
                return
            try:
                density = self._read_density()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
                return
            r0_grid = np.logspace(np.log10(params["r0_min"]),
                                  np.log10(params["r0_max"]), params["n_nodes"])
            self.logger.info(
                f"Building 1-D LUT ({params['dist_type']}): {params['n_nodes']} "
                f"r₀ nodes, {len(self.wavenumber)} wavenumbers")
            self._begin_op("Building LUT…", total=params["n_nodes"])
            threading.Thread(
                target=self._build_lut_worker,
                args=(r0_grid, params["sigma"], density, params["dist_type"]),
                daemon=False,
            ).start()
        else:
            # Total compute nodes = r₀ nodes × product of swept-axis lengths.
            total = params["n_nodes"]
            for info in registry_axes(params["substance"]).values():
                total *= len(info["points"])
            self.logger.info(
                f"Building N-D LUT: {params['substance']} ({params['source']}) "
                f"— {total} nodes, all quantities (Q/C/B/M + g)")
            self._begin_op("Building N-D LUT…", total=total)
            threading.Thread(
                target=self._build_ndlut_worker, args=(params,), daemon=False,
            ).start()

    def _build_ndlut_worker(self, params: dict) -> None:
        """Worker: build a MultiDimLUT (r₀ × audit sweep axes) and cache it."""
        try:
            r0_grid = np.logspace(np.log10(params["r0_min"]),
                                  np.log10(params["r0_max"]), params["n_nodes"])
            axes = [LUTAxis("r0", r0_grid, "µm", log_spacing=True,
                            fixed_for_ri_load=True)]
            for name, info in registry_axes(params["substance"]).items():
                axes.append(LUTAxis(name, np.asarray(info["points"], float),
                                    info["units"], fixed_for_ri_load=False))

            # Content-addressed cache: reuse an identical LUT or build+cache.
            mdlut = MultiDimLUT.load_or_compute(
                params["substance"], params["source"], axes, self.wavenumber,
                dist_type=params["dist_type"], s=params["s"],
                backend=self.backend, progress=False,
                progress_callback=self._post_progress,
            )
            self.mdlut = mdlut
            self._safe_after(lambda: (
                self.logger.info(
                    f"N-D LUT built: shape {mdlut.shape} "
                    f"({params['substance']}, {params['source']}, "
                    f"{len(mdlut.quantities)} quantities)"),
                self._end_op("N-D LUT ready"),
                self._activate_lut_display("nd"),
                self.notebook.select(self.lut_tab),
            ))
        except Exception as e:
            msg = f"N-D LUT build failed: {e}\n{traceback.format_exc()}"
            self._safe_after(lambda: (self.logger.error(msg), self._end_op("Error")))

    def _build_lut_worker(self, r0_grid: np.ndarray, sigma: float,
                          density: float | None = None,
                          dist_type: str = "log_normal") -> None:
        """Worker thread for LUT computation."""
        try:
            self.lut = AerosolLUT.load_or_compute(
                self.ri,
                r0_um=r0_grid,
                s=sigma,
                wn=self.wavenumber,
                dist_type=dist_type,
                backend=self.backend,
                cache_phase=True,
                density=density,
                progress=False,
                progress_callback=self._post_progress,
            )
            n = len(r0_grid)
            self._safe_after(lambda: (self.logger.info(f"LUT built: {n} nodes cached"),
                                      self._end_op("LUT ready"),
                                      self._on_lut_ready()))
        except Exception as e:
            msg = f"LUT build failed: {e}\n{traceback.format_exc()}"
            self._safe_after(lambda: (self.logger.error(msg), self._end_op("Error")))

    # 1-D vs N-D Load-chooser column layouts: (name, width)
    _LUT_LOAD_COLS_1D = (("substance", 150), ("dist", 80), ("s", 45),
                         ("N_r0", 45), ("r0 range (µm)", 120), ("N_wn", 50),
                         ("wn range (cm⁻¹)", 130), ("phase", 50), ("size", 70))
    _LUT_LOAD_COLS_ND = (("substance", 130), ("source", 110), ("dist", 90),
                         ("s", 45), ("axes", 200), ("N_wn", 50),
                         ("wn range (cm⁻¹)", 130), ("quantities", 60), ("size", 70))

    def _on_load_lut(self) -> None:
        """Open a chooser listing cached 1-D or N-D LUTs (Browse… and Delete)."""
        dlg = tk.Toplevel(self)
        dlg.title("Load LUT")
        dlg.geometry("880x380")
        dlg.transient(self)
        dlg.grab_set()

        top = ttk.Frame(dlg)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(top, text="Kind:").pack(side=tk.LEFT)
        kind_var = tk.StringVar(value="1d")
        ttk.Radiobutton(top, text="1-D", variable=kind_var, value="1d",
                        command=lambda: _populate()).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="N-D", variable=kind_var, value="nd",
                        command=lambda: _populate()).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(top, text="(double-click to load)", foreground="gray").pack(side=tk.LEFT)

        tree = ttk.Treeview(dlg, show="headings", height=11)
        vsb = ttk.Scrollbar(dlg, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(8, 0), pady=2)

        empty_label = ttk.Label(dlg, text="(cache empty)", foreground="gray")
        path_by_iid: dict[str, str] = {}

        def _set_columns(spec):
            cols = [c for c, _ in spec]
            tree.config(columns=cols)
            for c, w in spec:
                tree.heading(c, text=c)
                tree.column(c, width=w, anchor=tk.W, stretch=False)

        def _populate():
            tree.delete(*tree.get_children())
            path_by_iid.clear()
            nd = kind_var.get() == "nd"
            _set_columns(self._LUT_LOAD_COLS_ND if nd else self._LUT_LOAD_COLS_1D)
            try:
                entries = (MultiDimLUT.cache_entries() if nd
                           else AerosolLUT.cache_entries())
            except Exception as exc:
                self.logger.warning(f"Could not read LUT cache: {exc}")
                entries = []
            for e in entries:
                wn_rng = (f"{e['wn_min']:.0f}–{e['wn_max']:.0f}"
                          if "error" not in e else "")
                if "error" in e:
                    vals = ["(unreadable)"] + [""] * (len(tree["columns"]) - 2) \
                           + [f"{e['size_kb']:.0f} KB"]
                elif nd:
                    axes_str = ", ".join(f"{n}×{k}" for n, k in e["axes"])
                    vals = [e["substance"] or "—", e["source"], e["dist_type"],
                            f"{e['s']:.2f}", axes_str, e["n_wn"], wn_rng,
                            len(e["quantities"]), f"{e['size_kb']:.0f} KB"]
                else:
                    r0_rng = f"{e['r0_min']:.3g}–{e['r0_max']:.3g}"
                    vals = [e["substance"] or e["category"] or "—", e["dist_type"],
                            f"{e['s']:.2f}", e["n_r0"], r0_rng, e["n_wn"], wn_rng,
                            "✓" if e["phase"] else "", f"{e['size_kb']:.0f} KB"]
                iid = tree.insert("", tk.END, values=vals)
                path_by_iid[iid] = e["path"]
            (empty_label.pack(pady=4) if not path_by_iid else empty_label.pack_forget())

        def _load_path(path):
            if kind_var.get() == "nd":
                self._load_ndlut_path(path)
            else:
                self._load_lut_path(path)

        def _load_selected(_event=None):
            sel = tree.selection()
            if not sel:
                return
            path = path_by_iid.get(sel[0])
            dlg.destroy()
            if path:
                _load_path(path)

        def _browse():
            dlg.destroy()
            path = filedialog.askopenfilename(
                title="Load LUT", filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")])
            if path:
                _load_path(path)

        def _delete_selected():
            sel = tree.selection()
            if not sel:
                return
            path = path_by_iid.get(sel[0])
            if not path:
                return
            if not messagebox.askyesno(
                    "Delete LUT", f"Delete this cached LUT file?\n\n{Path(path).name}",
                    parent=dlg):
                return
            try:
                Path(path).unlink()
                self.logger.info(f"Deleted cached LUT {Path(path).name}")
            except OSError as exc:
                messagebox.showerror("Delete Error", str(exc), parent=dlg)
            _populate()

        tree.bind("<Double-1>", _load_selected)
        btns = ttk.Frame(dlg)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        ttk.Button(btns, text="Load", command=_load_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Browse…", command=_browse).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete", command=_delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=2)

        _populate()

    def _load_ndlut_path(self, path: str) -> None:
        """Load an N-D LUT (MultiDimLUT) and show it in the LUT tab."""
        self._begin_op("Loading N-D LUT…")
        try:
            self.mdlut = MultiDimLUT.load(path)
            self.logger.info(f"N-D LUT loaded from {Path(path).name}: "
                             f"shape {self.mdlut.shape}, "
                             f"{len(self.mdlut.quantities)} quantities")
            self._end_op("N-D LUT loaded")
            self._activate_lut_display("nd")
            self.notebook.select(self.lut_tab)
        except Exception as e:
            messagebox.showerror("LUT Load Error", str(e))
            self.logger.error(f"Failed to load N-D LUT: {e}")
            self._end_op("Error")

    def _load_lut_path(self, path: str) -> None:
        """Load a LUT from a concrete path with status feedback."""
        self._begin_op("Loading LUT…")
        try:
            self.lut = AerosolLUT.load(path)
            self.logger.info(f"LUT loaded from {Path(path).name}")
            self._end_op("LUT loaded")
            self._on_lut_ready()
        except Exception as e:
            messagebox.showerror("LUT Load Error", str(e))
            self.logger.error(f"Failed to load LUT: {e}")
            self._end_op("Error")

    def _on_lut_ready(self) -> None:
        """Refresh the LUT tab after a 1-D build/load and switch to it."""
        self._activate_lut_display("1d")
        self.notebook.select(self.lut_tab)

    def _on_save_lut(self) -> None:
        """Save current LUT to file."""
        if self.lut is None:
            messagebox.showwarning("No LUT", "Build or load a LUT first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save LUT",
            filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")],
            defaultextension=".nc",
        )
        if not path:
            return
        self._begin_op("Saving LUT…")
        try:
            self.lut.save(path)
            self.logger.info(f"LUT saved to {Path(path).name}")
            self._end_op("LUT saved")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            self.logger.error(f"Failed to save LUT: {e}")
            self._end_op("Error")

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About MieViewer",
            "MieViewer v0.1.0\n\n"
            "Interactive GUI for Mie scattering calculations.\n"
            "Part of mieEODG package.\n\n"
            "https://eodg.atm.ox.ac.uk/MIE/",
        )

    def _on_close(self) -> None:
        """Clean shutdown."""
        if self._sd_win is not None and self._sd_win.winfo_exists():
            self._sd_win.destroy()
        # Detach our log handlers so they don't linger on the module-level
        # 'mieEODG.gui' logger after the window is gone.
        for _h in (getattr(self, "gui_handler", None),
                   getattr(self, "file_handler", None)):
            if _h is not None:
                self.logger.removeHandler(_h)
                try:
                    _h.close()
                except Exception:
                    pass
        plt.close("all")
        self.destroy()
        # Also destroy parent to exit event loop completely
        if self.parent and self.parent.winfo_exists():
            self.parent.destroy()


# ============================================================================
# Entry Point
# ============================================================================

def launch() -> None:
    """Launch MieViewer GUI."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window; MieViewer is a Toplevel
    viewer = MieViewer(root)
    viewer.protocol("WM_DELETE_WINDOW", viewer._on_close)
    viewer.mainloop()


if __name__ == "__main__":
    launch()
