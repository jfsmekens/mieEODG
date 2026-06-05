"""
AerosolXSC — primary user-facing object for Mie-based aerosol cross-section
and mass extinction coefficient calculations.

This module ties together refractive index reading (:mod:`mieEODG.refindex`),
size distribution integration (:mod:`mieEODG.mie_size_dist`), and aerosol
bulk properties (:mod:`mieEODG.constants`) into a single, self-contained
workflow object.

Workflow
--------
::

    from mieEODG import AerosolXSC, LogNormalDist

    xsc = AerosolXSC(
        ri      = 'Volcanic_Ash/Etna/Dequine_2020/etna-ash_Deguine_2020.ri',
        dist    = LogNormalDist(r0=1e-4, s=1.8),   # r0 in cm
        wn      = np.arange(650, 1350.5, 0.5),
        density = 2.6,                              # g/cm³
    )
    xsc.compute(progress=True)

    # Volume-normalised outputs (per unit volume concentration, cm³/cm³):
    print(xsc.MEC)   # mass extinction coefficient [cm²/g]
    print(xsc.MAC)   # mass absorption coefficient [cm²/g]
    print(xsc.MSC)   # mass scattering coefficient [cm²/g]

    xsc.save('outputs/etna_ash_lognormal.npz')
    xsc2 = AerosolXSC.load('outputs/etna_ash_lognormal.npz')

Output quantities
-----------------
All spectral outputs are 1-D arrays of shape ``(N_wavenumber,)``.

Q_ext, Q_sca, Q_abs
    Area-weighted efficiency factors (dimensionless):
    ``Q = B / G_avg`` where ``G_avg = ∫ π r² n(r) dr`` [cm²].

C_ext, C_sca, C_abs
    Volume-normalised cross-sections per particle [cm²]:
    ``C_ext = Q_ext × G_avg``
    (i.e. the bulk-average cross-section for the distribution, not per-particle).

B_ext, B_sca, B_abs
    Volume extinction / scattering / absorption coefficients [cm⁻¹],
    computed by ``mieSizeDist`` with ``Nd = 1`` number density.
    These scale linearly with number density.

MEC, MSC, MAC
    Mass extinction / scattering / absorption coefficients [cm²/g]:
    ``MEC = B_ext / (ρ × V_avg)``
    where ``V_avg = ∫ (4π/3) r³ n(r) dr`` is the mean volume [cm³].
    These are independent of assumed aerosol loading and are directly
    usable as optical depth per column mass density [g/cm²].

g
    Asymmetry parameter (area-weighted mean over the distribution).

N, V_avg, G_avg
    Distribution integrals: total number density [cm⁻³], mean volume [cm³],
    mean geometric cross-section [cm²].

Persistence
-----------
``.save(path)``
    Saves to ``.npz`` (recommended for single runs) or ``.nc`` (NetCDF4,
    for LUT stacks).  Format is inferred from the file extension.
``.load(path)``
    Class method; reconstructs an ``AerosolXSC`` from a saved file.

LUT mode
--------
Pass an array of ``r0`` values (or an array of ``s`` values) to
``LogNormalDist`` to run a size-parameter sweep.  Not yet implemented;
reserved for a future version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import aria_density
from .mie_size_dist import mieSizeDist
from .refindex import ARIADatabase, RefrIndex, readRI
from .size_dist import LogNormalDist, ModifiedGammaDist, SizeDistribution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class _AerosolResult:
    """
    Internal container for spectral Mie outputs from a single
    (refractive index, size distribution) combination.

    All spectral arrays have shape ``(N_wavenumber,)``.
    """
    wavenumber: np.ndarray

    # Efficiency factors (dimensionless)
    Q_ext: np.ndarray
    Q_sca: np.ndarray
    Q_abs: np.ndarray

    # Volume-normalised cross-sections [cm²]
    C_ext: np.ndarray
    C_sca: np.ndarray
    C_abs: np.ndarray

    # Volume extinction / scattering / absorption coefficients [cm⁻¹], Nd=1
    B_ext: np.ndarray
    B_sca: np.ndarray
    B_abs: np.ndarray

    # Mass coefficients [cm²/g]
    MEC: np.ndarray
    MSC: np.ndarray
    MAC: np.ndarray

    # Asymmetry parameter
    g: np.ndarray

    # Distribution integrals (scalars)
    N_avg:   float   # total number density [cm⁻³]
    V_avg:   float   # mean volume [cm³]
    G_avg:   float   # mean geometric cross-section [cm²]

    # Phase matrix — only populated when theta_pts is supplied to compute()
    # shape (N_wn, N_theta, 4) — columns are [F11, F33, F12, F34]
    # normalised to (1/2) * integral(F11, d(cosθ)) = 1
    phase_matrix: np.ndarray | None = None
    theta:        np.ndarray | None = None   # scattering angles [degrees]

    # Legendre coefficients — only populated when n_legendre is supplied to compute()
    # shape (N_wn, n_legendre) — DISORT pmom convention:
    #   P(cosθ) = Σ_l (2l+1) · pmom[l] · P_l(cosθ),  pmom[0] = 1
    pmom: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AerosolXSC:
    """
    Mie-based aerosol cross-section and mass coefficient calculator.

    Parameters
    ----------
    ri : str or Path or RefrIndex
        Refractive index source.  Accepts:

        * A relative path within the bundled ARIA database
          (e.g. ``'Volcanic_Ash/Etna/Dequine_2020/etna-ash_Deguine_2020.ri'``).
        * An absolute path to any ``.ri`` file.
        * A pre-loaded :class:`~mieEODG.refindex.RefrIndex` object.
    dist : SizeDistribution
        A :class:`~mieEODG.size_dist.LogNormalDist` or
        :class:`~mieEODG.size_dist.ModifiedGammaDist` instance.
        Units: all radii in **centimetres**, consistent with the cm-based
        wavenumber grid.
    wn : np.ndarray
        Wavenumber grid [cm⁻¹] on which to compute the cross-sections.
    density : float or None
        Bulk particle density [g/cm³].  Used to convert volume coefficients
        to mass coefficients (MEC, MSC, MAC).  If *None* (default), the
        density is derived automatically from ``ri.category`` and
        ``ri.composition`` via :func:`~mieEODG.constants.aria_density`.
        An explicit value always overrides auto-detection.  A
        ``ValueError`` is raised at construction time if the density cannot
        be determined.
    n_pts : int
        Number of quadrature points for the size integration.  Default 100.
    aria : ARIADatabase or None
        ARIA database instance to use when ``ri`` is a relative path.
        If *None*, the default bundled database is used.

    Attributes
    ----------
    result : _AerosolResult or None
        Populated after :meth:`compute` is called.
    ri : RefrIndex
        The refractive index data used (interpolated onto ``wn``).
    """

    def __init__(
        self,
        ri:      str | Path | RefrIndex,
        dist:    SizeDistribution,
        wn:      np.ndarray,
        density: float | None = None,
        n_pts:   int = 100,
        aria:    ARIADatabase | None = None,
        backend: str = 'python',
    ) -> None:
        self.wn      = np.asarray(wn, dtype=np.float64)
        self.dist    = dist
        self.n_pts   = int(n_pts)
        self.backend = backend.lower()
        self.result: _AerosolResult | None = None

        # --- Resolve refractive index ---
        if isinstance(ri, RefrIndex):
            self.ri = ri
        else:
            ri_path = Path(ri)
            if ri_path.is_absolute():
                self.ri = readRI(ri_path, wavenumber=self.wn, out_of_range='clip')
            else:
                db = aria if aria is not None else ARIADatabase()
                self.ri = db.read(str(ri_path), wavenumber=self.wn, out_of_range='clip')

        # --- Resolve density ---
        if density is not None:
            self.density = float(density)
        else:
            auto = aria_density(
                category      = self.ri.category,
                composition   = self.ri.composition,
                T_K           = self.ri.T_K,
                substance_key = self.ri.substance_key,
            )
            if auto is None:
                raise ValueError(
                    f"Cannot determine density automatically for "
                    f"category='{self.ri.category}', "
                    f"substance_key='{self.ri.substance_key}', "
                    f"composition={self.ri.composition}. "
                    "Supply density explicitly."
                )
            self.density = auto
            logger.debug(
                "Auto-resolved density: %.4f g/cm³  "
                "(category='%s', substance_key='%s', T_K=%s, composition=%s)",
                self.density, self.ri.category, self.ri.substance_key,
                self.ri.T_K, self.ri.composition,
            )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(
        self,
        progress:    bool | str = False,
        theta_pts:   np.ndarray | None = None,
        n_legendre:  int | None = None,
    ) -> 'AerosolXSC':
        """
        Run the Mie size-distribution integration and populate :attr:`result`.

        Parameters
        ----------
        progress : bool or str
            If True or a string, display a tqdm progress bar.  A string value
            is used as the bar label.  Default False.
        theta_pts : np.ndarray or None
            Scattering angles [degrees] at which to compute the phase matrix.
            If supplied, the size-integrated phase matrix
            [F11, F33, F12, F34] is stored in :attr:`phase_matrix` with shape
            ``(N_wn, N_theta, 4)``, normalised to the RT convention
            ``(1/2) * integral(F11, d(cosθ)) = 1``.  If *None* (default),
            only bulk efficiencies and the asymmetry parameter ``g`` are
            computed.

            .. note::
                Phase matrix and Legendre expansion are only supported with
                ``backend='python'``.  Passing ``theta_pts`` or ``n_legendre``
                with ``backend='fortran'`` raises ``NotImplementedError``.

        n_legendre : int or None
            Number of Gauss-Legendre quadrature points to use for the
            Legendre expansion of F11.  If supplied, ``pmom`` is populated
            with shape ``(N_wn, n_legendre)``, following the DISORT convention:
            ``P(cosθ) = Σ_l (2l+1) · pmom[l] · P_l(cosθ)``, ``pmom[0] = 1``.
            The quadrature points are chosen internally; the user does not need
            to supply angles.  May be requested alongside ``theta_pts``.

        Returns
        -------
        AerosolXSC
            Returns ``self`` to allow chaining:
            ``xsc.compute().save('output.npz')``.

        Raises
        ------
        ImportError
            If ``backend='fortran'`` and the compiled extension is not available.
        NotImplementedError
            If ``theta_pts`` or ``n_legendre`` is supplied with
            ``backend='fortran'``.
        ValueError
            If ``backend`` is not ``'python'`` or ``'fortran'``.
        """
        if self.backend == 'python':
            self._compute_python(progress, theta_pts=theta_pts,
                                 n_legendre=n_legendre)
        elif self.backend == 'fortran':
            if theta_pts is not None or n_legendre is not None:
                raise NotImplementedError(
                    "Phase matrix and Legendre expansion are not supported by "
                    "the Fortran backend.  Use backend='python' when "
                    "theta_pts or n_legendre is supplied."
                )
            self._compute_fortran()
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. Choose 'python' or 'fortran'."
            )
        return self

    def _compute_python(
        self,
        progress:   bool | str = False,
        theta_pts:  np.ndarray | None = None,
        n_legendre: int | None = None,
    ) -> None:
        """Size-distribution integration via the pure-Python/NumPy kernel."""
        from .legendre import legendreExpand
        from .quadrature import quadrature

        prog_arg: bool | str = self.ri.substance or 'AerosolXSC' if progress is True else progress

        # --- Build the combined cosine array for a single mieSizeDist call ---
        # Gauss-Legendre quadrature points for Legendre expansion (on cosθ ∈ [-1, 1])
        qv_leg: np.ndarray | None = None
        qw_leg: np.ndarray | None = None
        theta_deg: np.ndarray | None = None
        mu_user:   np.ndarray | None = None

        if n_legendre is not None:
            qv_leg, qw_leg = quadrature('G', n_legendre)   # cosines on [-1, 1]

        if theta_pts is not None:
            theta_deg = np.asarray(theta_pts, dtype=np.float64)
            mu_user   = np.cos(np.deg2rad(theta_deg))

        # Merge angle arrays; remember split index so we can separate afterwards
        if qv_leg is not None and mu_user is not None:
            mu_all = np.concatenate([qv_leg, mu_user])
            split  = len(qv_leg)
        elif qv_leg is not None:
            mu_all = qv_leg
            split  = len(qv_leg)
        elif mu_user is not None:
            mu_all = mu_user
            split  = 0
        else:
            mu_all = None
            split  = 0

        res = mieSizeDist(
            self.dist,
            None,        # params not needed — dist object supplied
            None,        # Nd embedded in dist
            self.wn,
            self.ri.Cm,
            angles=mu_all,
            n_pts=self.n_pts,
            progress=prog_arg,
        )

        G_avg = float(res.Gavg[0])
        V_avg = float(res.Vavg[0])
        N_avg = float(self.dist.N)

        Q_ext = res.Bext / G_avg
        Q_sca = res.Bsca / G_avg
        Q_abs = Q_ext - Q_sca
        C_ext = res.Bext
        C_sca = res.Bsca
        C_abs = C_ext - C_sca
        B_ext = res.Bext
        B_sca = res.Bsca
        B_abs = B_ext - B_sca
        MEC   = B_ext / (self.density * V_avg)
        MSC   = B_sca / (self.density * V_avg)
        MAC   = MEC - MSC

        # SPM from mieSizeDist: shape (4, N_angles_total, N_wn)
        phase_matrix: np.ndarray | None = None
        pmom:         np.ndarray | None = None

        if res.SPM is not None:
            # Split SPM back into Legendre and user-angle portions
            if n_legendre is not None:
                spm_leg  = res.SPM[:, :split, :]        # (4, n_leg, N_wn)
                spm_user = res.SPM[:, split:, :] if mu_user is not None else None
            else:
                spm_leg  = None
                spm_user = res.SPM

            # Legendre expansion of F11 → pmom
            if spm_leg is not None:
                n_wn  = len(self.wn)
                pmom  = np.zeros((n_wn, n_legendre))
                l_idx = np.arange(n_legendre, dtype=np.float64)
                for i in range(n_wn):
                    f11_i      = spm_leg[0, :, i]   # F11 at Gauss points, wn i
                    lc, _      = legendreExpand(f11_i, qv_leg, qw_leg)
                    # DISORT convention: pmom[l] = lc[l] / (2l+1)
                    pmom[i, :] = lc[:n_legendre] / (2.0 * l_idx + 1.0)

            # Phase matrix at user angles: transpose to (N_wn, N_theta, 4)
            if spm_user is not None:
                phase_matrix = np.moveaxis(spm_user, [0, 1, 2], [2, 1, 0])

        self.result = _AerosolResult(
            wavenumber   = self.wn,
            Q_ext = Q_ext,  Q_sca = Q_sca,  Q_abs = Q_abs,
            C_ext = C_ext,  C_sca = C_sca,  C_abs = C_abs,
            B_ext = B_ext,  B_sca = B_sca,  B_abs = B_abs,
            MEC   = MEC,    MSC   = MSC,    MAC   = MAC,
            g     = res.g,
            N_avg = N_avg,  V_avg = V_avg,  G_avg = G_avg,
            phase_matrix = phase_matrix,
            theta        = theta_deg,
            pmom         = pmom,
        )

    def _compute_fortran(self) -> None:
        """Size-distribution integration via the compiled Fortran kernel."""
        try:
            from .fortran._mie_kernel_f90 import mie_batch as _mie_batch_f90
        except ImportError as exc:
            raise ImportError(
                "Fortran backend not available. "
                "Build the extension with: python setup.py build_ext --inplace"
            ) from exc

        from .quadrature import quadrature, shiftQuadrature

        Rl, Ru        = self.dist.bounds()
        mu_01, wt_01  = quadrature('T', self.n_pts)
        R_nodes, wts  = shiftQuadrature(mu_01, wt_01, Rl, Ru)
        pdf_vals      = self.dist.pdf(R_nodes)

        # Distribution integrals (independent of wavenumber)
        V_avg = float(np.dot(wts, (4.0 / 3.0) * np.pi * R_nodes ** 3 * pdf_vals))
        G_avg = float(np.dot(wts, np.pi * R_nodes ** 2 * pdf_vals))
        N_avg = float(self.dist.N)

        n_wn  = len(self.wn)
        B_ext = np.empty(n_wn)
        B_sca = np.empty(n_wn)
        g_arr = np.empty(n_wn)

        x_base = 2.0 * np.pi * R_nodes   # [cm] — multiply by wavenumber [cm⁻¹]
        w_ext  = wts * np.pi * R_nodes ** 2 * pdf_vals
        w_sca  = w_ext                    # same quadrature weights for sca

        for j, (nu, m) in enumerate(zip(self.wn, self.ri.Cm)):
            x_arr        = np.ascontiguousarray(x_base * nu)
            qext, qsca, g_node = _mie_batch_f90(x_arr, float(m.real), float(-m.imag))
            B_ext[j] = np.dot(w_ext, qext)
            B_sca[j] = np.dot(w_sca, qsca)
            # Area-weighted asymmetry parameter
            g_arr[j] = (np.dot(w_sca, qsca * g_node) / B_sca[j]
                        if B_sca[j] > 0.0 else 0.0)

        B_abs = B_ext - B_sca
        Q_ext = B_ext / G_avg
        Q_sca = B_sca / G_avg
        Q_abs = Q_ext - Q_sca
        C_ext = B_ext
        C_sca = B_sca
        C_abs = B_abs
        MEC   = B_ext / (self.density * V_avg)
        MSC   = B_sca / (self.density * V_avg)
        MAC   = MEC - MSC

        self.result = _AerosolResult(
            wavenumber = self.wn,
            Q_ext = Q_ext,  Q_sca = Q_sca,  Q_abs = Q_abs,
            C_ext = C_ext,  C_sca = C_sca,  C_abs = C_abs,
            B_ext = B_ext,  B_sca = B_sca,  B_abs = B_abs,
            MEC   = MEC,    MSC   = MSC,    MAC   = MAC,
            g     = g_arr,
            N_avg = N_avg,  V_avg = V_avg,  G_avg = G_avg,
        )

    # ------------------------------------------------------------------
    # Convenience accessors (delegate to result)
    # ------------------------------------------------------------------

    def _require_result(self, name: str) -> _AerosolResult:
        if self.result is None:
            raise RuntimeError(
                f"'{name}' is not available before .compute() is called."
            )
        return self.result

    @property
    def MEC(self) -> np.ndarray:
        """Mass extinction coefficient [cm²/g]."""
        return self._require_result('MEC').MEC

    @property
    def MSC(self) -> np.ndarray:
        """Mass scattering coefficient [cm²/g]."""
        return self._require_result('MSC').MSC

    @property
    def MAC(self) -> np.ndarray:
        """Mass absorption coefficient [cm²/g]."""
        return self._require_result('MAC').MAC

    @property
    def Q_ext(self) -> np.ndarray:
        """Extinction efficiency factor."""
        return self._require_result('Q_ext').Q_ext

    @property
    def Q_sca(self) -> np.ndarray:
        """Scattering efficiency factor."""
        return self._require_result('Q_sca').Q_sca

    @property
    def Q_abs(self) -> np.ndarray:
        """Absorption efficiency factor."""
        return self._require_result('Q_abs').Q_abs

    @property
    def g(self) -> np.ndarray:
        """Asymmetry parameter."""
        return self._require_result('g').g

    @property
    def phase_matrix(self) -> np.ndarray:
        """
        Size-integrated scattering phase matrix, shape ``(N_wn, N_theta, 4)``.

        Columns are [F11, F33, F12, F34], normalised to the RT convention
        ``(1/2) * integral(F11, d(cosθ)) = 1``.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called, or if it was called
            without supplying ``theta_pts``.
        """
        r = self._require_result('phase_matrix')
        if r.phase_matrix is None:
            raise RuntimeError(
                "'phase_matrix' is not available.  Call compute(theta_pts=...) "
                "with an array of scattering angles."
            )
        return r.phase_matrix

    @property
    def theta(self) -> np.ndarray:
        """Scattering angles [degrees] used for the phase matrix computation."""
        r = self._require_result('theta')
        if r.theta is None:
            raise RuntimeError(
                "'theta' is not available.  Call compute(theta_pts=...) "
                "with an array of scattering angles."
            )
        return r.theta

    @property
    def pmom(self) -> np.ndarray:
        """
        Legendre expansion coefficients of F11, shape ``(N_wn, n_legendre)``.

        Follows the DISORT ``pmom`` convention:
        ``P(cosθ) = Σ_l (2l+1) · pmom[l] · P_l(cosθ)``, ``pmom[0] = 1``.

        Raises
        ------
        RuntimeError
            If :meth:`compute` was not called with ``n_legendre`` supplied.
        """
        r = self._require_result('pmom')
        if r.pmom is None:
            raise RuntimeError(
                "'pmom' is not available.  Call compute(n_legendre=N) "
                "with the desired number of Legendre terms."
            )
        return r.pmom

    # ------------------------------------------------------------------
    # Upsampling
    # ------------------------------------------------------------------

    def upsample(self, wn_highres: np.ndarray) -> '_AerosolXSCUpsampled':
        """
        Linearly interpolate all spectral outputs onto a higher-resolution grid.

        Parameters
        ----------
        wn_highres : np.ndarray
            Target wavenumber grid [cm⁻¹].  Must lie within the computed range.

        Returns
        -------
        _AerosolXSCUpsampled
            A lightweight dataclass with all spectral fields interpolated onto
            ``wn_highres``.  The original ``AerosolXSC`` object is unchanged.
        """
        r = self._require_result('upsample')
        wn_out = np.asarray(wn_highres, dtype=np.float64)

        def _interp(arr: np.ndarray) -> np.ndarray:
            return np.interp(wn_out, r.wavenumber, arr)

        return _AerosolXSCUpsampled(
            wavenumber = wn_out,
            Q_ext = _interp(r.Q_ext),  Q_sca = _interp(r.Q_sca),  Q_abs = _interp(r.Q_abs),
            C_ext = _interp(r.C_ext),  C_sca = _interp(r.C_sca),  C_abs = _interp(r.C_abs),
            B_ext = _interp(r.B_ext),  B_sca = _interp(r.B_sca),  B_abs = _interp(r.B_abs),
            MEC   = _interp(r.MEC),    MSC   = _interp(r.MSC),    MAC   = _interp(r.MAC),
            g     = _interp(r.g),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the computed results to disk.

        Parameters
        ----------
        path : str or Path
            Output file path.  Extension determines format:

            ``.npz``  — NumPy compressed archive (default/recommended).
            ``.nc``   — NetCDF4 (requires ``netCDF4`` package; useful for LUTs).

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called.
        ValueError
            If the file extension is not ``.npz`` or ``.nc``.
        """
        r = self._require_result('save')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == '.npz':
            self._save_npz(path, r)
        elif path.suffix == '.nc':
            self._save_nc(path, r)
        else:
            raise ValueError(
                f"Unsupported output format '{path.suffix}'. "
                "Use '.npz' or '.nc'."
            )

    def _save_npz(self, path: Path, r: _AerosolResult) -> None:
        dist_params = self._dist_params_dict()
        arrays: dict = dict(
            wavenumber = r.wavenumber,
            Q_ext = r.Q_ext,  Q_sca = r.Q_sca,  Q_abs = r.Q_abs,
            C_ext = r.C_ext,  C_sca = r.C_sca,  C_abs = r.C_abs,
            B_ext = r.B_ext,  B_sca = r.B_sca,  B_abs = r.B_abs,
            MEC   = r.MEC,    MSC   = r.MSC,    MAC   = r.MAC,
            g     = r.g,
            # Scalar metadata stored as 0-d arrays
            density        = np.float64(self.density),
            N_avg          = np.float64(r.N_avg),
            V_avg          = np.float64(r.V_avg),
            G_avg          = np.float64(r.G_avg),
            ri_filepath    = np.bytes_(self.ri.filepath),
            ri_substance   = np.bytes_(self.ri.substance),
            dist_type      = np.bytes_(type(self.dist).__name__),
            **{f'dist_{k}': np.float64(v) for k, v in dist_params.items()},
        )
        if r.phase_matrix is not None:
            arrays['phase_matrix'] = r.phase_matrix
            arrays['theta']        = r.theta
        if r.pmom is not None:
            arrays['pmom'] = r.pmom
        np.savez_compressed(path, **arrays)
        logger.info("Saved → %s", path)

    def _save_nc(self, path: Path, r: _AerosolResult) -> None:
        try:
            import netCDF4 as nc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "NetCDF4 output requires the 'netCDF4' package: "
                "pip install netCDF4"
            ) from exc

        dist_params = self._dist_params_dict()
        ds = nc.Dataset(str(path), 'w', format='NETCDF4')
        try:
            ds.createDimension('wavenumber', len(r.wavenumber))
            wn_var = ds.createVariable('wavenumber', 'f8', ('wavenumber',))
            wn_var[:] = r.wavenumber
            wn_var.units = 'cm-1'
            wn_var.long_name = 'wavenumber'

            for name, arr in [
                ('Q_ext', r.Q_ext), ('Q_sca', r.Q_sca), ('Q_abs', r.Q_abs),
                ('C_ext', r.C_ext), ('C_sca', r.C_sca), ('C_abs', r.C_abs),
                ('B_ext', r.B_ext), ('B_sca', r.B_sca), ('B_abs', r.B_abs),
                ('MEC',   r.MEC),   ('MSC',   r.MSC),   ('MAC',   r.MAC),
                ('g',     r.g),
            ]:
                v = ds.createVariable(name, 'f8', ('wavenumber',))
                v[:] = arr

            if r.phase_matrix is not None:
                ds.createDimension('theta', r.phase_matrix.shape[1])
                ds.createDimension('spm_component', 4)
                theta_var = ds.createVariable('theta', 'f8', ('theta',))
                theta_var[:] = r.theta
                theta_var.units = 'degrees'
                theta_var.long_name = 'scattering angle'
                spm_var = ds.createVariable(
                    'phase_matrix', 'f8', ('wavenumber', 'theta', 'spm_component'))
                spm_var[:] = r.phase_matrix
                spm_var.long_name = 'phase matrix [F11, F33, F12, F34]'
                spm_var.normalization = '(1/2) * integral(F11, d(cos_theta)) = 1'

            # Global attributes
            ds.density    = float(self.density)
            ds.N_avg      = float(r.N_avg)
            ds.V_avg      = float(r.V_avg)
            ds.G_avg      = float(r.G_avg)
            ds.ri_filepath = self.ri.filepath
            ds.ri_substance = self.ri.substance
            ds.dist_type   = type(self.dist).__name__
            for k, v in dist_params.items():
                setattr(ds, f'dist_{k}', float(v))
        finally:
            ds.close()
        logger.info("Saved → %s", path)

    def _dist_params_dict(self) -> dict[str, float]:
        """Extract size distribution parameters as a flat dict."""
        if isinstance(self.dist, LogNormalDist):
            return {'r0': self.dist.r0, 's': self.dist.s, 'N': self.dist.N}
        elif isinstance(self.dist, ModifiedGammaDist):
            return {
                'alpha': self.dist.alpha,
                'rc':    self.dist.rc,
                'gamma': self.dist.gamma,
                'N':     self.dist.N,
            }
        return {}

    @classmethod
    def load(cls, path: str | Path) -> 'AerosolXSC':
        """
        Load a previously saved :class:`AerosolXSC` from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.npz`` file previously written by :meth:`save`.

        Returns
        -------
        AerosolXSC
            Reconstructed instance with :attr:`result` populated.
            The refractive index is re-read from the original file path
            if it is still accessible; otherwise ``ri`` is set to *None*
            and spectral data are read directly from the archive.

        Notes
        -----
        NetCDF4 loading is not yet implemented.
        """
        path = Path(path)
        if path.suffix == '.nc':
            raise NotImplementedError("NetCDF4 loading is not yet implemented.")

        data = np.load(path, allow_pickle=False)

        wn     = data['wavenumber']
        density = float(data['density'])

        # Attempt to reload the refractive index from its original path
        ri_filepath = data['ri_filepath'].tobytes().decode()
        try:
            ri = readRI(ri_filepath, wavenumber=wn, out_of_range='clip')
        except FileNotFoundError:
            logger.warning(
                "Original .ri file not found at '%s'. "
                "RefrIndex will be reconstructed without metadata.",
                ri_filepath,
            )
            ri = None  # type: ignore[assignment]

        # Reconstruct size distribution from saved parameters
        dist_type = data['dist_type'].tobytes().decode()
        dist: SizeDistribution
        if dist_type == 'LogNormalDist':
            dist = LogNormalDist(
                r0 = float(data['dist_r0']),
                s  = float(data['dist_s']),
                n  = float(data['dist_N']),
            )
        elif dist_type == 'ModifiedGammaDist':
            dist = ModifiedGammaDist(
                alpha = float(data['dist_alpha']),
                rc    = float(data['dist_rc']),
                gamma = float(data['dist_gamma']),
                N     = float(data['dist_N']),
            )
        else:
            raise ValueError(f"Unknown distribution type in archive: '{dist_type}'")

        obj = cls.__new__(cls)
        obj.wn      = wn
        obj.dist    = dist
        obj.density = density
        obj.n_pts   = 100  # not stored; default
        obj.ri      = ri   # may be None if file not found

        phase_matrix = data['phase_matrix'] if 'phase_matrix' in data else None
        theta        = data['theta']        if 'theta'        in data else None
        pmom         = data['pmom']         if 'pmom'         in data else None

        obj.result = _AerosolResult(
            wavenumber   = wn,
            Q_ext = data['Q_ext'],  Q_sca = data['Q_sca'],  Q_abs = data['Q_abs'],
            C_ext = data['C_ext'],  C_sca = data['C_sca'],  C_abs = data['C_abs'],
            B_ext = data['B_ext'],  B_sca = data['B_sca'],  B_abs = data['B_abs'],
            MEC   = data['MEC'],    MSC   = data['MSC'],    MAC   = data['MAC'],
            g     = data['g'],
            N_avg = float(data['N_avg']),
            V_avg = float(data['V_avg']),
            G_avg = float(data['G_avg']),
            phase_matrix = phase_matrix,
            theta        = theta,
            pmom         = pmom,
        )
        return obj

    # ------------------------------------------------------------------
    # RFM export
    # ------------------------------------------------------------------

    def to_rfm_xsc(self, path: str | Path, **kwargs) -> None:
        """
        Export results to a HITRAN-format ``.xsc`` file for RFM.

        Thin wrapper around :func:`~mieEODG.rfm.write_rfm_xsc`.
        See that function for full parameter documentation.

        Parameters
        ----------
        path : str or Path
            Output file path (use ``.xsc`` extension).
        **kwargs
            Passed directly to :func:`~mieEODG.rfm.write_rfm_xsc`:
            ``convention``, ``concentration``, ``quantity``, ``T_K``,
            ``P_torr``.
        """
        from .rfm import write_rfm_xsc
        write_rfm_xsc(self, path, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        substance = getattr(self.ri, 'substance', '') or 'unknown'
        dist_repr = repr(self.dist)
        status = 'computed' if self.result is not None else 'not computed'
        return (
            f"AerosolXSC(\n"
            f"  substance = '{substance}',\n"
            f"  dist      = {dist_repr},\n"
            f"  wn        = [{self.wn[0]:.1f}, {self.wn[-1]:.1f}] cm⁻¹ "
            f"({len(self.wn)} pts),\n"
            f"  density   = {self.density} g/cm³,\n"
            f"  status    = {status}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Upsampled result container
# ---------------------------------------------------------------------------

@dataclass
class _AerosolXSCUpsampled:
    """Interpolated spectral outputs from :meth:`AerosolXSC.upsample`."""
    wavenumber: np.ndarray
    Q_ext: np.ndarray;  Q_sca: np.ndarray;  Q_abs: np.ndarray
    C_ext: np.ndarray;  C_sca: np.ndarray;  C_abs: np.ndarray
    B_ext: np.ndarray;  B_sca: np.ndarray;  B_abs: np.ndarray
    MEC:   np.ndarray;  MSC:   np.ndarray;  MAC:   np.ndarray
    g:     np.ndarray
