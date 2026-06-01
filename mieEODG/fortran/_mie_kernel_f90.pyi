"""
Type stub for the f2py-compiled Fortran 90 Mie kernel extension module.

This file exists solely for static analysis (PyCharm, mypy, pyright).
The actual implementation is in mie_kernel.f90, compiled to
_mie_kernel_f90.cpython-*.so via numpy.f2py.
"""

import numpy as np
from numpy.typing import NDArray


def mie_batch(
    x_arr: NDArray[np.float64],
    m_re: float,
    m_im: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Mie efficiency factors for a batch of size parameters.

    Parameters
    ----------
    x_arr : ndarray, shape (N,)
        Size parameters  x = 2π r ν  [dimensionless].
    m_re : float
        Real part of the complex refractive index n.
    m_im : float
        Imaginary part k ≥ 0 (absorption coefficient, positive convention).
        The subroutine forms  m = m_re − i·m_im  internally.

    Returns
    -------
    qext : ndarray, shape (N,)
        Extinction efficiency Q_ext.
    qsca : ndarray, shape (N,)
        Scattering efficiency Q_sca.
    g : ndarray, shape (N,)
        Asymmetry parameter ⟨cos θ⟩.
    """
    ...
