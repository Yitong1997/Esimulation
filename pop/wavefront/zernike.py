"""Zernike polynomial utilities for POP wavefront reconstruction."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index (1-based) to (n, m)."""
    if j < 1:
        raise ValueError("Noll index must be >= 1")
    n = 0
    while j > (n + 1) * (n + 2) // 2:
        n += 1
    j_in = int(j - n * (n + 1) // 2)  # 1-based within degree n
    m_list = list(range(-n, n + 1, 2))
    if j_in < 1 or j_in > len(m_list):
        raise ValueError(f"Invalid Noll index: {j}")
    return n, m_list[j_in - 1]


def radial_polynomial(n: int, m: int, rho: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the radial Zernike polynomial R_n^m(rho)."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho, dtype=float)
    radial = np.zeros_like(rho, dtype=float)
    max_k = (n - m) // 2
    for k in range(max_k + 1):
        num = math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        radial = radial + ((-1) ** k) * (num / den) * rho ** (n - 2 * k)
    return radial


def zernike_nm(
    n: int,
    m: int,
    rho: NDArray[np.floating],
    theta: NDArray[np.floating],
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Compute Zernike polynomial Z_n^m(rho, theta)."""
    radial = radial_polynomial(n, m, rho)
    if m > 0:
        z = radial * np.cos(m * theta)
    elif m < 0:
        z = radial * np.sin(abs(m) * theta)
    else:
        z = radial
    if normalize:
        norm = math.sqrt(2 * (n + 1)) if m != 0 else math.sqrt(n + 1)
        z = z * norm
    return z


def build_zernike_matrix(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    radius: float,
    n_terms: int,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Build Zernike design matrix for scattered points."""
    rho = np.sqrt(x**2 + y**2) / float(radius)
    theta = np.arctan2(y, x)
    z_mat = np.zeros((x.size, n_terms), dtype=float)
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        z_mat[:, j - 1] = zernike_nm(n, m, rho, theta, normalize=normalize)
    return z_mat


def fit_zernike_phase(
    phase: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    radius: float,
    n_terms: int,
    normalize: bool = True,
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Fit Zernike coefficients to scattered phase data."""
    rho = np.sqrt(x**2 + y**2) / float(radius)
    valid_mask = np.isfinite(phase) & np.isfinite(x) & np.isfinite(y) & (rho <= 1.0)
    if np.count_nonzero(valid_mask) < n_terms:
        raise ValueError("Not enough valid points for requested Zernike terms.")
    z_mat = build_zernike_matrix(
        x[valid_mask], y[valid_mask], radius, n_terms, normalize=normalize
    )
    coeffs, _, _, _ = np.linalg.lstsq(z_mat, phase[valid_mask], rcond=None)
    return coeffs, valid_mask


def evaluate_zernike_grid(
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    radius: float,
    coeffs: NDArray[np.floating],
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Evaluate Zernike phase on a grid.
    
    Note: Zernike polynomials are mathematically defined for rho <= 1.0,
    but we evaluate on the full grid to allow visualization of the fit
    outside the fitting region. The extrapolation may be unreliable far
    outside the unit circle.
    """
    rho = np.sqrt(x_grid**2 + y_grid**2) / float(radius)
    theta = np.arctan2(y_grid, x_grid)
    phase = np.zeros_like(x_grid, dtype=float)
    for j, coeff in enumerate(coeffs, start=1):
        n, m = noll_to_nm(j)
        phase = phase + coeff * zernike_nm(n, m, rho, theta, normalize=normalize)
    # 不再限制 rho <= 1.0，允许在整个网格上评估（包括外推区域）
    return phase
