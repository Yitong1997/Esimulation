"""Wavefront to rays sampling."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pop.core import GridSampling, OpticalAxisState, PilotBeamParams
from pop.coordinates.transforms import transform_rays_to_global


def _compute_effective_beam_radius(
    amplitude: NDArray[np.floating],
    grid_sampling: GridSampling,
    sampling_sigma: float,
) -> float:
    """计算有效光束半径 (mm)。

    使用强度加权的二阶矩 (Gaussian Moments) 估算光束 sigma，
    然后乘以 sampling_sigma 得到有效采样半径。
    当 Gaussian Moments 失败时，回退到基于振幅阈值（1e-5 * peak）的
    最大有效像素半径，与下游 mask 裁剪逻辑保持一致。
    结果被限制在物理网格范围 (0.99 * half_size) 内。

    Returns:
        有效光束半径 (mm)
    """
    intensity_grid = amplitude ** 2
    intensity_sum = np.sum(intensity_grid)

    sigma_max = 0.0
    if intensity_sum > 0:
        ny, nx = intensity_grid.shape
        y_indices, x_indices = np.indices((ny, nx))
        cy, cx = ny // 2, nx // 2
        x_grid = (x_indices - cx) * grid_sampling.sampling_mm
        y_grid = (y_indices - cy) * grid_sampling.sampling_mm

        # 质心
        x_centroid = np.sum(x_grid * intensity_grid) / intensity_sum
        y_centroid = np.sum(y_grid * intensity_grid) / intensity_sum

        # 二阶矩 (方差)
        sigma_x_sq = np.sum((x_grid - x_centroid) ** 2 * intensity_grid) / intensity_sum
        sigma_y_sq = np.sum((y_grid - y_centroid) ** 2 * intensity_grid) / intensity_sum

        sigma_max = np.sqrt(max(sigma_x_sq, sigma_y_sq))

    beam_radius_gauss = sampling_sigma * sigma_max

    # 物理网格限制
    half_size = grid_sampling.physical_size_mm / 2.0
    max_physical = 0.99 * half_size

    if beam_radius_gauss < 1e-9:
        # Gaussian Moments 失败时，回退到阈值法：
        # 在完整 amplitude grid 上找 > 1e-5 * peak 的像素的最大半径
        peak_amp = np.max(amplitude)
        if peak_amp > 0:
            threshold = 1e-5 * peak_amp
            ny, nx = amplitude.shape
            y_indices, x_indices = np.indices((ny, nx))
            cy, cx = ny // 2, nx // 2
            x_grid = (x_indices - cx) * grid_sampling.sampling_mm
            y_grid = (y_indices - cy) * grid_sampling.sampling_mm
            dist_sq_grid = x_grid ** 2 + y_grid ** 2
            thresh_mask = amplitude > threshold
            if np.any(thresh_mask):
                return min(float(np.sqrt(np.max(dist_sq_grid[thresh_mask]))), max_physical)
        return max_physical

    return min(beam_radius_gauss, max_physical)


def sample_rays_from_wavefront(
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    grid_sampling: GridSampling,
    entrance_axis: OpticalAxisState,
    pilot_beam_params: Optional[PilotBeamParams],
    num_rays: int = 1000,
    sampling_sigma: float = 6.0,
) -> Tuple["RealRays", "RealRays"]:

    # ... (skipping unchanged code) ...

    # beam_radius_gauss = 6.0 * sigma_max  # ~99.5% energy radius for Gaussian (w=2sigma, R=1.7w)


    from optiland.rays import RealRays

    n = grid_sampling.grid_size
    dx = grid_sampling.sampling_mm
    dy = grid_sampling.sampling_mm

    if pilot_beam_params is None:
        raise ValueError("pilot_beam_params is required for sampling")
    pilot_phase = pilot_beam_params.compute_phase_grid(
        grid_sampling.grid_size, grid_sampling.physical_size_mm
    )
    sampling_phase = phase - pilot_phase

    # --- 提前计算有效光束半径，用于确定采样 stride ---
    effective_radius_mm = _compute_effective_beam_radius(
        amplitude, grid_sampling, sampling_sigma
    )
    effective_radius_sq = effective_radius_mm ** 2

    # 用有效区域的像素跨度 n_eff 代替整个网格 n 来计算 stride，
    # 使 num_rays 对应 mask 后实际生效的光线数。
    # 圆形面积修正：正方形内接圆面积 = π/4 × 正方形面积，
    # 所以要在正方形中放 num_rays * 4/π 个点才能让圆内约有 num_rays 个点。
    effective_diameter_px = 2.0 * effective_radius_mm / dx
    num_rays_target = max(1, int(num_rays))
    n_eff = max(1.0, effective_diameter_px)
    stride = max(1, int(n_eff / np.sqrt(num_rays_target * 4.0 / np.pi)))

    coords = (np.arange(n) - n // 2) * dx
    center_idx = n // 2
    # 采样偏移量限制在有效半径对应的像素范围内
    max_offset_px = max(1, int(effective_radius_mm / dx))
    offsets = np.arange(0, min(max_offset_px + 1, n // 2), stride)
    valid_offsets = np.unique(np.concatenate([offsets, -offsets]))
    sample_indices = center_idx + valid_offsets
    # 确保索引不越界
    sample_indices = sample_indices[(sample_indices >= 0) & (sample_indices < n)]

    ix_grid, iy_grid = np.meshgrid(sample_indices, sample_indices)
    ix_flat = ix_grid.flatten()
    iy_flat = iy_grid.flatten()

    sampled_phase = sampling_phase[iy_flat, ix_flat]
    sampled_amp = amplitude[iy_flat, ix_flat]

    x_rays = coords[ix_flat]
    y_rays = coords[iy_flat]

    grad_y, grad_x = np.gradient(sampling_phase, dy, dx)
    wavelength_mm = pilot_beam_params.wavelength_um * 1e-3
    current_n = pilot_beam_params.current_refractive_index
    k = 2.0 * np.pi * current_n / wavelength_mm

    l_rays = grad_x[iy_flat, ix_flat] / k
    m_rays = grad_y[iy_flat, ix_flat] / k

    r_curv = pilot_beam_params.curvature_radius_mm
    if np.isinf(r_curv):
        l_pilot = 0.0
        m_pilot = 0.0
    else:
        l_pilot = x_rays / r_curv
        m_pilot = y_rays / r_curv
    l_rays += l_pilot
    m_rays += m_pilot

    # 方向余弦有效性
    sin_sq = l_rays**2 + m_rays**2

    # 距离平方 (用于圆形 mask)
    dist_sq = x_rays**2 + y_rays**2

    # Final mask: 在有效光束圆内 且 方向余弦有效
    valid_mask = (dist_sq <= effective_radius_sq) & (sin_sq < 1.0)

    # 始终保留中心光线
    center_ray_mask = dist_sq < 1e-10
    valid_mask = valid_mask | center_ray_mask

    if not np.any(valid_mask):
        raise ValueError("No valid rays after sampling")

    # 日志输出：方便调试采样效率
    n_before_mask = len(ix_flat)
    n_after_mask = int(np.count_nonzero(valid_mask))
    print(
        f"[POP][Sampler] 目标光线={num_rays_target}, "
        f"stride={stride}, 采样前={n_before_mask}, "
        f"mask后={n_after_mask}, "
        f"有效半径={effective_radius_mm:.2f}mm"
    )

    x_final = x_rays[valid_mask]
    y_final = y_rays[valid_mask]
    l_final = l_rays[valid_mask]
    m_final = m_rays[valid_mask]
    n_final = np.sqrt(1.0 - (l_final**2 + m_final**2))

    local_rays = RealRays(
        x=x_final,
        y=y_final,
        z=np.zeros_like(x_final),
        L=l_final,
        M=m_final,
        N=n_final,
        wavelength=pilot_beam_params.wavelength_um,
        intensity=sampled_amp[valid_mask] ** 2,
    )
    # Preserve local rays before globalizing to avoid mixing coordinate frames.
    # CoordinateSystem.globalize() mutates in place.
    import copy

    global_rays = copy.deepcopy(local_rays)
    global_rays = transform_rays_to_global(global_rays, entrance_axis)

    full_phase = phase[iy_flat, ix_flat][valid_mask]
    initial_opd_mm = full_phase * wavelength_mm / (2.0 * np.pi)
    global_rays.opd = initial_opd_mm
    global_rays.i = sampled_amp[valid_mask] ** 2

    return local_rays, global_rays
