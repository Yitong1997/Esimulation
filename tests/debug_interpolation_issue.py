"""调试插值问题

问题：像差计算正确，但插值到 PROPER 网格后结果不对
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata
import proper


def debug_interpolation():
    """调试插值过程"""
    print("=" * 70)
    print("调试插值过程")
    print("=" * 70)
    
    wavelength_um = 0.633
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    # 模拟 PROPER 参数
    grid_size = 512
    beam_ratio = 0.25
    w0 = 5.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3
    
    # 初始化 PROPER 波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_um * 1e-6, grid_size, beam_ratio)
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    print(f"PROPER 网格大小: {n}")
    print(f"PROPER 采样间隔: {sampling_mm:.4f} mm/pixel")
    print(f"PROPER 网格范围: [{-sampling_mm*n/2:.2f}, {sampling_mm*n/2:.2f}] mm")
    
    # 采样范围（is_fold=False）
    half_size_mm = w0 * 2  # 10 mm
    print(f"\n采样范围: half_size_mm = {half_size_mm:.2f} mm")
    
    # 创建采样光线
    n_rays_1d = 10
    ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    print(f"采样光线: {n_rays_1d}x{n_rays_1d} = {n_rays}")
    print(f"光线范围: [{ray_x.min():.2f}, {ray_x.max():.2f}] mm")
    
    # 带倾斜的抛物面镜
    surface_tilted = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        conic=-1.0,
        tilt_x=tilt_rad,
        tilt_y=0.0,
    )
    
    raytracer_tilted = ElementRaytracer(
        surfaces=[surface_tilted],
        wavelength=wavelength_um,
    )
    
    rays_in = RealRays(
        x=ray_x.copy(),
        y=ray_y.copy(),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    rays_out_tilted = raytracer_tilted.trace(rays_in)
    opd_tilted = raytracer_tilted.get_relative_opd_waves()
    valid_tilted = raytracer_tilted.get_valid_ray_mask()
    
    # 不带倾斜的抛物面镜
    surface_no_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        conic=-1.0,
        tilt_x=0.0,
        tilt_y=0.0,
    )
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    
    rays_in_ref = RealRays(
        x=ray_x.copy(),
        y=ray_y.copy(),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    rays_out_no_tilt = raytracer_no_tilt.trace(rays_in_ref)
    opd_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
    valid_no_tilt = raytracer_no_tilt.get_valid_ray_mask()
    
    # 差分
    center_idx = n_rays // 2
    opd_tilted_aligned = opd_tilted - opd_tilted[center_idx]
    opd_no_tilt_aligned = opd_no_tilt - opd_no_tilt[center_idx]
    diff_opd = opd_tilted_aligned - opd_no_tilt_aligned
    
    valid_mask = valid_tilted & valid_no_tilt
    
    # 去除倾斜
    valid_x = ray_x[valid_mask]
    valid_y = ray_y[valid_mask]
    valid_diff = diff_opd[valid_mask]
    
    max_r = max(np.max(np.abs(valid_x)), np.max(np.abs(valid_y)))
    norm_x = valid_x / max_r
    norm_y = valid_y / max_r
    
    A = np.column_stack([np.ones_like(norm_x), norm_x, norm_y])
    coeffs, _, _, _ = np.linalg.lstsq(A, valid_diff, rcond=None)
    
    tilt_component = coeffs[0] + coeffs[1] * (ray_x / max_r) + coeffs[2] * (ray_y / max_r)
    aberration_waves = diff_opd - tilt_component
    
    print(f"\n去除倾斜后的像差:")
    print(f"  RMS = {np.std(aberration_waves[valid_mask]):.4f} waves")
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
    valid_phase = aberration_phase[valid_mask]
    
    print(f"\n像差相位:")
    print(f"  min = {np.min(valid_phase):.4f} rad")
    print(f"  max = {np.max(valid_phase):.4f} rad")
    
    # 插值到 PROPER 网格
    # 关键问题：插值范围应该是采样范围，而不是 PROPER 网格范围
    coords_mm = np.linspace(-half_size_mm, half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    
    print(f"\n插值网格范围: [{coords_mm.min():.2f}, {coords_mm.max():.2f}] mm")
    
    points = np.column_stack([valid_x, valid_y])
    phase_grid = griddata(
        points,
        valid_phase,
        (X_mm, Y_mm),
        method='cubic',
        fill_value=0.0,
    )
    
    phase_grid = np.nan_to_num(phase_grid, nan=0.0)
    
    print(f"\n插值后的相位网格:")
    print(f"  min = {np.nanmin(phase_grid):.4f} rad")
    print(f"  max = {np.nanmax(phase_grid):.4f} rad")
    
    # 计算 WFE RMS（在光束区域内）
    R = np.sqrt(X_mm**2 + Y_mm**2)
    beam_mask = R <= w0 * 2  # 2 * w0
    
    valid_phase_grid = phase_grid[beam_mask]
    wfe_rms = np.std(valid_phase_grid - np.mean(valid_phase_grid)) / (2 * np.pi)
    
    print(f"\n在光束区域内的 WFE RMS: {wfe_rms:.4f} waves")
    
    # 问题分析：插值网格范围是 [-10, 10] mm，但 PROPER 网格范围是 [-40, 40] mm
    # 这意味着相位只被应用到了网格的中心部分
    # 但是 PROPER 的高斯光束也主要集中在中心，所以这应该是正确的
    
    # 让我检查一下 PROPER 波前的振幅分布
    amp = proper.prop_get_amplitude(wfo)
    amp_centered = proper.prop_shift_center(amp)
    
    # 计算光束半径
    total_power = np.sum(amp_centered**2)
    threshold = 0.01 * np.max(amp_centered)
    beam_pixels = amp_centered > threshold
    
    print(f"\n光束分析:")
    print(f"  振幅最大值: {np.max(amp_centered):.4f}")
    print(f"  有效像素数: {np.sum(beam_pixels)}")
    
    # 计算光束在网格中的范围
    y_indices, x_indices = np.where(beam_pixels)
    if len(x_indices) > 0:
        x_min_px = np.min(x_indices)
        x_max_px = np.max(x_indices)
        y_min_px = np.min(y_indices)
        y_max_px = np.max(y_indices)
        
        x_min_mm = (x_min_px - n/2) * sampling_mm
        x_max_mm = (x_max_px - n/2) * sampling_mm
        y_min_mm = (y_min_px - n/2) * sampling_mm
        y_max_mm = (y_max_px - n/2) * sampling_mm
        
        print(f"  光束 X 范围: [{x_min_mm:.2f}, {x_max_mm:.2f}] mm")
        print(f"  光束 Y 范围: [{y_min_mm:.2f}, {y_max_mm:.2f}] mm")


if __name__ == "__main__":
    debug_interpolation()
