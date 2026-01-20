"""调试像差应用过程

问题：像差计算正确，但应用到 PROPER 波前时出现问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata


def debug_aberration_interpolation():
    """调试像差插值过程"""
    print("=" * 70)
    print("调试像差插值过程")
    print("=" * 70)
    
    wavelength_um = 0.633
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    # 模拟 _apply_element_hybrid 的参数
    grid_size = 512
    beam_ratio = 0.25
    beam_diameter_m = 4 * 5.0 * 1e-3  # 4 * w0
    sampling_m = beam_diameter_m / (grid_size * beam_ratio)
    sampling_mm = sampling_m * 1e3
    half_size_mm = sampling_mm * grid_size / 2
    
    print(f"网格大小: {grid_size}")
    print(f"采样间隔: {sampling_mm:.4f} mm/pixel")
    print(f"半尺寸: {half_size_mm:.2f} mm")
    
    # 创建采样光线（模拟 hybrid_num_rays=100）
    n_rays_1d = 10
    ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    print(f"光线数量: {n_rays}")
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
    
    print(f"\n差分 OPD:")
    print(f"  min={np.min(diff_opd[valid_mask]):.4f}, max={np.max(diff_opd[valid_mask]):.4f}")
    print(f"  RMS={np.std(diff_opd[valid_mask]):.4f} waves")
    
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
    print(f"  min={np.min(aberration_waves[valid_mask]):.4f}, max={np.max(aberration_waves[valid_mask]):.4f}")
    print(f"  RMS={np.std(aberration_waves[valid_mask]):.4f} waves")
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
    print(f"\n像差相位:")
    print(f"  min={np.min(aberration_phase[valid_mask]):.4f}, max={np.max(aberration_phase[valid_mask]):.4f} rad")
    
    # 插值到网格
    valid_phase = aberration_phase[valid_mask]
    
    coords_mm = np.linspace(-half_size_mm, half_size_mm, grid_size)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    
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
    print(f"  min={np.nanmin(phase_grid):.4f}, max={np.nanmax(phase_grid):.4f} rad")
    
    # 检查相位梯度
    grad_x = np.diff(phase_grid, axis=1)
    grad_y = np.diff(phase_grid, axis=0)
    max_grad = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
    
    print(f"  最大相位梯度: {max_grad:.4f} rad/pixel")
    print(f"  是否超过 π: {max_grad > np.pi}")
    
    # 计算插值后的 WFE RMS
    # 创建圆形掩模
    R = np.sqrt(X_mm**2 + Y_mm**2)
    beam_radius = 5.0 * 2  # 2 * w0
    mask = R <= beam_radius
    
    valid_phase_grid = phase_grid[mask]
    wfe_rms = np.std(valid_phase_grid - np.mean(valid_phase_grid)) / (2 * np.pi)
    
    print(f"\n插值后的 WFE RMS: {wfe_rms:.4f} waves")


if __name__ == "__main__":
    debug_aberration_interpolation()
