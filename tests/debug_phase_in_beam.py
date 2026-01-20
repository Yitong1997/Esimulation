"""调试光束内的相位梯度

验证使用振幅掩模后，光束内的相位梯度是否正确
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata


def analyze_phase_in_beam():
    """分析光束内的相位梯度"""
    print("=" * 70)
    print("分析光束内的相位梯度")
    print("=" * 70)
    
    wavelength_um = 0.633
    wavelength_m = wavelength_um * 1e-6
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    # 初始化 PROPER 波前
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 512
    beam_ratio = 0.25
    
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取 PROPER 参数
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    print(f"\nPROPER 参数:")
    print(f"  网格大小: {n}x{n}")
    print(f"  采样间隔: {sampling_mm:.4f} mm/pixel")
    print(f"  网格范围: ±{sampling_mm * n / 2:.2f} mm")
    
    # 获取振幅掩模
    amp = proper.prop_get_amplitude(wfo)
    amp_threshold = 0.01 * np.max(amp)
    beam_mask = amp > amp_threshold
    
    print(f"  光束内像素数: {np.sum(beam_mask)}")
    
    # 计算光束范围
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    
    beam_x = X_mm[beam_mask]
    beam_y = Y_mm[beam_mask]
    print(f"  光束 X 范围: [{np.min(beam_x):.2f}, {np.max(beam_x):.2f}] mm")
    print(f"  光束 Y 范围: [{np.min(beam_y):.2f}, {np.max(beam_y):.2f}] mm")
    
    # 创建采样光线（密集采样）
    n_side = 128
    half_size = 7.5  # mm (1.5 * w0)
    x = np.linspace(-half_size, half_size, n_side)
    y = np.linspace(-half_size, half_size, n_side)
    ray_X, ray_Y = np.meshgrid(x, y)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    print(f"\n采样参数:")
    print(f"  采样范围: ±{half_size} mm")
    print(f"  采样点数: {n_side}x{n_side}")
    
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
    
    # 计算差分 OPD
    center_idx = n_rays // 2
    opd_tilted_aligned = opd_tilted - opd_tilted[center_idx]
    opd_no_tilt_aligned = opd_no_tilt - opd_no_tilt[center_idx]
    diff_opd = opd_tilted_aligned - opd_no_tilt_aligned
    
    valid_both = valid_tilted & valid_no_tilt
    
    # 去除倾斜
    valid_x = ray_x[valid_both]
    valid_y = ray_y[valid_both]
    valid_diff = diff_opd[valid_both]
    
    max_r = max(np.max(np.abs(valid_x)), np.max(np.abs(valid_y)))
    norm_x = valid_x / max_r
    norm_y = valid_y / max_r
    
    A = np.column_stack([np.ones_like(norm_x), norm_x, norm_y])
    coeffs, _, _, _ = np.linalg.lstsq(A, valid_diff, rcond=None)
    
    tilt_component = coeffs[0] + coeffs[1] * (ray_x / max_r) + coeffs[2] * (ray_y / max_r)
    aberration_waves = diff_opd - tilt_component
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
    print(f"\n像差统计:")
    print(f"  像差 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
    print(f"  像差 PV: {np.max(aberration_waves[valid_both])-np.min(aberration_waves[valid_both]):.4f} waves")
    
    # 插值到 PROPER 网格
    valid_x_interp = ray_x[valid_both]
    valid_y_interp = ray_y[valid_both]
    valid_phase_interp = aberration_phase[valid_both]
    
    points = np.column_stack([valid_x_interp, valid_y_interp])
    phase_grid = griddata(
        points,
        valid_phase_interp,
        (X_mm, Y_mm),
        method='cubic',
        fill_value=np.nan,
    )
    
    # 应用振幅掩模
    phase_grid_masked = np.where(beam_mask & ~np.isnan(phase_grid), phase_grid, 0.0)
    
    # 计算光束内的相位梯度
    phase_in_beam = phase_grid.copy()
    phase_in_beam[~beam_mask] = np.nan
    
    grad_x = np.diff(phase_in_beam, axis=1)
    grad_y = np.diff(phase_in_beam, axis=0)
    
    # 只考虑光束内的梯度
    max_grad_x = np.nanmax(np.abs(grad_x))
    max_grad_y = np.nanmax(np.abs(grad_y))
    max_grad = max(max_grad_x, max_grad_y)
    
    print(f"\n光束内相位梯度:")
    print(f"  最大 X 梯度: {max_grad_x:.2f} rad/pixel")
    print(f"  最大 Y 梯度: {max_grad_y:.2f} rad/pixel")
    print(f"  是否超过 π: {'是 ⚠️' if max_grad > np.pi else '否 ✓'}")
    
    # 计算应用掩模后的相位梯度
    grad_x_masked = np.diff(phase_grid_masked, axis=1)
    grad_y_masked = np.diff(phase_grid_masked, axis=0)
    
    max_grad_masked = max(np.nanmax(np.abs(grad_x_masked)), np.nanmax(np.abs(grad_y_masked)))
    
    print(f"\n应用掩模后的相位梯度:")
    print(f"  最大梯度: {max_grad_masked:.2f} rad/pixel")
    print(f"  是否超过 π: {'是 ⚠️' if max_grad_masked > np.pi else '否 ✓'}")
    
    # 测量应用相位后的 WFE
    wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    phase_field = np.exp(1j * phase_grid_masked)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
    
    amp_out = proper.prop_get_amplitude(wfo_test)
    phase_out = proper.prop_get_phase(wfo_test)
    mask_out = amp_out > 0.01 * np.max(amp_out)
    
    valid_phase_out = phase_out[mask_out]
    wfe_rms = np.std(valid_phase_out - np.mean(valid_phase_out)) / (2 * np.pi)
    
    print(f"\n测量 WFE RMS: {wfe_rms:.4f} waves")
    print(f"预期 WFE RMS: {np.std(aberration_waves[valid_both]):.4f} waves")


if __name__ == "__main__":
    analyze_phase_in_beam()
