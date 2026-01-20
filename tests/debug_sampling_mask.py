"""调试采样范围掩模的效果

验证使用采样范围掩模后的相位应用是否正确
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata


def analyze_sampling_mask():
    """分析采样范围掩模的效果"""
    print("=" * 70)
    print("分析采样范围掩模的效果")
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
    
    print(f"\n像差统计（采样点）:")
    print(f"  像差 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
    print(f"  像差 PV: {np.max(aberration_waves[valid_both])-np.min(aberration_waves[valid_both]):.4f} waves")
    
    # 插值到 PROPER 网格
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    
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
    
    # 创建采样范围掩模
    R_mm = np.sqrt(X_mm**2 + Y_mm**2)
    sampling_range_mask = R_mm <= half_size
    
    # 在采样范围外，相位设为 0
    phase_grid_masked = np.where(sampling_range_mask & ~np.isnan(phase_grid), phase_grid, 0.0)
    
    # 计算采样范围内的相位梯度
    phase_in_range = phase_grid.copy()
    phase_in_range[~sampling_range_mask] = np.nan
    
    grad_x = np.diff(phase_in_range, axis=1)
    grad_y = np.diff(phase_in_range, axis=0)
    
    max_grad = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
    
    print(f"\n采样范围内相位梯度:")
    print(f"  最大梯度: {max_grad:.2f} rad/pixel")
    print(f"  是否超过 π: {'是 ⚠️' if max_grad > np.pi else '否 ✓'}")
    
    # 测量应用相位后的 WFE
    wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    phase_field = np.exp(1j * phase_grid_masked)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
    
    amp_out = proper.prop_get_amplitude(wfo_test)
    phase_out = proper.prop_get_phase(wfo_test)
    
    # 使用采样范围掩模测量 WFE
    mask_out = sampling_range_mask & (amp_out > 0.01 * np.max(amp_out))
    
    valid_phase_out = phase_out[mask_out]
    wfe_rms = np.std(valid_phase_out - np.mean(valid_phase_out)) / (2 * np.pi)
    
    print(f"\n测量 WFE RMS（采样范围内）: {wfe_rms:.4f} waves")
    print(f"预期 WFE RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
    
    # 使用整个光束测量 WFE
    amp_threshold = 0.01 * np.max(amp_out)
    beam_mask = amp_out > amp_threshold
    
    valid_phase_beam = phase_out[beam_mask]
    wfe_rms_beam = np.std(valid_phase_beam - np.mean(valid_phase_beam)) / (2 * np.pi)
    
    print(f"测量 WFE RMS（整个光束）: {wfe_rms_beam:.4f} waves")


if __name__ == "__main__":
    analyze_sampling_mask()
