"""调试 WFE 测量问题

问题：
- 像差 RMS 在采样点处是 1.4114 waves
- 但测量的 WFE RMS 只有 0.2842 waves
- 相位梯度在采样范围内是 0.94 rad/pixel（< π），没有混叠

可能原因：
1. 插值后的相位分布与原始不同
2. 掩模区域选择问题
3. 相位测量方式问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def debug_wfe_measurement():
    """调试 WFE 测量问题"""
    print("=" * 70)
    print("调试 WFE 测量问题")
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
    print(f"  采样间隔: {2*half_size/(n_side-1):.4f} mm/sample")
    
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
    
    print(f"\n像差统计（采样点）:")
    print(f"  像差 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
    print(f"  像差 PV: {np.max(aberration_waves[valid_both])-np.min(aberration_waves[valid_both]):.4f} waves")
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
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
    
    # =====================================================================
    # 关键检查：插值后的相位 RMS
    # =====================================================================
    print("\n" + "=" * 70)
    print("插值后的相位分析")
    print("=" * 70)
    
    # 在采样范围内的相位
    phase_in_range = phase_grid[sampling_range_mask & ~np.isnan(phase_grid)]
    phase_rms_in_range = np.std(phase_in_range - np.mean(phase_in_range)) / (2 * np.pi)
    
    print(f"  采样范围内相位 RMS: {phase_rms_in_range:.4f} waves")
    print(f"  预期 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
    
    # 检查插值是否正确
    # 在采样点位置检查插值值
    sample_indices = np.random.choice(len(valid_x_interp), min(10, len(valid_x_interp)), replace=False)
    print("\n  采样点插值验证（随机 10 个点）:")
    for idx in sample_indices:
        x_sample = valid_x_interp[idx]
        y_sample = valid_y_interp[idx]
        phase_original = valid_phase_interp[idx]
        
        # 找到最近的网格点
        i = int((y_sample + proper_half_size_mm) / sampling_mm)
        j = int((x_sample + proper_half_size_mm) / sampling_mm)
        
        if 0 <= i < n and 0 <= j < n:
            phase_interp = phase_grid[i, j]
            if not np.isnan(phase_interp):
                print(f"    ({x_sample:.2f}, {y_sample:.2f}): 原始={phase_original:.4f}, 插值={phase_interp:.4f}")

    # =====================================================================
    # 应用相位到 PROPER 波前
    # =====================================================================
    print("\n" + "=" * 70)
    print("应用相位到 PROPER 波前")
    print("=" * 70)
    
    wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取初始相位
    phase_before = proper.prop_get_phase(wfo_test)
    
    # 应用相位
    phase_field = np.exp(1j * phase_grid_masked)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
    
    # 获取应用后的相位
    phase_after = proper.prop_get_phase(wfo_test)
    amp_after = proper.prop_get_amplitude(wfo_test)
    
    # 计算相位变化
    phase_change = phase_after - phase_before
    
    # 在采样范围内测量
    valid_change = phase_change[sampling_range_mask]
    change_rms = np.std(valid_change - np.mean(valid_change)) / (2 * np.pi)
    
    print(f"  相位变化 RMS（采样范围内）: {change_rms:.4f} waves")
    
    # =====================================================================
    # 关键发现：检查 phase_grid_masked 的实际值
    # =====================================================================
    print("\n" + "=" * 70)
    print("检查 phase_grid_masked 的实际值")
    print("=" * 70)
    
    # 在采样范围内的 phase_grid_masked
    masked_in_range = phase_grid_masked[sampling_range_mask]
    masked_rms = np.std(masked_in_range - np.mean(masked_in_range)) / (2 * np.pi)
    
    print(f"  phase_grid_masked RMS（采样范围内）: {masked_rms:.4f} waves")
    print(f"  phase_grid_masked 非零元素数: {np.sum(phase_grid_masked != 0)}")
    print(f"  采样范围内元素数: {np.sum(sampling_range_mask)}")
    
    # 检查 NaN 的数量
    nan_count = np.sum(np.isnan(phase_grid))
    nan_in_range = np.sum(np.isnan(phase_grid) & sampling_range_mask)
    print(f"  phase_grid 中 NaN 数量: {nan_count}")
    print(f"  采样范围内 NaN 数量: {nan_in_range}")
    
    # =====================================================================
    # 可视化
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始像差（采样点）
    ax = axes[0, 0]
    aberration_2d = aberration_waves.reshape(n_side, n_side)
    im = ax.imshow(aberration_2d, cmap='RdBu_r', origin='lower',
                   extent=[-half_size, half_size, -half_size, half_size])
    ax.set_title(f'原始像差（采样点）\nRMS={np.std(aberration_waves[valid_both]):.4f} waves')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax, label='waves')
    
    # 2. 插值后的相位（整个网格）
    ax = axes[0, 1]
    phase_waves = phase_grid / (2 * np.pi)
    im = ax.imshow(phase_waves, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm, 
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('插值后的相位（整个网格）')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='waves')
    
    # 3. 掩模后的相位
    ax = axes[0, 2]
    masked_waves = phase_grid_masked / (2 * np.pi)
    im = ax.imshow(masked_waves, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm, 
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title(f'掩模后的相位\nRMS={masked_rms:.4f} waves')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='waves')
    
    # 4. 相位变化
    ax = axes[1, 0]
    change_waves = phase_change / (2 * np.pi)
    im = ax.imshow(change_waves, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm, 
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title(f'相位变化\nRMS={change_rms:.4f} waves')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='waves')
    
    # 5. 采样范围掩模
    ax = axes[1, 1]
    im = ax.imshow(sampling_range_mask.astype(float), cmap='gray', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm, 
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('采样范围掩模')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # 6. 振幅
    ax = axes[1, 2]
    im = ax.imshow(amp_after, cmap='hot', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm, 
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('振幅')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('tests/output/debug_wfe_measurement.png', dpi=150)
    plt.close()
    print("\n图像已保存到 tests/output/debug_wfe_measurement.png")


if __name__ == "__main__":
    debug_wfe_measurement()
