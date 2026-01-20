"""详细调试插值过程

问题：
- 光线追迹采样点数量（hybrid_num_rays）默认为 100（10x10）
- 但 PROPER 网格大小为 512x512 或更大
- 插值从 10x10 到 512x512 会导致相位梯度被放大

解决方案：
- 增加光线追迹采样点数量
- 或者使用更智能的插值方法

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import warnings

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from gaussian_beam_simulation.optical_elements import (
    ParabolicMirror,
)


def test_hybrid_num_rays_effect():
    """测试 hybrid_num_rays 对结果的影响"""
    print("=" * 70)
    print("测试：hybrid_num_rays 对结果的影响")
    print("=" * 70)
    
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    print(f"\n配置：抛物面镜 f={focal_length}mm, 倾斜={tilt_deg}°, w0=5mm, 网格=512")
    print(f"预期像差 RMS ≈ 0.68 waves")
    print("-" * 70)
    
    for num_rays in [100, 400, 900, 1600, 2500, 10000]:
        n_side = int(np.sqrt(num_rays))
        
        system = SequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=num_rays,
        )
        
        system.add_surface(ParabolicMirror(
            parent_focal_length=focal_length,
            thickness=200.0,
            semi_aperture=15.0,
            tilt_x=tilt_rad,
            is_fold=False,
        ))
        
        system.add_sampling_plane(distance=200.0, name="output")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = system.run()
            
            phase_warnings = [x for x in w if "相位采样不足" in str(x.message)]
        
        output = results["output"]
        
        # 分析相位
        phase = output.phase
        amp = output.amplitude
        mask = amp > 0.01 * np.max(amp)
        
        # 去除倾斜
        n = phase.shape[0]
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        
        valid_phase = phase[mask]
        valid_x = X[mask]
        valid_y = Y[mask]
        
        if len(valid_phase) > 10:
            A = np.column_stack([np.ones_like(valid_x), valid_x, valid_y])
            coeffs, _, _, _ = np.linalg.lstsq(A, valid_phase, rcond=None)
            
            tilt_phase = coeffs[0] + coeffs[1] * X + coeffs[2] * Y
            phase_no_tilt = phase - tilt_phase
            
            valid_no_tilt = phase_no_tilt[mask]
            rms_no_tilt = np.std(valid_no_tilt - np.mean(valid_no_tilt)) / (2 * np.pi)
        else:
            rms_no_tilt = 0.0
        
        warning_str = "⚠️" if phase_warnings else "✓"
        print(f"  {num_rays:5d} rays ({n_side:3d}x{n_side:3d}): "
              f"WFE RMS = {output.wavefront_rms:.4f}, "
              f"去倾斜 RMS = {rms_no_tilt:.4f} waves  {warning_str}")


def test_direct_phase_application():
    """直接测试相位应用，绕过系统"""
    print("\n" + "=" * 70)
    print("测试：直接相位应用（绕过系统）")
    print("=" * 70)
    
    import proper
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    from scipy.interpolate import griddata
    
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
    
    # 测试不同的光线采样数量
    for n_rays_1d in [10, 20, 50, 100]:
        print(f"\n光线采样: {n_rays_1d}x{n_rays_1d}")
        print("-" * 50)
        
        # 创建采样光线
        half_size_mm = 7.5  # 1.5 * w0
        ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
        ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
        ray_x = ray_X.flatten()
        ray_y = ray_Y.flatten()
        n_rays = len(ray_x)
        
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
        
        print(f"  像差 RMS（采样点）: {np.std(aberration_waves[valid_both]):.4f} waves")
        
        # 转换为相位
        aberration_phase = -2 * np.pi * aberration_waves
        
        # 插值到 PROPER 网格
        valid_x_interp = ray_x[valid_both]
        valid_y_interp = ray_y[valid_both]
        valid_phase_interp = aberration_phase[valid_both]
        
        proper_half_size_mm = sampling_mm * n / 2
        coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
        X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
        
        points = np.column_stack([valid_x_interp, valid_y_interp])
        phase_grid = griddata(
            points,
            valid_phase_interp,
            (X_mm, Y_mm),
            method='cubic',
            fill_value=0.0,
        )
        
        phase_grid = np.nan_to_num(phase_grid, nan=0.0)
        
        # 检查相位梯度
        grad_x = np.diff(phase_grid, axis=1)
        grad_y = np.diff(phase_grid, axis=0)
        max_grad = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
        
        print(f"  最大相位梯度: {max_grad:.2f} rad/pixel")
        print(f"  是否超过 π: {'是 ⚠️' if max_grad > np.pi else '否 ✓'}")
        
        # 应用相位到波前
        wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        phase_field = np.exp(1j * phase_grid)
        phase_field_fft = proper.prop_shift_center(phase_field)
        wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
        
        # 测量 WFE
        amp = proper.prop_get_amplitude(wfo_test)
        phase = proper.prop_get_phase(wfo_test)
        mask = amp > 0.01 * np.max(amp)
        
        valid_phase = phase[mask]
        wfe_rms = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
        
        print(f"  测量 WFE RMS: {wfe_rms:.4f} waves")


if __name__ == "__main__":
    test_hybrid_num_rays_effect()
    test_direct_phase_application()
