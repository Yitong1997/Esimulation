"""调试相位混叠问题并验证修复方案

问题：
- ElementRaytracer 计算的像差是正确的
- 但应用到 PROPER 网格时，相位梯度太大导致混叠
- 相邻像素间相位差超过 π 弧度

解决方案探索：
1. 增加网格大小
2. 减小采样范围（只在光束核心区域采样）
3. 使用更小的光束尺寸
4. 验证插值是否正确

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
    FlatMirror,
)


def test_grid_size_effect():
    """测试网格大小对相位采样的影响"""
    print("=" * 70)
    print("测试 1：网格大小对相位采样的影响")
    print("=" * 70)
    
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    print(f"\n配置：抛物面镜 f={focal_length}mm, 倾斜={tilt_deg}°, w0=5mm")
    print(f"预期像差 RMS ≈ 0.68 waves（来自 debug_parabolic_is_fold_false.py）")
    print("-" * 70)
    
    for grid_size in [256, 512, 1024, 2048]:
        system = SequentialOpticalSystem(
            source,
            grid_size=grid_size,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=400,  # 20x20 网格
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
            
            # 检查是否有相位采样警告
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
        
        warning_str = "⚠️ 相位采样不足" if phase_warnings else "✓"
        print(f"  网格 {grid_size:4d}: WFE RMS = {output.wavefront_rms:.4f}, "
              f"去倾斜 RMS = {rms_no_tilt:.4f} waves  {warning_str}")


def test_beam_size_effect():
    """测试光束尺寸对相位采样的影响"""
    print("\n" + "=" * 70)
    print("测试 2：光束尺寸对相位采样的影响")
    print("=" * 70)
    
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    print(f"\n配置：抛物面镜 f={focal_length}mm, 倾斜={tilt_deg}°, 网格=512")
    print("-" * 70)
    
    for w0 in [2.0, 3.0, 5.0, 7.0]:
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=w0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=400,
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
        
        warning_str = "⚠️ 相位采样不足" if phase_warnings else "✓"
        print(f"  w0 = {w0:.1f}mm: WFE RMS = {output.wavefront_rms:.4f}, "
              f"去倾斜 RMS = {rms_no_tilt:.4f} waves  {warning_str}")


def test_small_tilt_angles():
    """测试小倾斜角度下的像差计算"""
    print("\n" + "=" * 70)
    print("测试 3：小倾斜角度下的像差计算")
    print("=" * 70)
    
    focal_length = 100.0
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=3.0,  # 使用较小的光束
        z0=0.0,
    )
    
    print(f"\n配置：抛物面镜 f={focal_length}mm, w0=3mm, 网格=1024")
    print("-" * 70)
    
    for tilt_deg in [0.0, 0.1, 0.2, 0.3, 0.5]:
        tilt_rad = np.deg2rad(tilt_deg)
        
        system = SequentialOpticalSystem(
            source,
            grid_size=1024,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=400,
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
        print(f"  倾斜 {tilt_deg:.1f}°: WFE RMS = {output.wavefront_rms:.4f}, "
              f"去倾斜 RMS = {rms_no_tilt:.4f} waves  {warning_str}")


def analyze_phase_gradient():
    """分析相位梯度，找出混叠的根本原因"""
    print("\n" + "=" * 70)
    print("测试 4：分析相位梯度")
    print("=" * 70)
    
    import proper
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    from scipy.interpolate import griddata
    
    focal_length = 100.0
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    
    for tilt_deg in [0.5, 1.0, 2.0]:
        tilt_rad = np.deg2rad(tilt_deg)
        
        print(f"\n倾斜角度: {tilt_deg}°")
        print("-" * 50)
        
        # 创建采样光线
        n_side = 21
        half_size = 5.0  # mm
        x = np.linspace(-half_size, half_size, n_side)
        y = np.linspace(-half_size, half_size, n_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
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
        
        # 转换为相位
        aberration_phase = -2 * np.pi * aberration_waves
        
        # 计算相位梯度（在采样点之间）
        aberration_grid = aberration_phase.reshape(n_side, n_side)
        grad_x = np.diff(aberration_grid, axis=1)
        grad_y = np.diff(aberration_grid, axis=0)
        
        max_grad = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
        
        # 计算采样间隔
        sampling_mm = 2 * half_size / (n_side - 1)
        
        print(f"  采样范围: ±{half_size}mm, 采样点: {n_side}x{n_side}")
        print(f"  采样间隔: {sampling_mm:.3f} mm")
        print(f"  像差 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
        print(f"  像差 PV: {np.max(aberration_waves[valid_both])-np.min(aberration_waves[valid_both]):.4f} waves")
        print(f"  最大相位梯度: {max_grad:.2f} rad/sample")
        print(f"  是否超过 π: {'是 ⚠️' if max_grad > np.pi else '否 ✓'}")
        
        # 计算需要的采样密度
        if max_grad > np.pi:
            required_samples = int(np.ceil(max_grad / np.pi * (n_side - 1))) + 1
            print(f"  建议采样点数: {required_samples}x{required_samples}")


if __name__ == "__main__":
    test_grid_size_effect()
    test_beam_size_effect()
    test_small_tilt_angles()
    analyze_phase_gradient()
