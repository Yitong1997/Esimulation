"""调试相位梯度的来源

问题：即使使用密集采样，相位梯度仍然很大
原因分析：
1. 采样点上的相位梯度可能本身就很大
2. 插值到 PROPER 网格后，梯度被放大

需要验证：
1. 采样点上的相位梯度
2. 插值后的相位梯度
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from scipy.interpolate import griddata


def analyze_phase_gradient_at_source():
    """分析采样点上的相位梯度"""
    print("=" * 70)
    print("分析采样点上的相位梯度")
    print("=" * 70)
    
    wavelength_um = 0.633
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    # 不同的采样密度
    for n_side in [32, 64, 128]:
        print(f"\n采样点数: {n_side}x{n_side}")
        print("-" * 50)
        
        # 创建采样光线
        half_size = 7.5  # mm (1.5 * w0)
        x = np.linspace(-half_size, half_size, n_side)
        y = np.linspace(-half_size, half_size, n_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        sampling_interval = 2 * half_size / (n_side - 1)
        print(f"  采样间隔: {sampling_interval:.4f} mm")
        
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
        
        # 计算采样点上的相位梯度
        aberration_grid = aberration_phase.reshape(n_side, n_side)
        grad_x = np.diff(aberration_grid, axis=1)
        grad_y = np.diff(aberration_grid, axis=0)
        
        max_grad_sample = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
        
        print(f"  像差 RMS: {np.std(aberration_waves[valid_both]):.4f} waves")
        print(f"  采样点上最大相位梯度: {max_grad_sample:.2f} rad/sample")
        print(f"  是否超过 π: {'是 ⚠️' if max_grad_sample > np.pi else '否 ✓'}")
        
        # 模拟插值到 PROPER 网格
        proper_grid_size = 512
        proper_sampling_mm = 0.078  # 典型值
        proper_half_size_mm = proper_sampling_mm * proper_grid_size / 2
        
        coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, proper_grid_size)
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
            fill_value=0.0,
        )
        
        phase_grid = np.nan_to_num(phase_grid, nan=0.0)
        
        # 计算插值后的相位梯度
        grad_x_interp = np.diff(phase_grid, axis=1)
        grad_y_interp = np.diff(phase_grid, axis=0)
        
        max_grad_interp = max(np.nanmax(np.abs(grad_x_interp)), np.nanmax(np.abs(grad_y_interp)))
        
        print(f"  插值后最大相位梯度: {max_grad_interp:.2f} rad/pixel")
        print(f"  是否超过 π: {'是 ⚠️' if max_grad_interp > np.pi else '否 ✓'}")


if __name__ == "__main__":
    analyze_phase_gradient_at_source()
