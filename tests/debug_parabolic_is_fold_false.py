"""调试抛物面镜 is_fold=False 的像差计算

问题：抛物面镜 is_fold=False 时，所有倾斜角度都显示 ~0.29 waves 残余
预期：像差应该随倾斜角度变化

分析步骤：
1. 检查 ElementRaytracer 对抛物面镜的 OPD 计算
2. 检查带倾斜和不带倾斜的 OPD 差异
3. 验证差分方法是否正确
4. 检查倾斜去除是否正确
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def analyze_parabolic_mirror_opd():
    """分析抛物面镜的 OPD 计算"""
    print("=" * 70)
    print("抛物面镜 OPD 分析")
    print("=" * 70)
    
    wavelength_um = 0.633
    focal_length = 100.0  # mm
    
    # 创建光线网格
    n_side = 11
    half_size = 5.0  # mm
    x = np.linspace(-half_size, half_size, n_side)
    y = np.linspace(-half_size, half_size, n_side)
    X, Y = np.meshgrid(x, y)
    ray_x = X.flatten()
    ray_y = Y.flatten()
    n_rays = len(ray_x)
    
    # 测试不同倾斜角度
    for tilt_deg in [0.0, 0.5, 1.0, 2.0]:
        tilt_rad = np.deg2rad(tilt_deg)
        
        print(f"\n倾斜角度: {tilt_deg}°")
        print("-" * 50)
        
        # 带倾斜的抛物面镜
        surface_tilted = SurfaceDefinition(
            surface_type='mirror',
            radius=2 * focal_length,  # R = 2f
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=-1.0,  # 抛物面
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
        
        print(f"  带倾斜 OPD: min={np.min(opd_tilted[valid_tilted]):.4f}, "
              f"max={np.max(opd_tilted[valid_tilted]):.4f}, "
              f"PV={np.max(opd_tilted[valid_tilted])-np.min(opd_tilted[valid_tilted]):.4f} waves")
        print(f"  带倾斜 OPD RMS: {np.std(opd_tilted[valid_tilted]):.4f} waves")

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
        
        print(f"  不带倾斜 OPD: min={np.min(opd_no_tilt[valid_no_tilt]):.4f}, "
              f"max={np.max(opd_no_tilt[valid_no_tilt]):.4f}, "
              f"PV={np.max(opd_no_tilt[valid_no_tilt])-np.min(opd_no_tilt[valid_no_tilt]):.4f} waves")
        print(f"  不带倾斜 OPD RMS: {np.std(opd_no_tilt[valid_no_tilt]):.4f} waves")
        
        # 差分
        center_idx = n_rays // 2
        opd_tilted_aligned = opd_tilted - opd_tilted[center_idx]
        opd_no_tilt_aligned = opd_no_tilt - opd_no_tilt[center_idx]
        diff = opd_tilted_aligned - opd_no_tilt_aligned
        
        valid_both = valid_tilted & valid_no_tilt
        
        print(f"  差分 OPD: min={np.min(diff[valid_both]):.4f}, "
              f"max={np.max(diff[valid_both]):.4f}, "
              f"PV={np.max(diff[valid_both])-np.min(diff[valid_both]):.4f} waves")
        print(f"  差分 RMS: {np.std(diff[valid_both]):.4f} waves")
        
        # 去除倾斜
        valid_x = ray_x[valid_both]
        valid_y = ray_y[valid_both]
        valid_diff = diff[valid_both]
        
        if len(valid_x) > 3:
            # 归一化坐标
            max_r = max(np.max(np.abs(valid_x)), np.max(np.abs(valid_y)))
            if max_r > 0:
                norm_x = valid_x / max_r
                norm_y = valid_y / max_r
            else:
                norm_x = valid_x
                norm_y = valid_y
            
            # 最小二乘拟合：OPD = a0 + a1*x + a2*y
            A = np.column_stack([np.ones_like(norm_x), norm_x, norm_y])
            coeffs, _, _, _ = np.linalg.lstsq(A, valid_diff, rcond=None)
            
            # 计算倾斜分量
            tilt_component = coeffs[0] + coeffs[1] * norm_x + coeffs[2] * norm_y
            
            # 去除倾斜
            aberration = valid_diff - tilt_component
            
            print(f"  倾斜系数: a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}, a2={coeffs[2]:.4f}")
            print(f"  去除倾斜后 RMS: {np.std(aberration):.4f} waves")
            print(f"  去除倾斜后 PV: {np.max(aberration)-np.min(aberration):.4f} waves")


def analyze_theoretical_aberration():
    """分析理论像差值"""
    print("\n" + "=" * 70)
    print("理论像差分析")
    print("=" * 70)
    
    # 对于抛物面镜，倾斜引入的主要像差是像散和彗差
    # 像散 W_22 ∝ θ² * r²
    # 彗差 W_31 ∝ θ * r³
    
    # 使用 Seidel 像差理论估算
    focal_length = 100.0  # mm
    beam_radius = 5.0  # mm
    wavelength = 0.633e-3  # mm
    
    for tilt_deg in [0.5, 1.0, 2.0]:
        tilt_rad = np.deg2rad(tilt_deg)
        
        # 简化估算：
        # 彗差 ≈ (beam_radius / focal_length)³ * tilt_rad * focal_length
        # 像散 ≈ (beam_radius / focal_length)² * tilt_rad² * focal_length
        
        # 更准确的估算需要考虑具体的光学系统
        # 这里只是粗略估算
        
        coma_coeff = (beam_radius / focal_length)**2 * tilt_rad * beam_radius
        astig_coeff = (beam_radius / focal_length) * tilt_rad**2 * beam_radius
        
        coma_waves = coma_coeff / wavelength
        astig_waves = astig_coeff / wavelength
        
        print(f"\n倾斜 {tilt_deg}°:")
        print(f"  估算彗差: {coma_waves:.4f} waves")
        print(f"  估算像散: {astig_waves:.4f} waves")


if __name__ == "__main__":
    analyze_parabolic_mirror_opd()
    analyze_theoretical_aberration()
