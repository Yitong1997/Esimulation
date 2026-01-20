"""深入分析倾斜抛物面镜的 OPD

问题：为什么倾斜抛物面镜的相对 OPD RMS（110 waves）比无倾斜的（99 waves）大？

假设：
1. 抛物面镜对轴上点源是无像差的
2. 倾斜只改变光轴方向，不应该引入额外的像差
3. 但是，倾斜会改变光线在表面上的入射位置分布

关键问题：
- 对于倾斜的抛物面镜，入射光线在表面上的分布不再对称
- 这可能导致 OPD 分布的变化
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def analyze_tilted_parabola():
    """分析倾斜抛物面镜的 OPD"""
    
    print("=" * 70)
    print("倾斜抛物面镜 OPD 深入分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    tilt_x = np.pi / 4  # 45°
    focal_length = 100.0  # mm
    
    # 创建带倾斜的抛物面镜
    surface_with_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=10.0,
        conic=-1.0,
        tilt_x=tilt_x,
        tilt_y=0.0,
    )
    
    # 创建不带倾斜的抛物面镜
    surface_no_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=10.0,
        conic=-1.0,
        tilt_x=0.0,
        tilt_y=0.0,
    )
    
    # 创建采样光线网格
    n_rays_1d = 21
    half_size = 5.0  # mm
    ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    def create_rays():
        return RealRays(
            x=ray_x.copy(),
            y=ray_y.copy(),
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
    
    # =========================================================================
    # 追迹无倾斜抛物面镜
    # =========================================================================
    print("\n1. 无倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    
    # 获取绝对 OPD
    abs_opd_no_tilt = np.asarray(rays_out_no_tilt.opd)
    
    print(f"   绝对 OPD 范围: [{np.min(abs_opd_no_tilt):.6f}, {np.max(abs_opd_no_tilt):.6f}] mm")
    print(f"   绝对 OPD RMS: {np.std(abs_opd_no_tilt):.6f} mm")
    
    # =========================================================================
    # 追迹带倾斜抛物面镜
    # =========================================================================
    print("\n2. 45° 倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    
    # 获取绝对 OPD
    abs_opd_with_tilt = np.asarray(rays_out_with_tilt.opd)
    
    print(f"   绝对 OPD 范围: [{np.min(abs_opd_with_tilt):.6f}, {np.max(abs_opd_with_tilt):.6f}] mm")
    print(f"   绝对 OPD RMS: {np.std(abs_opd_with_tilt):.6f} mm")
    
    # =========================================================================
    # 分析 OPD 差异
    # =========================================================================
    print("\n3. OPD 差异分析：")
    print("-" * 50)
    
    opd_diff = abs_opd_with_tilt - abs_opd_no_tilt
    print(f"   OPD 差异范围: [{np.min(opd_diff):.6f}, {np.max(opd_diff):.6f}] mm")
    print(f"   OPD 差异 RMS: {np.std(opd_diff):.6f} mm")
    print(f"   OPD 差异 RMS: {np.std(opd_diff) / wavelength_mm:.4f} waves")
    
    # =========================================================================
    # 分析出射光线方向
    # =========================================================================
    print("\n4. 出射光线方向分析：")
    print("-" * 50)
    
    L_no_tilt = np.asarray(rays_out_no_tilt.L)
    M_no_tilt = np.asarray(rays_out_no_tilt.M)
    N_no_tilt = np.asarray(rays_out_no_tilt.N)
    
    L_with_tilt = np.asarray(rays_out_with_tilt.L)
    M_with_tilt = np.asarray(rays_out_with_tilt.M)
    N_with_tilt = np.asarray(rays_out_with_tilt.N)
    
    # 中心光线
    center_idx = n_rays // 2
    print(f"   无倾斜中心光线方向: ({L_no_tilt[center_idx]:.4f}, {M_no_tilt[center_idx]:.4f}, {N_no_tilt[center_idx]:.4f})")
    print(f"   有倾斜中心光线方向: ({L_with_tilt[center_idx]:.4f}, {M_with_tilt[center_idx]:.4f}, {N_with_tilt[center_idx]:.4f})")
    
    # =========================================================================
    # 关键分析：检查 OPD 是否包含倾斜分量
    # =========================================================================
    print("\n5. OPD 倾斜分量分析：")
    print("-" * 50)
    
    # 对于 tilt_x 倾斜，如果 OPD 包含倾斜分量，它应该与 y 坐标线性相关
    # OPD_tilt = 2 * y * sin(tilt_x)（反射镜 OPD 加倍）
    
    # 理论倾斜 OPD
    theoretical_tilt_opd = 2 * ray_y * np.sin(tilt_x)  # mm
    
    print(f"   理论倾斜 OPD 范围: [{np.min(theoretical_tilt_opd):.4f}, {np.max(theoretical_tilt_opd):.4f}] mm")
    print(f"   理论倾斜 OPD RMS: {np.std(theoretical_tilt_opd):.4f} mm")
    print(f"   理论倾斜 OPD RMS: {np.std(theoretical_tilt_opd) / wavelength_mm:.4f} waves")
    
    # 检查实际 OPD 差异是否与理论倾斜 OPD 相关
    correlation = np.corrcoef(opd_diff, theoretical_tilt_opd)[0, 1]
    print(f"   OPD 差异与理论倾斜 OPD 的相关系数: {correlation:.4f}")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if abs(correlation) > 0.9:
        print("""
OPD 差异主要来自倾斜分量！

这说明 ElementRaytracer 计算的绝对 OPD 包含了倾斜引入的 OPD。
但是，get_relative_opd_waves() 应该已经减去了主光线的 OPD，
所以相对 OPD 不应该包含倾斜分量。

需要进一步检查 get_relative_opd_waves() 的实现。
""")
    else:
        print("""
OPD 差异不是来自倾斜分量。

可能的原因：
1. 倾斜改变了光线在表面上的入射位置分布
2. 这导致了不同的 OPD 分布
3. 但这不是"像差"，而是几何效应
""")


if __name__ == "__main__":
    analyze_tilted_parabola()
