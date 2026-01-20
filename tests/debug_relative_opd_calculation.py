"""检查相对 OPD 的计算

验证 ElementRaytracer.get_relative_opd_waves() 的计算是否正确。

关键问题：
- 相对 OPD = 光线 OPD - 主光线 OPD
- 主光线是出射面局部坐标系中最接近原点的光线
- 对于理想的抛物面镜，相对 OPD 应该只包含聚焦效果
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def check_relative_opd_calculation():
    """检查相对 OPD 的计算"""
    
    print("=" * 70)
    print("相对 OPD 计算检查")
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
    
    # 创建简单的输入光线（只有中心光线和边缘光线）
    ray_x = np.array([0.0, 5.0, -5.0, 0.0, 0.0])
    ray_y = np.array([0.0, 0.0, 0.0, 5.0, -5.0])
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
    # 检查无倾斜情况
    # =========================================================================
    print("\n1. 无倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    
    # 手动计算相对 OPD
    opd_mm = np.array([float(rays_out_no_tilt.opd[i]) for i in range(n_rays)])
    x_out = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    
    # 找到主光线（出射面局部坐标系中最接近原点的光线）
    distances = np.sqrt(x_out**2 + y_out**2)
    chief_idx = np.argmin(distances)
    chief_opd_mm = opd_mm[chief_idx]
    
    # 计算相对 OPD
    relative_opd_mm = opd_mm - chief_opd_mm
    relative_opd_waves = relative_opd_mm / wavelength_mm
    
    print(f"   主光线索引: {chief_idx}")
    print(f"   主光线 OPD: {chief_opd_mm:.6f} mm")
    print("\n   光线 OPD 和相对 OPD：")
    for i in range(n_rays):
        print(f"     光线 {i} (入射 {ray_x[i]:.1f}, {ray_y[i]:.1f}): "
              f"OPD={opd_mm[i]:.6f} mm, 相对 OPD={relative_opd_waves[i]:.4f} waves")
    
    # 使用 get_relative_opd_waves() 方法
    opd_waves_method = raytracer_no_tilt.get_relative_opd_waves()
    print("\n   get_relative_opd_waves() 结果：")
    for i in range(n_rays):
        print(f"     光线 {i}: {opd_waves_method[i]:.4f} waves")
    
    # =========================================================================
    # 检查有倾斜情况
    # =========================================================================
    print("\n2. 45° 倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    
    # 手动计算相对 OPD
    opd_mm_tilt = np.array([float(rays_out_with_tilt.opd[i]) for i in range(n_rays)])
    x_out_tilt = np.array([float(rays_out_with_tilt.x[i]) for i in range(n_rays)])
    y_out_tilt = np.array([float(rays_out_with_tilt.y[i]) for i in range(n_rays)])
    
    # 找到主光线
    distances_tilt = np.sqrt(x_out_tilt**2 + y_out_tilt**2)
    chief_idx_tilt = np.argmin(distances_tilt)
    chief_opd_mm_tilt = opd_mm_tilt[chief_idx_tilt]
    
    # 计算相对 OPD
    relative_opd_mm_tilt = opd_mm_tilt - chief_opd_mm_tilt
    relative_opd_waves_tilt = relative_opd_mm_tilt / wavelength_mm
    
    print(f"   主光线索引: {chief_idx_tilt}")
    print(f"   主光线 OPD: {chief_opd_mm_tilt:.6f} mm")
    print("\n   光线 OPD 和相对 OPD：")
    for i in range(n_rays):
        print(f"     光线 {i} (入射 {ray_x[i]:.1f}, {ray_y[i]:.1f}): "
              f"OPD={opd_mm_tilt[i]:.6f} mm, 相对 OPD={relative_opd_waves_tilt[i]:.4f} waves")
    
    # 使用 get_relative_opd_waves() 方法
    opd_waves_method_tilt = raytracer_with_tilt.get_relative_opd_waves()
    print("\n   get_relative_opd_waves() 结果：")
    for i in range(n_rays):
        print(f"     光线 {i}: {opd_waves_method_tilt[i]:.4f} waves")
    
    # =========================================================================
    # 比较理想 OPD
    # =========================================================================
    print("\n3. 理想 OPD 比较：")
    print("-" * 50)
    
    # 对于无倾斜情况，使用入射位置计算理想 OPD
    r_sq = ray_x**2 + ray_y**2
    
    def calculate_exact_mirror_opd(r_sq, f):
        """计算理想抛物面镜的精确 OPD"""
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    ideal_opd_mm = calculate_exact_mirror_opd(r_sq, focal_length)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    
    print("   理想 OPD（使用入射位置）：")
    for i in range(n_rays):
        print(f"     光线 {i} (r={np.sqrt(r_sq[i]):.1f}): {ideal_opd_waves[i]:.4f} waves")
    
    # 计算像差
    print("\n   无倾斜像差 = 实际相对 OPD - 理想 OPD：")
    for i in range(n_rays):
        aberration = relative_opd_waves[i] - ideal_opd_waves[i]
        print(f"     光线 {i}: {aberration:.6f} waves")
    
    print("\n   有倾斜像差 = 实际相对 OPD - 理想 OPD：")
    for i in range(n_rays):
        aberration = relative_opd_waves_tilt[i] - ideal_opd_waves[i]
        print(f"     光线 {i}: {aberration:.6f} waves")


if __name__ == "__main__":
    check_relative_opd_calculation()
