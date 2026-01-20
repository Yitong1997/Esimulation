"""使用出射位置计算理想 OPD

验证假设：对于倾斜的抛物面镜，应该使用出射光线在出射面局部坐标系中的位置
来计算理想 OPD。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_output_position_ideal_opd():
    """使用出射位置计算理想 OPD"""
    
    print("=" * 70)
    print("使用出射位置计算理想 OPD")
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
    
    # 创建简单的输入光线
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
    
    def calculate_exact_mirror_opd(r_sq, f):
        """计算理想抛物面镜的精确 OPD"""
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    # 追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out = raytracer.trace(create_rays())
    
    # 获取出射光线位置和 OPD
    x_out = np.array([float(rays_out.x[i]) for i in range(n_rays)])
    y_out = np.array([float(rays_out.y[i]) for i in range(n_rays)])
    opd_mm = np.array([float(rays_out.opd[i]) for i in range(n_rays)])
    
    # 计算相对 OPD
    chief_opd_mm = opd_mm[0]  # 中心光线
    relative_opd_mm = opd_mm - chief_opd_mm
    relative_opd_waves = relative_opd_mm / wavelength_mm
    
    print("\n1. 出射光线位置和相对 OPD：")
    print("-" * 50)
    for i in range(n_rays):
        print(f"   光线 {i}: 入射({ray_x[i]:.1f}, {ray_y[i]:.1f}) -> "
              f"出射({x_out[i]:.4f}, {y_out[i]:.4f}), "
              f"相对 OPD={relative_opd_waves[i]:.4f} waves")
    
    # 使用出射位置计算理想 OPD
    r_sq_out = x_out**2 + y_out**2
    ideal_opd_mm_out = calculate_exact_mirror_opd(r_sq_out, focal_length)
    ideal_opd_waves_out = ideal_opd_mm_out / wavelength_mm
    
    # 计算相对理想 OPD
    ideal_opd_waves_out_relative = ideal_opd_waves_out - ideal_opd_waves_out[0]
    
    print("\n2. 使用出射位置计算的理想 OPD：")
    print("-" * 50)
    for i in range(n_rays):
        print(f"   光线 {i}: r_out={np.sqrt(r_sq_out[i]):.4f}, "
              f"理想 OPD={ideal_opd_waves_out_relative[i]:.4f} waves")
    
    # 计算像差
    aberration_out = relative_opd_waves - ideal_opd_waves_out_relative
    
    print("\n3. 像差（使用出射位置）：")
    print("-" * 50)
    for i in range(n_rays):
        print(f"   光线 {i}: {aberration_out[i]:.6f} waves")
    
    print(f"\n   像差 RMS: {np.std(aberration_out):.6f} waves")
    
    # 使用入射位置计算理想 OPD（对比）
    r_sq_in = ray_x**2 + ray_y**2
    ideal_opd_mm_in = calculate_exact_mirror_opd(r_sq_in, focal_length)
    ideal_opd_waves_in = ideal_opd_mm_in / wavelength_mm
    ideal_opd_waves_in_relative = ideal_opd_waves_in - ideal_opd_waves_in[0]
    
    aberration_in = relative_opd_waves - ideal_opd_waves_in_relative
    
    print("\n4. 像差（使用入射位置，对比）：")
    print("-" * 50)
    for i in range(n_rays):
        print(f"   光线 {i}: {aberration_in[i]:.6f} waves")
    
    print(f"\n   像差 RMS: {np.std(aberration_in):.6f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if np.std(aberration_out) < np.std(aberration_in):
        print(f"使用出射位置计算理想 OPD 的像差更小：")
        print(f"  出射位置像差 RMS: {np.std(aberration_out):.6f} waves")
        print(f"  入射位置像差 RMS: {np.std(aberration_in):.6f} waves")
    else:
        print(f"使用入射位置计算理想 OPD 的像差更小：")
        print(f"  入射位置像差 RMS: {np.std(aberration_in):.6f} waves")
        print(f"  出射位置像差 RMS: {np.std(aberration_out):.6f} waves")


if __name__ == "__main__":
    test_output_position_ideal_opd()
