"""验证修复后的混合传播模式 OPD 流程

验证使用出射光线位置计算理想 OPD 是否解决了 is_fold=False 时的大像差问题。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_fixed_hybrid_opd_flow():
    """测试修复后的混合传播模式 OPD 流程"""
    
    print("=" * 70)
    print("修复后的混合传播模式 OPD 流程验证")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    tilt_x = np.pi / 4  # 45°
    focal_length = 100.0  # mm
    
    # 创建带倾斜的抛物面镜（模拟 is_fold=False）
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
    
    # 创建不带倾斜的抛物面镜（模拟 is_fold=True）
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
    n_rays_1d = 10
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
    
    def calculate_exact_mirror_opd(r_sq, f):
        """计算理想抛物面镜的精确 OPD"""
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    # =========================================================================
    # 测试 is_fold=True 的流程（使用入射光线位置）
    # =========================================================================
    print("\n1. is_fold=True 的流程（使用入射光线位置）：")
    print("-" * 50)
    
    raytracer_fold = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_fold = raytracer_fold.trace(create_rays())
    opd_waves_fold = raytracer_fold.get_relative_opd_waves()
    valid_mask_fold = raytracer_fold.get_valid_ray_mask()
    
    # 使用入射光线位置计算理想 OPD
    ray_r_sq_in = ray_x**2 + ray_y**2
    ideal_opd_mm_fold = calculate_exact_mirror_opd(ray_r_sq_in, focal_length)
    ideal_opd_waves_fold = ideal_opd_mm_fold / wavelength_mm
    
    aberration_fold = opd_waves_fold - ideal_opd_waves_fold
    aberration_fold_valid = aberration_fold[valid_mask_fold]
    
    print(f"   实际 OPD RMS: {np.std(opd_waves_fold[valid_mask_fold]):.4f} waves")
    print(f"   理想 OPD RMS: {np.std(ideal_opd_waves_fold):.4f} waves")
    print(f"   像差 RMS: {np.std(aberration_fold_valid):.4f} waves")
    
    # =========================================================================
    # 测试 is_fold=False 的旧流程（使用入射光线位置 - 错误）
    # =========================================================================
    print("\n2. is_fold=False 的旧流程（使用入射光线位置 - 错误）：")
    print("-" * 50)
    
    raytracer_no_fold = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_fold = raytracer_no_fold.trace(create_rays())
    opd_waves_no_fold = raytracer_no_fold.get_relative_opd_waves()
    valid_mask_no_fold = raytracer_no_fold.get_valid_ray_mask()
    
    # 旧方法：使用入射光线位置计算理想 OPD（错误）
    ideal_opd_waves_old = ideal_opd_mm_fold / wavelength_mm  # 使用入射位置
    
    aberration_old = opd_waves_no_fold - ideal_opd_waves_old
    aberration_old_valid = aberration_old[valid_mask_no_fold]
    
    print(f"   实际 OPD RMS: {np.std(opd_waves_no_fold[valid_mask_no_fold]):.4f} waves")
    print(f"   理想 OPD RMS (入射位置): {np.std(ideal_opd_waves_old):.4f} waves")
    print(f"   像差 RMS (旧方法): {np.std(aberration_old_valid):.4f} waves")
    
    # =========================================================================
    # 测试 is_fold=False 的新流程（使用出射光线位置 - 正确）
    # =========================================================================
    print("\n3. is_fold=False 的新流程（使用出射光线位置 - 正确）：")
    print("-" * 50)
    
    # 获取出射光线位置
    x_out = np.array([float(rays_out_no_fold.x[i]) for i in range(n_rays)])
    y_out = np.array([float(rays_out_no_fold.y[i]) for i in range(n_rays)])
    
    # 新方法：使用出射光线位置计算理想 OPD
    ray_r_sq_out = x_out**2 + y_out**2
    ideal_opd_mm_new = calculate_exact_mirror_opd(ray_r_sq_out, focal_length)
    ideal_opd_waves_new = ideal_opd_mm_new / wavelength_mm
    
    aberration_new = opd_waves_no_fold - ideal_opd_waves_new
    aberration_new_valid = aberration_new[valid_mask_no_fold]
    
    print(f"   实际 OPD RMS: {np.std(opd_waves_no_fold[valid_mask_no_fold]):.4f} waves")
    print(f"   理想 OPD RMS (出射位置): {np.std(ideal_opd_waves_new[valid_mask_no_fold]):.4f} waves")
    print(f"   像差 RMS (新方法): {np.std(aberration_new_valid):.4f} waves")
    
    # =========================================================================
    # 结果对比
    # =========================================================================
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"is_fold=True  像差 RMS: {np.std(aberration_fold_valid):.6f} waves")
    print(f"is_fold=False 旧方法像差 RMS: {np.std(aberration_old_valid):.6f} waves")
    print(f"is_fold=False 新方法像差 RMS: {np.std(aberration_new_valid):.6f} waves")
    
    print("\n结论：")
    if np.std(aberration_new_valid) < 0.01:
        print("✓ 修复成功！使用出射光线位置计算理想 OPD 后，像差接近零。")
    else:
        print("✗ 修复可能不完整，像差仍然较大。")


if __name__ == "__main__":
    test_fixed_hybrid_opd_flow()
