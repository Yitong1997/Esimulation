"""调试混合传播模式的 OPD 流程

检查 _apply_element_hybrid 中的 OPD 计算流程，
找出为什么 is_fold=False 时会有大的 WFE。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_hybrid_opd_flow():
    """测试混合传播模式的 OPD 流程"""
    
    print("=" * 70)
    print("混合传播模式 OPD 流程调试")
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
    
    # 创建采样光线网格（模拟 _create_sampling_rays）
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
    
    # =========================================================================
    # 模拟 _apply_element_hybrid 的流程
    # =========================================================================
    
    print("\n1. 模拟 is_fold=True 的流程：")
    print("-" * 50)
    print("   使用不带倾斜的 surface_def_for_trace")
    
    raytracer_fold = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_fold = raytracer_fold.trace(create_rays())
    opd_waves_fold = raytracer_fold.get_relative_opd_waves()
    valid_mask_fold = raytracer_fold.get_valid_ray_mask()
    
    # 计算理想 OPD
    ray_r_sq = ray_x**2 + ray_y**2
    
    def calculate_exact_mirror_opd(r_sq, f):
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    ideal_opd_mm = calculate_exact_mirror_opd(ray_r_sq, focal_length)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    
    # 像差 = 实际 OPD - 理想 OPD
    aberration_fold = opd_waves_fold - ideal_opd_waves
    aberration_fold_valid = aberration_fold[valid_mask_fold]
    
    print(f"   实际 OPD RMS: {np.std(opd_waves_fold[valid_mask_fold]):.4f} waves")
    print(f"   理想 OPD RMS: {np.std(ideal_opd_waves):.4f} waves")
    print(f"   像差 RMS: {np.std(aberration_fold_valid):.4f} waves")
    
    print("\n2. 模拟 is_fold=False 的流程：")
    print("-" * 50)
    print("   使用带倾斜的 surface_def")
    
    raytracer_no_fold = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_fold = raytracer_no_fold.trace(create_rays())
    opd_waves_no_fold = raytracer_no_fold.get_relative_opd_waves()
    valid_mask_no_fold = raytracer_no_fold.get_valid_ray_mask()
    
    # 像差 = 实际 OPD - 理想 OPD
    # 注意：理想 OPD 仍然使用无倾斜的公式！
    aberration_no_fold = opd_waves_no_fold - ideal_opd_waves
    aberration_no_fold_valid = aberration_no_fold[valid_mask_no_fold]
    
    print(f"   实际 OPD RMS: {np.std(opd_waves_no_fold[valid_mask_no_fold]):.4f} waves")
    print(f"   理想 OPD RMS: {np.std(ideal_opd_waves):.4f} waves")
    print(f"   像差 RMS: {np.std(aberration_no_fold_valid):.4f} waves")
    
    print("\n3. 关键问题分析：")
    print("-" * 50)
    print("   在 _apply_element_hybrid 中：")
    print("   - 理想 OPD 使用 _calculate_exact_mirror_opd 计算")
    print("   - 这个公式假设表面没有倾斜")
    print("   - 对于 is_fold=False，实际 OPD 包含倾斜效果")
    print("   - 但理想 OPD 不包含倾斜效果")
    print("   - 因此，像差 = 实际 OPD - 理想 OPD 包含了倾斜引入的差异")
    
    print("\n4. 验证：检查出射光线位置")
    print("-" * 50)
    
    # 检查出射光线在出射面局部坐标系中的位置
    x_out_fold = np.array([float(rays_out_fold.x[i]) for i in range(n_rays)])
    y_out_fold = np.array([float(rays_out_fold.y[i]) for i in range(n_rays)])
    
    x_out_no_fold = np.array([float(rays_out_no_fold.x[i]) for i in range(n_rays)])
    y_out_no_fold = np.array([float(rays_out_no_fold.y[i]) for i in range(n_rays)])
    
    print(f"   is_fold=True 出射光线位置范围:")
    print(f"     x: [{np.min(x_out_fold):.2f}, {np.max(x_out_fold):.2f}]")
    print(f"     y: [{np.min(y_out_fold):.2f}, {np.max(y_out_fold):.2f}]")
    
    print(f"   is_fold=False 出射光线位置范围:")
    print(f"     x: [{np.min(x_out_no_fold):.2f}, {np.max(x_out_no_fold):.2f}]")
    print(f"     y: [{np.min(y_out_no_fold):.2f}, {np.max(y_out_no_fold):.2f}]")
    
    print("\n5. 问题根源：")
    print("-" * 50)
    print("   理想 OPD 公式使用入射光线位置 (ray_x, ray_y) 计算")
    print("   但对于带倾斜的表面，出射光线位置已经变化了")
    print("   ")
    print("   更重要的是：理想 OPD 公式假设表面没有倾斜")
    print("   对于倾斜的表面，理想 OPD 应该考虑倾斜效果")
    print("   ")
    print("   但是，如果采样面垂直于出射光轴，")
    print("   那么 ElementRaytracer 计算的相对 OPD 应该已经是正确的")
    print("   不需要再减去理想 OPD")
    
    print("\n6. 验证 ElementRaytracer 的相对 OPD：")
    print("-" * 50)
    print("   ElementRaytracer.get_relative_opd_waves() 使用")
    print("   出射面局部坐标系中最接近原点的光线作为主光线")
    print("   相对 OPD = 光线 OPD - 主光线 OPD")
    print("   ")
    print("   如果采样面垂直于出射光轴，这个相对 OPD 应该")
    print("   只包含元件引入的像差，不包含倾斜分量")
    
    # 检查主光线
    print("\n   主光线信息：")
    
    # is_fold=True
    distances_fold = np.sqrt(x_out_fold**2 + y_out_fold**2)
    chief_idx_fold = np.argmin(distances_fold)
    print(f"   is_fold=True:")
    print(f"     主光线索引: {chief_idx_fold}")
    print(f"     主光线入射位置: ({ray_x[chief_idx_fold]:.2f}, {ray_y[chief_idx_fold]:.2f})")
    print(f"     主光线出射位置: ({x_out_fold[chief_idx_fold]:.4f}, {y_out_fold[chief_idx_fold]:.4f})")
    
    # is_fold=False
    distances_no_fold = np.sqrt(x_out_no_fold**2 + y_out_no_fold**2)
    chief_idx_no_fold = np.argmin(distances_no_fold)
    print(f"   is_fold=False:")
    print(f"     主光线索引: {chief_idx_no_fold}")
    print(f"     主光线入射位置: ({ray_x[chief_idx_no_fold]:.2f}, {ray_y[chief_idx_no_fold]:.2f})")
    print(f"     主光线出射位置: ({x_out_no_fold[chief_idx_no_fold]:.4f}, {y_out_no_fold[chief_idx_no_fold]:.4f})")


if __name__ == "__main__":
    test_hybrid_opd_flow()
