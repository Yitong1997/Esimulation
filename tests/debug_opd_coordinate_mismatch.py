"""分析 OPD 计算中的坐标系不匹配问题

核心问题：
- opd_waves 是 ElementRaytracer 计算的相对 OPD
- ideal_opd_waves 是使用理想公式计算的 OPD
- 两者使用的坐标系可能不一致

对于 is_fold=True：
- surface_def_for_trace 不包含倾斜
- 入射和出射光线位置相同（对称）
- 两者坐标系一致

对于 is_fold=False：
- surface_def 包含倾斜
- 入射和出射光线位置不同
- 需要确保两者使用相同的坐标系
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def analyze_coordinate_mismatch():
    """分析坐标系不匹配问题"""
    
    print("=" * 70)
    print("OPD 计算坐标系不匹配分析")
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
    n_rays_1d = 11
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
    # 测试 is_fold=True 的情况
    # =========================================================================
    print("\n1. is_fold=True（使用不带倾斜的 surface_def）：")
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
    ideal_opd_mm_in = calculate_exact_mirror_opd(ray_r_sq_in, focal_length)
    ideal_opd_waves_in = ideal_opd_mm_in / wavelength_mm
    
    # 计算相对理想 OPD（相对于中心光线）
    center_idx = n_rays // 2
    ideal_opd_waves_relative = ideal_opd_waves_in - ideal_opd_waves_in[center_idx]
    
    aberration_fold = opd_waves_fold - ideal_opd_waves_relative
    
    print(f"   实际相对 OPD RMS: {np.nanstd(opd_waves_fold):.4f} waves")
    print(f"   理想相对 OPD RMS: {np.std(ideal_opd_waves_relative):.4f} waves")
    print(f"   像差 RMS: {np.nanstd(aberration_fold):.6f} waves")
    
    # =========================================================================
    # 测试 is_fold=False 的情况（旧方法：使用入射位置）
    # =========================================================================
    print("\n2. is_fold=False 旧方法（使用入射位置计算理想 OPD）：")
    print("-" * 50)
    
    raytracer_no_fold = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_fold = raytracer_no_fold.trace(create_rays())
    opd_waves_no_fold = raytracer_no_fold.get_relative_opd_waves()
    valid_mask_no_fold = raytracer_no_fold.get_valid_ray_mask()
    
    # 旧方法：使用入射光线位置计算理想 OPD
    aberration_old = opd_waves_no_fold - ideal_opd_waves_relative
    
    print(f"   实际相对 OPD RMS: {np.nanstd(opd_waves_no_fold):.4f} waves")
    print(f"   理想相对 OPD RMS: {np.std(ideal_opd_waves_relative):.4f} waves")
    print(f"   像差 RMS (旧方法): {np.nanstd(aberration_old):.4f} waves")
    
    # =========================================================================
    # 测试 is_fold=False 的情况（新方法：使用出射位置）
    # =========================================================================
    print("\n3. is_fold=False 新方法（使用出射位置计算理想 OPD）：")
    print("-" * 50)
    
    # 获取出射光线位置
    x_out = np.array([float(rays_out_no_fold.x[i]) for i in range(n_rays)])
    y_out = np.array([float(rays_out_no_fold.y[i]) for i in range(n_rays)])
    
    # 新方法：使用出射光线位置计算理想 OPD
    ray_r_sq_out = x_out**2 + y_out**2
    ideal_opd_mm_out = calculate_exact_mirror_opd(ray_r_sq_out, focal_length)
    ideal_opd_waves_out = ideal_opd_mm_out / wavelength_mm
    
    # 找到主光线（出射面局部坐标系中最接近原点的光线）
    distances = np.sqrt(x_out**2 + y_out**2)
    distances_valid = np.where(valid_mask_no_fold, distances, np.inf)
    chief_idx = np.argmin(distances_valid)
    
    ideal_opd_waves_out_relative = ideal_opd_waves_out - ideal_opd_waves_out[chief_idx]
    
    aberration_new = opd_waves_no_fold - ideal_opd_waves_out_relative
    
    print(f"   实际相对 OPD RMS: {np.nanstd(opd_waves_no_fold):.4f} waves")
    print(f"   理想相对 OPD RMS (出射位置): {np.nanstd(ideal_opd_waves_out_relative):.4f} waves")
    print(f"   像差 RMS (新方法): {np.nanstd(aberration_new):.4f} waves")
    
    # =========================================================================
    # 关键分析：检查主光线选择
    # =========================================================================
    print("\n4. 主光线选择分析：")
    print("-" * 50)
    
    print(f"   入射中心光线索引: {center_idx}")
    print(f"   入射中心光线位置: ({ray_x[center_idx]:.4f}, {ray_y[center_idx]:.4f})")
    
    print(f"   出射主光线索引: {chief_idx}")
    print(f"   出射主光线入射位置: ({ray_x[chief_idx]:.4f}, {ray_y[chief_idx]:.4f})")
    print(f"   出射主光线出射位置: ({x_out[chief_idx]:.4f}, {y_out[chief_idx]:.4f})")
    
    # =========================================================================
    # 正确的方法：使用相同的主光线
    # =========================================================================
    print("\n5. 正确方法（使用相同的主光线）：")
    print("-" * 50)
    
    # ElementRaytracer 使用出射面局部坐标系中最接近原点的光线作为主光线
    # 我们需要使用相同的主光线来计算理想 OPD
    
    # 获取 ElementRaytracer 使用的主光线的入射位置
    chief_ray_x_in = ray_x[chief_idx]
    chief_ray_y_in = ray_y[chief_idx]
    chief_ray_r_sq_in = chief_ray_x_in**2 + chief_ray_y_in**2
    
    # 计算主光线的理想 OPD
    chief_ideal_opd_mm = calculate_exact_mirror_opd(chief_ray_r_sq_in, focal_length)
    chief_ideal_opd_waves = chief_ideal_opd_mm / wavelength_mm
    
    # 使用入射位置计算所有光线的理想 OPD，然后减去主光线的理想 OPD
    ideal_opd_waves_correct = ideal_opd_waves_in - chief_ideal_opd_waves
    
    aberration_correct = opd_waves_no_fold - ideal_opd_waves_correct
    
    print(f"   主光线入射位置: ({chief_ray_x_in:.4f}, {chief_ray_y_in:.4f})")
    print(f"   主光线理想 OPD: {chief_ideal_opd_waves:.4f} waves")
    print(f"   像差 RMS (正确方法): {np.nanstd(aberration_correct):.4f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"""
问题根源：
1. ElementRaytracer.get_relative_opd_waves() 使用出射面局部坐标系中
   最接近原点的光线作为主光线（索引 {chief_idx}）
2. 但 _apply_element_hybrid 中计算理想 OPD 时，使用的是入射中心光线
   （索引 {center_idx}）作为参考
3. 对于倾斜表面，这两个光线不是同一条光线！
4. 这导致了像差计算的不匹配

解决方案：
- 在计算理想 OPD 时，使用与 ElementRaytracer 相同的主光线作为参考
- 即：使用出射面局部坐标系中最接近原点的光线的入射位置来计算理想 OPD
""")


if __name__ == "__main__":
    analyze_coordinate_mismatch()
