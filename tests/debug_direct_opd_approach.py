"""测试直接使用 ElementRaytracer 相对 OPD 的方法

假设：
- 对于 is_fold=False 的倾斜表面，ElementRaytracer 的相对 OPD
  已经是正确的像差（相对于主光线）
- 不需要再减去理想 OPD
- 因为 ElementRaytracer 已经在出射面局部坐标系中计算了相对 OPD

验证方法：
- 对于理想的抛物面镜，相对 OPD 应该只包含聚焦效果
- 减去聚焦效果后，残差应该接近零
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_direct_opd_approach():
    """测试直接使用相对 OPD 的方法"""
    
    print("=" * 70)
    print("直接使用 ElementRaytracer 相对 OPD 的方法测试")
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
    
    def calculate_exact_mirror_opd(r_sq, f):
        """计算理想抛物面镜的精确 OPD"""
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    # =========================================================================
    # 测试无倾斜情况
    # =========================================================================
    print("\n1. 无倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    opd_waves_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
    valid_mask_no_tilt = raytracer_no_tilt.get_valid_ray_mask()
    
    # 获取出射光线位置
    x_out_no_tilt = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out_no_tilt = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    
    # 使用出射光线位置计算理想 OPD
    ray_r_sq_out = x_out_no_tilt**2 + y_out_no_tilt**2
    ideal_opd_mm = calculate_exact_mirror_opd(ray_r_sq_out, focal_length)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    
    # 找到主光线
    distances = np.sqrt(x_out_no_tilt**2 + y_out_no_tilt**2)
    distances_valid = np.where(valid_mask_no_tilt, distances, np.inf)
    chief_idx = np.argmin(distances_valid)
    
    # 计算相对理想 OPD
    ideal_opd_waves_relative = ideal_opd_waves - ideal_opd_waves[chief_idx]
    
    # 像差 = 实际相对 OPD - 理想相对 OPD
    aberration_no_tilt = opd_waves_no_tilt - ideal_opd_waves_relative
    
    print(f"   实际相对 OPD RMS: {np.nanstd(opd_waves_no_tilt):.4f} waves")
    print(f"   理想相对 OPD RMS: {np.nanstd(ideal_opd_waves_relative):.4f} waves")
    print(f"   像差 RMS: {np.nanstd(aberration_no_tilt):.6f} waves")
    
    # =========================================================================
    # 测试有倾斜情况
    # =========================================================================
    print("\n2. 45° 倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    opd_waves_with_tilt = raytracer_with_tilt.get_relative_opd_waves()
    valid_mask_with_tilt = raytracer_with_tilt.get_valid_ray_mask()
    
    # 获取出射光线位置
    x_out_with_tilt = np.array([float(rays_out_with_tilt.x[i]) for i in range(n_rays)])
    y_out_with_tilt = np.array([float(rays_out_with_tilt.y[i]) for i in range(n_rays)])
    
    # 使用出射光线位置计算理想 OPD
    ray_r_sq_out_tilt = x_out_with_tilt**2 + y_out_with_tilt**2
    ideal_opd_mm_tilt = calculate_exact_mirror_opd(ray_r_sq_out_tilt, focal_length)
    ideal_opd_waves_tilt = ideal_opd_mm_tilt / wavelength_mm
    
    # 找到主光线
    distances_tilt = np.sqrt(x_out_with_tilt**2 + y_out_with_tilt**2)
    distances_valid_tilt = np.where(valid_mask_with_tilt, distances_tilt, np.inf)
    chief_idx_tilt = np.argmin(distances_valid_tilt)
    
    # 计算相对理想 OPD
    ideal_opd_waves_relative_tilt = ideal_opd_waves_tilt - ideal_opd_waves_tilt[chief_idx_tilt]
    
    # 像差 = 实际相对 OPD - 理想相对 OPD
    aberration_with_tilt = opd_waves_with_tilt - ideal_opd_waves_relative_tilt
    
    print(f"   实际相对 OPD RMS: {np.nanstd(opd_waves_with_tilt):.4f} waves")
    print(f"   理想相对 OPD RMS: {np.nanstd(ideal_opd_waves_relative_tilt):.4f} waves")
    print(f"   像差 RMS: {np.nanstd(aberration_with_tilt):.6f} waves")
    
    # =========================================================================
    # 关键分析：检查主光线位置
    # =========================================================================
    print("\n3. 主光线位置分析：")
    print("-" * 50)
    
    print(f"   无倾斜主光线索引: {chief_idx}")
    print(f"   无倾斜主光线入射位置: ({ray_x[chief_idx]:.4f}, {ray_y[chief_idx]:.4f})")
    print(f"   无倾斜主光线出射位置: ({x_out_no_tilt[chief_idx]:.4f}, {y_out_no_tilt[chief_idx]:.4f})")
    
    print(f"   有倾斜主光线索引: {chief_idx_tilt}")
    print(f"   有倾斜主光线入射位置: ({ray_x[chief_idx_tilt]:.4f}, {ray_y[chief_idx_tilt]:.4f})")
    print(f"   有倾斜主光线出射位置: ({x_out_with_tilt[chief_idx_tilt]:.4f}, {y_out_with_tilt[chief_idx_tilt]:.4f})")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if np.nanstd(aberration_with_tilt) < 0.01:
        print("✓ 使用出射光线位置计算理想 OPD 后，像差接近零。")
        print("  这说明 ElementRaytracer 的相对 OPD 是正确的。")
    else:
        print(f"✗ 像差仍然较大（{np.nanstd(aberration_with_tilt):.4f} waves）。")
        print("  需要进一步分析原因。")
        
        # 进一步分析
        print("\n进一步分析：")
        print("-" * 50)
        
        # 检查 OPD 差异的空间分布
        opd_diff = opd_waves_with_tilt - ideal_opd_waves_relative_tilt
        opd_diff_valid = opd_diff[valid_mask_with_tilt]
        
        # 检查是否有系统性偏差
        print(f"   OPD 差异均值: {np.nanmean(opd_diff):.4f} waves")
        print(f"   OPD 差异 RMS: {np.nanstd(opd_diff):.4f} waves")
        
        # 检查与坐标的相关性
        x_valid = x_out_with_tilt[valid_mask_with_tilt]
        y_valid = y_out_with_tilt[valid_mask_with_tilt]
        
        corr_x = np.corrcoef(x_valid, opd_diff_valid)[0, 1]
        corr_y = np.corrcoef(y_valid, opd_diff_valid)[0, 1]
        corr_r = np.corrcoef(np.sqrt(x_valid**2 + y_valid**2), opd_diff_valid)[0, 1]
        
        print(f"   OPD 差异与 x 的相关系数: {corr_x:.4f}")
        print(f"   OPD 差异与 y 的相关系数: {corr_y:.4f}")
        print(f"   OPD 差异与 r 的相关系数: {corr_r:.4f}")


if __name__ == "__main__":
    test_direct_opd_approach()
