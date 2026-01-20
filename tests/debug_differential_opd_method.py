"""测试差分 OPD 方法

验证使用差分方法计算 is_fold=False 倾斜表面的像差：
- 像差 = 带倾斜的 OPD - 不带倾斜的 OPD
- 这样计算的像差只包含倾斜引入的波前畸变，不包含聚焦效果
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_differential_opd_method():
    """测试差分 OPD 方法"""
    
    print("=" * 70)
    print("差分 OPD 方法测试")
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
    center_idx = n_rays // 2
    
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
    # 追迹带倾斜的表面
    # =========================================================================
    print("\n1. 追迹带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_tilt = raytracer_tilt.trace(create_rays())
    opd_waves_tilt = raytracer_tilt.get_relative_opd_waves()
    valid_mask_tilt = raytracer_tilt.get_valid_ray_mask()
    
    print(f"   相对 OPD RMS: {np.nanstd(opd_waves_tilt):.4f} waves")
    
    # =========================================================================
    # 追迹不带倾斜的表面
    # =========================================================================
    print("\n2. 追迹不带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    opd_waves_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
    valid_mask_no_tilt = raytracer_no_tilt.get_valid_ray_mask()
    
    print(f"   相对 OPD RMS: {np.nanstd(opd_waves_no_tilt):.4f} waves")
    
    # =========================================================================
    # 计算差分像差
    # =========================================================================
    print("\n3. 差分像差计算：")
    print("-" * 50)
    
    # 对齐到中心光线
    opd_tilt_aligned = opd_waves_tilt - opd_waves_tilt[center_idx]
    opd_no_tilt_aligned = opd_waves_no_tilt - opd_waves_no_tilt[center_idx]
    
    # 计算差异
    aberration_diff = opd_tilt_aligned - opd_no_tilt_aligned
    
    # 合并有效掩模
    valid_mask = valid_mask_tilt & valid_mask_no_tilt
    aberration_diff_valid = aberration_diff[valid_mask]
    
    print(f"   差分像差 RMS: {np.std(aberration_diff_valid):.6f} waves")
    print(f"   差分像差 PV: {np.max(aberration_diff_valid) - np.min(aberration_diff_valid):.6f} waves")
    
    # =========================================================================
    # 分析差分像差的组成
    # =========================================================================
    print("\n4. 差分像差分析：")
    print("-" * 50)
    
    # 检查与坐标的相关性
    x_valid = ray_x[valid_mask]
    y_valid = ray_y[valid_mask]
    
    corr_x = np.corrcoef(x_valid, aberration_diff_valid)[0, 1]
    corr_y = np.corrcoef(y_valid, aberration_diff_valid)[0, 1]
    corr_r = np.corrcoef(np.sqrt(x_valid**2 + y_valid**2), aberration_diff_valid)[0, 1]
    
    print(f"   与 x 的相关系数: {corr_x:.4f}")
    print(f"   与 y 的相关系数: {corr_y:.4f}")
    print(f"   与 r 的相关系数: {corr_r:.4f}")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if np.std(aberration_diff_valid) < 0.1:
        print("✓ 差分像差很小，说明倾斜不引入额外的像差。")
        print("  这符合预期：抛物面镜对轴上点源是无像差的，")
        print("  倾斜只改变光轴方向，不引入额外的像差。")
    else:
        print(f"✗ 差分像差较大（{np.std(aberration_diff_valid):.4f} waves）。")
        print("  这可能是由于：")
        print("  1. 倾斜改变了光线在表面上的入射位置分布")
        print("  2. 导致了不同的 OPD 分布")
        print("  3. 但这不是真正的像差，而是几何效应")


if __name__ == "__main__":
    test_differential_opd_method()
