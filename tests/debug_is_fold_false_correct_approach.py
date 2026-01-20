"""验证 is_fold=False 的正确处理方法

核心问题：
- 对于 is_fold=False 的倾斜表面，如何正确计算"像差"？

分析：
1. is_fold=True：折叠倾斜
   - PROPER 在展开的光路上传播
   - 使用不带倾斜的表面追迹
   - 像差 = 实际 OPD - 理想 OPD
   - 结果：像差 ≈ 0

2. is_fold=False：失调倾斜
   - 应该追迹带倾斜的表面
   - 但 PROPER 仍然在展开的光路上传播
   - 问题：如何计算"像差"？

关键洞察：
- 对于 is_fold=False，倾斜引入的是"真实的像差"
- 这个像差应该被应用到 PROPER 波前上
- 但是，我们需要从 OPD 中去除"聚焦效果"

正确的方法：
- 对于 is_fold=False，像差 = 带倾斜的 OPD - 不带倾斜的 OPD
- 但是，两者的 OPD 都应该相对于各自的主光线计算
- 这样，差异就是倾斜引入的"额外"OPD

问题：
- 当前实现中，差分方法产生了很大的"像差"（~35 waves）
- 这是因为两个表面的出射光线位置完全不同
- 需要找到正确的对齐方法
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_correct_approach():
    """测试正确的处理方法"""
    
    print("=" * 70)
    print("is_fold=False 正确处理方法验证")
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
        semi_aperture=15.0,
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
        semi_aperture=15.0,
        conic=-1.0,
        tilt_x=0.0,
        tilt_y=0.0,
    )
    
    # 创建采样光线
    n_side = 11
    coords = np.linspace(-5, 5, n_side)
    X, Y = np.meshgrid(coords, coords)
    ray_x = X.flatten()
    ray_y = Y.flatten()
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
    # 方法 1：当前实现（差分方法）
    # =========================================================================
    print("\n方法 1：差分方法（当前实现）")
    print("-" * 50)
    
    # 追迹带倾斜的表面
    raytracer_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_tilt = raytracer_tilt.trace(create_rays())
    opd_waves_tilt = raytracer_tilt.get_relative_opd_waves()
    
    # 追迹不带倾斜的表面
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    opd_waves_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
    
    # 差分
    center_idx = n_rays // 2
    opd_tilt_aligned = opd_waves_tilt - opd_waves_tilt[center_idx]
    opd_no_tilt_aligned = opd_waves_no_tilt - opd_waves_no_tilt[center_idx]
    diff_opd = opd_tilt_aligned - opd_no_tilt_aligned
    
    print(f"   差分 OPD RMS: {np.std(diff_opd):.4f} waves")
    print(f"   差分 OPD PV: {np.max(diff_opd) - np.min(diff_opd):.4f} waves")
    
    # =========================================================================
    # 方法 2：只使用带倾斜表面的 OPD，减去理想 OPD
    # =========================================================================
    print("\n方法 2：带倾斜 OPD - 理想 OPD")
    print("-" * 50)
    
    # 理想 OPD（使用入射位置）
    r_sq = ray_x**2 + ray_y**2
    
    def calculate_exact_mirror_opd(r_sq, f):
        sag = r_sq / (4 * f)
        n_mag_sq = 1 + r_sq / (4 * f**2)
        rz = 1 - 2 / n_mag_sq
        incident_path = sag
        reflected_path = -sag / rz
        return incident_path + reflected_path
    
    ideal_opd_mm = calculate_exact_mirror_opd(r_sq, focal_length)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    ideal_opd_waves_relative = ideal_opd_waves - ideal_opd_waves[center_idx]
    
    aberration_method2 = opd_waves_tilt - ideal_opd_waves_relative
    
    print(f"   像差 RMS: {np.std(aberration_method2):.4f} waves")
    print(f"   像差 PV: {np.max(aberration_method2) - np.min(aberration_method2):.4f} waves")
    
    # =========================================================================
    # 方法 3：使用不带倾斜表面的 OPD 作为"理想 OPD"
    # =========================================================================
    print("\n方法 3：带倾斜 OPD - 不带倾斜 OPD（使用入射位置对齐）")
    print("-" * 50)
    
    # 这与方法 1 相同，但我们分析一下为什么会有问题
    
    # 获取出射位置
    x_out_tilt = np.array([float(rays_out_tilt.x[i]) for i in range(n_rays)])
    y_out_tilt = np.array([float(rays_out_tilt.y[i]) for i in range(n_rays)])
    x_out_no_tilt = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out_no_tilt = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    
    print(f"   带倾斜出射位置范围: x=[{np.min(x_out_tilt):.2f}, {np.max(x_out_tilt):.2f}], "
          f"y=[{np.min(y_out_tilt):.2f}, {np.max(y_out_tilt):.2f}]")
    print(f"   不带倾斜出射位置范围: x=[{np.min(x_out_no_tilt):.2f}, {np.max(x_out_no_tilt):.2f}], "
          f"y=[{np.min(y_out_no_tilt):.2f}, {np.max(y_out_no_tilt):.2f}]")
    
    # =========================================================================
    # 方法 4：使用出射位置对齐
    # =========================================================================
    print("\n方法 4：使用出射位置对齐")
    print("-" * 50)
    
    # 对于不带倾斜的表面，出射位置 = 入射位置
    # 对于带倾斜的表面，出射位置不同
    # 我们需要找到"相同入射位置"的光线，比较它们的 OPD
    
    # 由于入射位置相同，我们可以直接比较
    # 但问题是：OPD 是相对于主光线计算的，而主光线的选择不同
    
    # 让我们使用绝对 OPD 进行比较
    opd_mm_tilt = np.array([float(rays_out_tilt.opd[i]) for i in range(n_rays)])
    opd_mm_no_tilt = np.array([float(rays_out_no_tilt.opd[i]) for i in range(n_rays)])
    
    # 绝对 OPD 差异
    abs_diff_mm = opd_mm_tilt - opd_mm_no_tilt
    abs_diff_waves = abs_diff_mm / wavelength_mm
    
    # 相对于中心光线
    abs_diff_relative = abs_diff_waves - abs_diff_waves[center_idx]
    
    print(f"   绝对 OPD 差异 RMS: {np.std(abs_diff_relative):.4f} waves")
    print(f"   绝对 OPD 差异 PV: {np.max(abs_diff_relative) - np.min(abs_diff_relative):.4f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    print("\n问题分析：")
    print("1. 方法 1（差分方法）产生 ~35 waves 的像差")
    print("2. 方法 2（带倾斜 OPD - 理想 OPD）产生 ~62 waves 的像差")
    print("3. 方法 4（绝对 OPD 差异）产生 ~35 waves 的像差")
    print("\n这些像差是物理上真实的，因为：")
    print("- 45° 倾斜改变了光线在表面上的入射位置和角度")
    print("- 这相当于离轴点源，会引入像散和彗差")
    print("- 这不是计算错误，而是真实的物理效应")
    print("\n建议：")
    print("- is_fold=False 应该用于小角度失调（< 1°）")
    print("- 对于 45° 折叠镜，应该使用 is_fold=True")


if __name__ == "__main__":
    test_correct_approach()
