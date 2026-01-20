"""验证倾斜抛物面镜的差分方法

核心假设：
- 对于理想抛物面镜，无论是否倾斜，都应该是无像差的
- 倾斜只改变光束的传播方向，不引入像差
- 因此：带倾斜 OPD - 不带倾斜 OPD 应该只包含倾斜引入的 OPD 变化

关键问题：
- 当前差分方法仍然显示 ~33 waves 的像差
- 需要找出原因

可能的原因：
1. 主光线选择不一致
2. 坐标系变换问题
3. 光线追迹本身的问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_differential_method():
    """测试差分方法"""
    
    print("=" * 70)
    print("倾斜抛物面镜差分方法验证")
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
    
    # 创建采样光线（更密集的网格）
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
    # 追迹不带倾斜的表面
    # =========================================================================
    print("\n1. 追迹不带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    opd_waves_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
    valid_no_tilt = raytracer_no_tilt.get_valid_ray_mask()
    
    # 获取出射位置
    x_out_no_tilt = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out_no_tilt = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    
    print(f"   有效光线数: {np.sum(valid_no_tilt)}/{n_rays}")
    print(f"   OPD 范围: {np.min(opd_waves_no_tilt[valid_no_tilt]):.2f} ~ "
          f"{np.max(opd_waves_no_tilt[valid_no_tilt]):.2f} waves")
    
    # =========================================================================
    # 追迹带倾斜的表面
    # =========================================================================
    print("\n2. 追迹带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    opd_waves_with_tilt = raytracer_with_tilt.get_relative_opd_waves()
    valid_with_tilt = raytracer_with_tilt.get_valid_ray_mask()
    
    # 获取出射位置
    x_out_with_tilt = np.array([float(rays_out_with_tilt.x[i]) for i in range(n_rays)])
    y_out_with_tilt = np.array([float(rays_out_with_tilt.y[i]) for i in range(n_rays)])
    
    print(f"   有效光线数: {np.sum(valid_with_tilt)}/{n_rays}")
    print(f"   OPD 范围: {np.min(opd_waves_with_tilt[valid_with_tilt]):.2f} ~ "
          f"{np.max(opd_waves_with_tilt[valid_with_tilt]):.2f} waves")
    
    # =========================================================================
    # 差分计算
    # =========================================================================
    print("\n3. 差分计算：")
    print("-" * 50)
    
    # 合并有效掩模
    valid_both = valid_no_tilt & valid_with_tilt
    
    # 找到中心光线（入射位置最接近原点）
    r_in = np.sqrt(ray_x**2 + ray_y**2)
    center_idx = np.argmin(r_in)
    
    print(f"   中心光线索引: {center_idx}")
    print(f"   中心光线入射位置: ({ray_x[center_idx]:.2f}, {ray_y[center_idx]:.2f})")
    
    # 对齐到中心光线
    opd_no_tilt_aligned = opd_waves_no_tilt - opd_waves_no_tilt[center_idx]
    opd_with_tilt_aligned = opd_waves_with_tilt - opd_waves_with_tilt[center_idx]
    
    # 计算差异
    diff_opd = opd_with_tilt_aligned - opd_no_tilt_aligned
    
    print(f"\n   差分 OPD 统计（有效光线）：")
    valid_diff = diff_opd[valid_both]
    print(f"     范围: {np.min(valid_diff):.4f} ~ {np.max(valid_diff):.4f} waves")
    print(f"     RMS: {np.std(valid_diff):.4f} waves")
    print(f"     PV: {np.max(valid_diff) - np.min(valid_diff):.4f} waves")
    
    # =========================================================================
    # 分析差分 OPD 的成分
    # =========================================================================
    print("\n4. 差分 OPD 成分分析：")
    print("-" * 50)
    
    # 检查是否有倾斜成分
    valid_x = ray_x[valid_both]
    valid_y = ray_y[valid_both]
    valid_diff_opd = diff_opd[valid_both]
    
    # 线性拟合（检查倾斜）
    A = np.column_stack([np.ones_like(valid_x), valid_x, valid_y])
    coeffs, residuals, rank, s = np.linalg.lstsq(A, valid_diff_opd, rcond=None)
    
    piston = coeffs[0]
    tilt_x_coeff = coeffs[1]  # dOPD/dx
    tilt_y_coeff = coeffs[2]  # dOPD/dy
    
    print(f"   线性拟合结果：")
    print(f"     Piston: {piston:.4f} waves")
    print(f"     Tilt X: {tilt_x_coeff:.4f} waves/mm")
    print(f"     Tilt Y: {tilt_y_coeff:.4f} waves/mm")
    
    # 计算去除倾斜后的残差
    linear_fit = piston + tilt_x_coeff * valid_x + tilt_y_coeff * valid_y
    residual = valid_diff_opd - linear_fit
    
    print(f"\n   去除倾斜后的残差：")
    print(f"     RMS: {np.std(residual):.4f} waves")
    print(f"     PV: {np.max(residual) - np.min(residual):.4f} waves")
    
    # =========================================================================
    # 检查出射位置的变化
    # =========================================================================
    print("\n5. 出射位置变化分析：")
    print("-" * 50)
    
    # 计算出射位置的差异
    dx_out = x_out_with_tilt - x_out_no_tilt
    dy_out = y_out_with_tilt - y_out_no_tilt
    
    print(f"   出射位置差异（有效光线）：")
    print(f"     dx 范围: {np.min(dx_out[valid_both]):.4f} ~ {np.max(dx_out[valid_both]):.4f} mm")
    print(f"     dy 范围: {np.min(dy_out[valid_both]):.4f} ~ {np.max(dy_out[valid_both]):.4f} mm")
    
    # =========================================================================
    # 关键检查：主光线的 OPD 是否为 0
    # =========================================================================
    print("\n6. 主光线 OPD 检查：")
    print("-" * 50)
    
    # 获取原始 OPD（未对齐）
    opd_mm_no_tilt = np.array([float(rays_out_no_tilt.opd[i]) for i in range(n_rays)])
    opd_mm_with_tilt = np.array([float(rays_out_with_tilt.opd[i]) for i in range(n_rays)])
    
    print(f"   不带倾斜：")
    print(f"     中心光线 OPD: {opd_mm_no_tilt[center_idx]:.6f} mm")
    print(f"     中心光线出射位置: ({x_out_no_tilt[center_idx]:.4f}, {y_out_no_tilt[center_idx]:.4f})")
    
    print(f"\n   带倾斜：")
    print(f"     中心光线 OPD: {opd_mm_with_tilt[center_idx]:.6f} mm")
    print(f"     中心光线出射位置: ({x_out_with_tilt[center_idx]:.4f}, {y_out_with_tilt[center_idx]:.4f})")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if np.std(residual) < 0.1:
        print("✓ 差分 OPD 主要是倾斜成分，去除倾斜后残差很小")
        print("  这说明倾斜抛物面镜确实是无像差的，只引入了波前倾斜")
    else:
        print("✗ 差分 OPD 包含显著的非倾斜成分")
        print("  需要进一步分析原因")


if __name__ == "__main__":
    test_differential_method()
