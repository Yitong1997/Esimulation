"""深入分析 ElementRaytracer 的相对 OPD 计算

验证假设：对于倾斜的抛物面镜，ElementRaytracer 计算的相对 OPD
是否已经只包含像差（不包含倾斜分量）。

关键问题：
1. ElementRaytracer 使用出射面局部坐标系中最接近原点的光线作为主光线
2. 相对 OPD = 光线 OPD - 主光线 OPD
3. 如果出射面垂直于出射光轴，这个相对 OPD 应该不包含倾斜分量
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def analyze_relative_opd():
    """分析相对 OPD 的组成"""
    
    print("=" * 70)
    print("ElementRaytracer 相对 OPD 深入分析")
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
    
    # =========================================================================
    # 分析无倾斜情况
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
    
    print(f"   相对 OPD RMS: {np.nanstd(opd_waves_no_tilt):.6f} waves")
    print(f"   相对 OPD PV: {np.nanmax(opd_waves_no_tilt) - np.nanmin(opd_waves_no_tilt):.6f} waves")
    
    # =========================================================================
    # 分析有倾斜情况
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
    
    print(f"   相对 OPD RMS: {np.nanstd(opd_waves_with_tilt):.6f} waves")
    print(f"   相对 OPD PV: {np.nanmax(opd_waves_with_tilt) - np.nanmin(opd_waves_with_tilt):.6f} waves")
    
    # =========================================================================
    # 分析绝对 OPD
    # =========================================================================
    print("\n3. 绝对 OPD 分析：")
    print("-" * 50)
    
    # 获取绝对 OPD（单位：mm）
    abs_opd_no_tilt = np.asarray(rays_out_no_tilt.opd)
    abs_opd_with_tilt = np.asarray(rays_out_with_tilt.opd)
    
    print(f"   无倾斜 - 绝对 OPD 范围: [{np.min(abs_opd_no_tilt):.4f}, {np.max(abs_opd_no_tilt):.4f}] mm")
    print(f"   有倾斜 - 绝对 OPD 范围: [{np.min(abs_opd_with_tilt):.4f}, {np.max(abs_opd_with_tilt):.4f}] mm")
    
    # =========================================================================
    # 分析出射光线位置
    # =========================================================================
    print("\n4. 出射光线位置分析：")
    print("-" * 50)
    
    x_out_no_tilt = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out_no_tilt = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    
    x_out_with_tilt = np.array([float(rays_out_with_tilt.x[i]) for i in range(n_rays)])
    y_out_with_tilt = np.array([float(rays_out_with_tilt.y[i]) for i in range(n_rays)])
    
    print(f"   无倾斜 - x 范围: [{np.min(x_out_no_tilt):.4f}, {np.max(x_out_no_tilt):.4f}]")
    print(f"   无倾斜 - y 范围: [{np.min(y_out_no_tilt):.4f}, {np.max(y_out_no_tilt):.4f}]")
    print(f"   有倾斜 - x 范围: [{np.min(x_out_with_tilt):.4f}, {np.max(x_out_with_tilt):.4f}]")
    print(f"   有倾斜 - y 范围: [{np.min(y_out_with_tilt):.4f}, {np.max(y_out_with_tilt):.4f}]")
    
    # =========================================================================
    # 关键分析：检查相对 OPD 是否包含倾斜分量
    # =========================================================================
    print("\n5. 倾斜分量分析：")
    print("-" * 50)
    
    # 如果相对 OPD 包含倾斜分量，它应该与 y 坐标（或 x 坐标）线性相关
    # 对于 tilt_x 倾斜，倾斜 OPD ∝ y
    
    # 使用出射光线的 y 坐标
    y_valid = y_out_with_tilt[valid_mask_with_tilt]
    opd_valid = opd_waves_with_tilt[valid_mask_with_tilt]
    
    # 线性拟合：OPD = a * y + b
    if len(y_valid) > 2:
        coeffs = np.polyfit(y_valid, opd_valid, 1)
        tilt_coeff = coeffs[0]  # waves/mm
        
        print(f"   OPD vs y 线性拟合斜率: {tilt_coeff:.4f} waves/mm")
        
        # 计算倾斜分量
        tilt_component = tilt_coeff * y_valid
        residual = opd_valid - tilt_component - coeffs[1]
        
        print(f"   倾斜分量 RMS: {np.std(tilt_component):.4f} waves")
        print(f"   残差（去除倾斜后）RMS: {np.std(residual):.6f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    print("""
问题分析：
1. ElementRaytracer 的相对 OPD 确实包含了倾斜分量
2. 这是因为 get_relative_opd_waves() 使用出射面局部坐标系中
   最接近原点的光线作为主光线
3. 但是，对于倾斜的表面，出射光线的分布不再以原点为中心
4. 因此，选择的"主光线"可能不是真正的主光线（入射时 x=0, y=0 的光线）

可能的解决方案：
1. 修改 get_relative_opd_waves() 使用入射时 x=0, y=0 的光线作为主光线
2. 或者，在 _apply_element_hybrid 中直接使用 ElementRaytracer 的
   相对 OPD，不再减去理想 OPD（因为相对 OPD 已经是相对于主光线的）
""")


if __name__ == "__main__":
    analyze_relative_opd()
