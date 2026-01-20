"""分析倾斜抛物面镜的表面交点

核心问题：
- 对于倾斜的抛物面镜，入射光线在表面上的交点位置与无倾斜时完全不同
- 这导致 OPD 分布也完全不同
- 需要理解这种差异的物理意义

关键假设验证：
- 抛物面镜对于沿光轴入射的平行光是无像差的
- 但当抛物面镜倾斜后，入射光不再沿光轴入射
- 这会引入像差吗？

物理分析：
- 抛物面镜的无像差特性是针对"从无穷远来的轴上点源"
- 当镜面倾斜时，入射光相对于镜面的光轴有一个角度
- 这相当于"离轴点源"，会引入彗差等像差
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def analyze_surface_intersection():
    """分析表面交点"""
    
    print("=" * 70)
    print("倾斜抛物面镜表面交点分析")
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
        semi_aperture=15.0,  # 增大半口径以容纳倾斜后的光束
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
    # 追迹不带倾斜的表面
    # =========================================================================
    print("\n1. 不带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    
    # 获取表面交点（从 optiland 内部数据）
    # 注意：ElementRaytracer 不直接提供表面交点，需要从 optiland 获取
    # 这里我们通过分析入射和出射光线来推断
    
    # 获取出射光线数据
    x_out_no_tilt = np.array([float(rays_out_no_tilt.x[i]) for i in range(n_rays)])
    y_out_no_tilt = np.array([float(rays_out_no_tilt.y[i]) for i in range(n_rays)])
    L_out_no_tilt = np.array([float(rays_out_no_tilt.L[i]) for i in range(n_rays)])
    M_out_no_tilt = np.array([float(rays_out_no_tilt.M[i]) for i in range(n_rays)])
    N_out_no_tilt = np.array([float(rays_out_no_tilt.N[i]) for i in range(n_rays)])
    
    print(f"   出射光线方向（中心光线）：")
    center_idx = n_rays // 2
    print(f"     L={L_out_no_tilt[center_idx]:.6f}, M={M_out_no_tilt[center_idx]:.6f}, N={N_out_no_tilt[center_idx]:.6f}")
    
    # =========================================================================
    # 追迹带倾斜的表面
    # =========================================================================
    print("\n2. 带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    
    # 获取出射光线数据
    x_out_with_tilt = np.array([float(rays_out_with_tilt.x[i]) for i in range(n_rays)])
    y_out_with_tilt = np.array([float(rays_out_with_tilt.y[i]) for i in range(n_rays)])
    L_out_with_tilt = np.array([float(rays_out_with_tilt.L[i]) for i in range(n_rays)])
    M_out_with_tilt = np.array([float(rays_out_with_tilt.M[i]) for i in range(n_rays)])
    N_out_with_tilt = np.array([float(rays_out_with_tilt.N[i]) for i in range(n_rays)])
    
    print(f"   出射光线方向（中心光线）：")
    print(f"     L={L_out_with_tilt[center_idx]:.6f}, M={M_out_with_tilt[center_idx]:.6f}, N={N_out_with_tilt[center_idx]:.6f}")
    
    # =========================================================================
    # 分析出射光线方向的变化
    # =========================================================================
    print("\n3. 出射光线方向分析：")
    print("-" * 50)
    
    # 对于无倾斜的抛物面镜，所有出射光线应该汇聚到焦点
    # 检查出射光线方向的一致性
    
    # 计算出射光线与 z 轴的夹角
    angle_no_tilt = np.arctan2(np.sqrt(L_out_no_tilt**2 + M_out_no_tilt**2), N_out_no_tilt)
    angle_with_tilt = np.arctan2(np.sqrt(L_out_with_tilt**2 + M_out_with_tilt**2), N_out_with_tilt)
    
    print(f"   无倾斜：出射角度范围 {np.min(angle_no_tilt)*180/np.pi:.2f}° ~ {np.max(angle_no_tilt)*180/np.pi:.2f}°")
    print(f"   有倾斜：出射角度范围 {np.min(angle_with_tilt)*180/np.pi:.2f}° ~ {np.max(angle_with_tilt)*180/np.pi:.2f}°")
    
    # =========================================================================
    # 关键分析：倾斜后的光束是否仍然汇聚到一点？
    # =========================================================================
    print("\n4. 焦点分析：")
    print("-" * 50)
    
    # 对于无倾斜的抛物面镜，所有光线应该汇聚到 (0, 0, f)
    # 对于有倾斜的抛物面镜，光线应该汇聚到旋转后的焦点位置
    
    # 计算光线与 z=f 平面的交点（无倾斜）
    t_no_tilt = (focal_length - 0) / N_out_no_tilt  # 假设出射面在 z=0
    x_focus_no_tilt = x_out_no_tilt + L_out_no_tilt * t_no_tilt
    y_focus_no_tilt = y_out_no_tilt + M_out_no_tilt * t_no_tilt
    
    print(f"   无倾斜焦点位置（z={focal_length}mm 平面）：")
    print(f"     x 范围: {np.min(x_focus_no_tilt):.6f} ~ {np.max(x_focus_no_tilt):.6f} mm")
    print(f"     y 范围: {np.min(y_focus_no_tilt):.6f} ~ {np.max(y_focus_no_tilt):.6f} mm")
    print(f"     焦点散布: {np.std(x_focus_no_tilt):.6f} mm (x), {np.std(y_focus_no_tilt):.6f} mm (y)")
    
    # 对于有倾斜的情况，焦点位置会改变
    # 45° 倾斜后，焦点应该在 (0, -f, 0) 附近
    # 计算光线与 y=-f 平面的交点
    t_with_tilt = (-focal_length - y_out_with_tilt) / M_out_with_tilt
    x_focus_with_tilt = x_out_with_tilt + L_out_with_tilt * t_with_tilt
    z_focus_with_tilt = 0 + N_out_with_tilt * t_with_tilt  # 假设出射面在 z=0
    
    print(f"\n   有倾斜焦点位置（y={-focal_length}mm 平面）：")
    print(f"     x 范围: {np.min(x_focus_with_tilt):.6f} ~ {np.max(x_focus_with_tilt):.6f} mm")
    print(f"     z 范围: {np.min(z_focus_with_tilt):.6f} ~ {np.max(z_focus_with_tilt):.6f} mm")
    print(f"     焦点散布: {np.std(x_focus_with_tilt):.6f} mm (x), {np.std(z_focus_with_tilt):.6f} mm (z)")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    focus_spread_no_tilt = np.sqrt(np.std(x_focus_no_tilt)**2 + np.std(y_focus_no_tilt)**2)
    focus_spread_with_tilt = np.sqrt(np.std(x_focus_with_tilt)**2 + np.std(z_focus_with_tilt)**2)
    
    print(f"   无倾斜焦点散布: {focus_spread_no_tilt:.6f} mm")
    print(f"   有倾斜焦点散布: {focus_spread_with_tilt:.6f} mm")
    
    if focus_spread_with_tilt < 0.001:
        print("\n✓ 倾斜后的抛物面镜仍然是无像差的（焦点散布 < 1μm）")
        print("  问题可能在于 OPD 计算方法，而不是光学系统本身")
    else:
        print(f"\n✗ 倾斜后的抛物面镜有像差（焦点散布 = {focus_spread_with_tilt:.3f} mm）")
        print("  这可能是因为入射光相对于镜面光轴有角度")


if __name__ == "__main__":
    analyze_surface_intersection()
