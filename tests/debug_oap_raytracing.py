"""
调试 OAP 光线追迹问题

检查 ElementRaytracer 对于 45° 倾斜 OAP 的处理是否正确
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.rays import RealRays
from wavefront_to_rays.element_raytracer import (
    ElementRaytracer, 
    SurfaceDefinition,
    compute_rotation_matrix,
)

def test_45deg_oap():
    """测试 45° 倾斜 OAP 的光线追迹"""
    
    # OAP1 参数：f=-50mm 凸面镜
    f1 = -50.0  # mm
    vertex_radius = 2 * f1  # -100 mm
    
    # 创建 SurfaceDefinition
    oap1 = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,  # 抛物面
        tilt_x=np.pi/4,  # 45° 倾斜
        tilt_y=0.0,
    )
    
    print("=" * 60)
    print("OAP1 参数:")
    print(f"  焦距: {f1} mm")
    print(f"  顶点曲率半径: {vertex_radius} mm")
    print(f"  圆锥常数: {oap1.conic}")
    print(f"  倾斜角: {np.degrees(oap1.tilt_x):.1f}°")
    print("=" * 60)
    
    # 创建光线追迹器（默认 chief_ray_direction=(0,0,1)）
    raytracer = ElementRaytracer(
        surfaces=[oap1],
        wavelength=10.64,  # μm
    )
    
    print("\n入射主光线方向:", raytracer.chief_ray_direction)
    print("出射主光线方向:", raytracer.exit_chief_direction)
    print("入射面旋转矩阵:\n", raytracer.rotation_matrix)
    print("出射面旋转矩阵:\n", raytracer.exit_rotation_matrix)
    
    # 创建简单的输入光线（在入射面局部坐标系中）
    # 5x5 网格，范围 [-10, 10] mm
    n_side = 5
    x_1d = np.linspace(-10, 10, n_side)
    y_1d = np.linspace(-10, 10, n_side)
    X, Y = np.meshgrid(x_1d, y_1d)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    n_rays = len(x_flat)
    
    # 所有光线沿局部 +Z 方向
    input_rays = RealRays(
        x=x_flat,
        y=y_flat,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, 10.64),
    )
    input_rays.opd = np.zeros(n_rays)
    
    print(f"\n输入光线数量: {n_rays}")
    print(f"输入光线位置范围: x=[{x_flat.min():.1f}, {x_flat.max():.1f}], y=[{y_flat.min():.1f}, {y_flat.max():.1f}]")
    print(f"输入光线方向: (L, M, N) = (0, 0, 1)")
    
    # 执行光线追迹
    output_rays = raytracer.trace(input_rays)
    
    # 分析输出光线
    out_x = np.asarray(output_rays.x)
    out_y = np.asarray(output_rays.y)
    out_z = np.asarray(output_rays.z)
    out_L = np.asarray(output_rays.L)
    out_M = np.asarray(output_rays.M)
    out_N = np.asarray(output_rays.N)
    
    print("\n" + "=" * 60)
    print("输出光线（出射面局部坐标系）:")
    print("=" * 60)
    print(f"位置范围: x=[{out_x.min():.3f}, {out_x.max():.3f}]")
    print(f"          y=[{out_y.min():.3f}, {out_y.max():.3f}]")
    print(f"          z=[{out_z.min():.3f}, {out_z.max():.3f}]")
    print(f"方向范围: L=[{out_L.min():.6f}, {out_L.max():.6f}]")
    print(f"          M=[{out_M.min():.6f}, {out_M.max():.6f}]")
    print(f"          N=[{out_N.min():.6f}, {out_N.max():.6f}]")
    
    # 检查中心光线
    center_idx = n_rays // 2
    print(f"\n中心光线 (index={center_idx}):")
    print(f"  输入: pos=({x_flat[center_idx]:.1f}, {y_flat[center_idx]:.1f}, 0)")
    print(f"        dir=(0, 0, 1)")
    print(f"  输出: pos=({out_x[center_idx]:.3f}, {out_y[center_idx]:.3f}, {out_z[center_idx]:.3f})")
    print(f"        dir=({out_L[center_idx]:.6f}, {out_M[center_idx]:.6f}, {out_N[center_idx]:.6f})")
    
    # 检查 OPD
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n有效光线数量: {np.sum(valid_mask)}/{n_rays}")
    print(f"OPD 范围: [{np.nanmin(opd_waves):.4f}, {np.nanmax(opd_waves):.4f}] waves")
    print(f"OPD RMS: {np.nanstd(opd_waves):.4f} waves")
    
    # 对于凸面镜（f<0），出射光线应该发散
    # 检查出射光线方向是否合理
    print("\n" + "=" * 60)
    print("物理检查:")
    print("=" * 60)
    
    # 凸面镜应该使光线发散
    # 在出射面局部坐标系中，边缘光线应该向外偏转
    edge_indices = [0, n_side-1, n_rays-n_side, n_rays-1]  # 四个角
    print("边缘光线方向（应该向外发散）:")
    for idx in edge_indices:
        print(f"  光线 {idx}: pos=({out_x[idx]:.1f}, {out_y[idx]:.1f}), dir=({out_L[idx]:.4f}, {out_M[idx]:.4f}, {out_N[idx]:.4f})")


def test_optiland_surface_setup():
    """检查 optiland 表面设置"""
    from optiland.optic import Optic
    
    f1 = -50.0  # mm
    vertex_radius = 2 * f1  # -100 mm
    
    print("\n" + "=" * 60)
    print("测试 1: 无倾斜的抛物面镜")
    print("=" * 60)
    
    # 创建 optiland 光学系统（无倾斜）
    optic1 = Optic()
    optic1.set_aperture(aperture_type='EPD', value=40.0)
    optic1.set_field_type(field_type='angle')
    optic1.add_field(y=0, x=0)
    optic1.add_wavelength(value=10.64, is_primary=True)
    
    optic1.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic1.add_surface(
        index=1,
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        conic=-1.0,
    )
    optic1.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 追迹单条光线
    from optiland.rays import RealRays
    
    test_ray1 = RealRays(
        x=np.array([0.0, 5.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([10.64, 10.64, 10.64]),
    )
    test_ray1.opd = np.array([0.0, 0.0, 0.0])
    
    surface_group1 = optic1.surface_group
    surface_group1.trace(test_ray1, skip=1)
    
    print("追迹后的光线（无倾斜）:")
    for i in range(3):
        print(f"  光线 {i}: pos=({test_ray1.x[i]:.3f}, {test_ray1.y[i]:.3f}, {test_ray1.z[i]:.3f})")
        print(f"           dir=({test_ray1.L[i]:.6f}, {test_ray1.M[i]:.6f}, {test_ray1.N[i]:.6f})")
        print(f"           OPD={test_ray1.opd[i]:.6f}")
    
    print("\n" + "=" * 60)
    print("测试 2: 带 45° 倾斜的抛物面镜")
    print("=" * 60)
    
    # 创建 optiland 光学系统（带倾斜）
    optic2 = Optic()
    optic2.set_aperture(aperture_type='EPD', value=40.0)
    optic2.set_field_type(field_type='angle')
    optic2.add_field(y=0, x=0)
    optic2.add_wavelength(value=10.64, is_primary=True)
    
    optic2.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic2.add_surface(
        index=1,
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        conic=-1.0,
        rx=np.pi/4,  # 45° 倾斜
    )
    optic2.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    test_ray2 = RealRays(
        x=np.array([0.0, 5.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([10.64, 10.64, 10.64]),
    )
    test_ray2.opd = np.array([0.0, 0.0, 0.0])
    
    surface_group2 = optic2.surface_group
    surface_group2.trace(test_ray2, skip=1)
    
    print("追迹后的光线（45° 倾斜）:")
    for i in range(3):
        print(f"  光线 {i}: pos=({test_ray2.x[i]:.3f}, {test_ray2.y[i]:.3f}, {test_ray2.z[i]:.3f})")
        print(f"           dir=({test_ray2.L[i]:.6f}, {test_ray2.M[i]:.6f}, {test_ray2.N[i]:.6f})")
        print(f"           OPD={test_ray2.opd[i]:.6f}")
    
    print("\n" + "=" * 60)
    print("测试 3: 简单平面镜 45° 倾斜")
    print("=" * 60)
    
    optic3 = Optic()
    optic3.set_aperture(aperture_type='EPD', value=40.0)
    optic3.set_field_type(field_type='angle')
    optic3.add_field(y=0, x=0)
    optic3.add_wavelength(value=10.64, is_primary=True)
    
    optic3.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic3.add_surface(
        index=1,
        radius=np.inf,  # 平面镜
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=np.pi/4,  # 45° 倾斜
    )
    optic3.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    test_ray3 = RealRays(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        L=np.array([0.0]),
        M=np.array([0.0]),
        N=np.array([1.0]),
        intensity=np.array([1.0]),
        wavelength=np.array([10.64]),
    )
    test_ray3.opd = np.array([0.0])
    
    surface_group3 = optic3.surface_group
    surface_group3.trace(test_ray3, skip=1)
    
    print("追迹后的光线（平面镜 45° 倾斜）:")
    print(f"  位置: ({test_ray3.x[0]:.6f}, {test_ray3.y[0]:.6f}, {test_ray3.z[0]:.6f})")
    print(f"  方向: ({test_ray3.L[0]:.6f}, {test_ray3.M[0]:.6f}, {test_ray3.N[0]:.6f})")
    print(f"  OPD: {test_ray3.opd[0]:.6f}")
    print(f"  预期方向: (0, -1, 0) 或 (0, 1, 0)")
    
    print("\n" + "=" * 60)
    print("测试 4: 检查 ElementRaytracer 内部创建的 optic")
    print("=" * 60)
    
    # 使用 ElementRaytracer 创建的 optic
    oap1 = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.pi/4,
    )
    
    raytracer = ElementRaytracer(
        surfaces=[oap1],
        wavelength=10.64,
    )
    
    print("ElementRaytracer 创建的 optic:")
    print(f"  表面数量: {len(raytracer.optic.surface_group.surfaces)}")
    for i, surf in enumerate(raytracer.optic.surface_group.surfaces):
        print(f"  表面 {i}:")
        print(f"    类型: {type(surf).__name__}")
        if hasattr(surf, 'geometry'):
            print(f"    几何: {type(surf.geometry).__name__}")
            if hasattr(surf.geometry, 'c'):
                print(f"    曲率: {surf.geometry.c}")
            if hasattr(surf.geometry, 'k'):
                print(f"    圆锥常数: {surf.geometry.k}")
        if hasattr(surf, 'cs'):
            print(f"    坐标系: rx={surf.cs.rx:.4f}, ry={surf.cs.ry:.4f}, rz={surf.cs.rz:.4f}")


if __name__ == "__main__":
    test_45deg_oap()
    print("\n" + "=" * 80 + "\n")
    test_optiland_surface_setup()
