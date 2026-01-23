"""
定位真正的问题

已确认：
1. optiland 的反射方向计算是正确的
2. ElementRaytracer._compute_exit_chief_direction 计算也是正确的
3. 两者一致

那么问题出在哪里？

回顾之前的发现：
- 5° 倾斜时，出射 z 坐标不为 0（应该在出射面上）
- OPD 与 z_out 高度相关
- 出射光线方向在出射面坐标系中是 (0, 0, 1)，这是正确的

让我重新分析问题...
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import warnings
warnings.filterwarnings('ignore')


def analyze_real_problem(tilt_deg: float):
    """分析真正的问题"""
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import (
        ElementRaytracer, SurfaceDefinition, compute_rotation_matrix
    )
    from optiland.rays import RealRays
    
    print(f"\n{'='*70}")
    print(f"分析真正的问题: {tilt_deg}°")
    print(f"{'='*70}")
    
    wavelength_um = 0.633
    tilt_rad = np.radians(tilt_deg)
    
    # 计算出射方向
    d_in = np.array([0, 0, 1])
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)
    
    print(f"\n入射方向: {d_in}")
    print(f"出射方向: {d_out}")
    
    # 创建简单的输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 5.0, -5.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 5),
    )
    input_rays.opd = np.zeros(5)
    
    # 创建表面定义
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=30.0,
        conic=0.0,
        tilt_x=tilt_rad,
        tilt_y=0.0,
    )
    
    # 创建光线追迹器
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=tuple(d_out),
    )
    
    # 手动执行追迹过程
    
    # 1. 复制光线
    traced_rays = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    traced_rays.opd = np.asarray(input_rays.opd).copy()
    
    print(f"\n--- 追迹前（入射面局部坐标系）---")
    print(f"  光线 0: pos=({traced_rays.x[0]:.4f}, {traced_rays.y[0]:.4f}, {traced_rays.z[0]:.4f})")
    print(f"  光线 3: pos=({traced_rays.x[3]:.4f}, {traced_rays.y[3]:.4f}, {traced_rays.z[3]:.4f})")
    print(f"  光线 4: pos=({traced_rays.x[4]:.4f}, {traced_rays.y[4]:.4f}, {traced_rays.z[4]:.4f})")
    
    # 2. 调用 _trace_with_signed_opd
    raytracer._trace_with_signed_opd(traced_rays, skip=1)
    
    print(f"\n--- optiland 追迹后（入射面局部坐标系）---")
    print(f"  光线 0: pos=({traced_rays.x[0]:.4f}, {traced_rays.y[0]:.4f}, {traced_rays.z[0]:.4f}), dir=({traced_rays.L[0]:.4f}, {traced_rays.M[0]:.4f}, {traced_rays.N[0]:.4f}), OPD={traced_rays.opd[0]:.6f}")
    print(f"  光线 3: pos=({traced_rays.x[3]:.4f}, {traced_rays.y[3]:.4f}, {traced_rays.z[3]:.4f}), dir=({traced_rays.L[3]:.4f}, {traced_rays.M[3]:.4f}, {traced_rays.N[3]:.4f}), OPD={traced_rays.opd[3]:.6f}")
    print(f"  光线 4: pos=({traced_rays.x[4]:.4f}, {traced_rays.y[4]:.4f}, {traced_rays.z[4]:.4f}), dir=({traced_rays.L[4]:.4f}, {traced_rays.M[4]:.4f}, {traced_rays.N[4]:.4f}), OPD={traced_rays.opd[4]:.6f}")
    
    # 分析：光线 3 和 4 在 y 方向偏移 ±5mm
    # 对于倾斜的平面镜，这些光线应该在镜面上的不同位置反射
    # 反射后，它们应该在出射面上的不同位置
    
    # 关键问题：optiland 追迹后，光线位置是在哪个坐标系中？
    # 根据 optiland 的设计，追迹后的位置应该是光线与最后一个表面的交点
    # 在我们的设置中，最后一个表面是出射面（透明平面）
    
    # 但是，出射面是倾斜的！
    # 出射面的法向量是 d_out（出射方向）
    # 出射面上的点满足：d_out · (r - r0) = 0
    
    # 让我检查光线是否在出射面上
    x_traced = np.asarray(traced_rays.x)
    y_traced = np.asarray(traced_rays.y)
    z_traced = np.asarray(traced_rays.z)
    
    # 出射方向在入射面局部坐标系中
    d_out_local = raytracer.rotation_matrix.T @ d_out
    print(f"\n出射方向（入射面局部坐标系）: {d_out_local}")
    
    # 检查光线是否在出射面上
    # 出射面方程：d_out_local · r = 0（假设出射面过原点）
    distance_to_exit_plane = d_out_local[0]*x_traced + d_out_local[1]*y_traced + d_out_local[2]*z_traced
    print(f"\n光线到出射面的距离:")
    for i in range(5):
        print(f"  光线 {i}: {distance_to_exit_plane[i]:.6f} mm")
    
    # 3. 坐标变换到出射面局部坐标系
    R_entrance_to_exit = raytracer.exit_rotation_matrix.T @ raytracer.rotation_matrix
    
    pos_entrance = np.stack([x_traced, y_traced, z_traced], axis=0)
    pos_exit = R_entrance_to_exit @ pos_entrance
    
    print(f"\n--- 坐标变换后（出射面局部坐标系）---")
    print(f"  光线 0: pos=({pos_exit[0,0]:.4f}, {pos_exit[1,0]:.4f}, {pos_exit[2,0]:.4f})")
    print(f"  光线 3: pos=({pos_exit[0,3]:.4f}, {pos_exit[1,3]:.4f}, {pos_exit[2,3]:.4f})")
    print(f"  光线 4: pos=({pos_exit[0,4]:.4f}, {pos_exit[1,4]:.4f}, {pos_exit[2,4]:.4f})")
    
    # 关键问题：为什么 z_exit 不为 0？
    # 
    # 原因分析：
    # 1. optiland 追迹后，光线位置是光线与出射面的交点
    # 2. 但是，这个交点是在入射面局部坐标系中表示的
    # 3. 坐标变换 R_entrance_to_exit 只是旋转，没有平移
    # 4. 如果出射面不过原点，那么变换后 z_exit 不会为 0
    
    # 让我检查出射面是否过原点
    # 出射面应该过主光线与镜面的交点
    # 对于正入射到倾斜平面镜，主光线与镜面的交点就是原点
    # 所以出射面应该过原点
    
    # 但是，optiland 的出射面定义可能不同...
    # 让我检查 optiland 的表面定义
    
    print(f"\n--- optiland 表面分析 ---")
    surfaces = raytracer.optic.surface_group.surfaces
    for i, surf in enumerate(surfaces):
        print(f"  表面 {i}: type={type(surf).__name__}")
        if hasattr(surf, 'geometry'):
            geom = surf.geometry
            print(f"    geometry: {type(geom).__name__}")
            if hasattr(geom, 'cs'):
                cs = geom.cs
                print(f"    cs.rx: {np.degrees(cs.rx):.2f}°")
                print(f"    cs.ry: {np.degrees(cs.ry):.2f}°")
                print(f"    cs.x: {cs.x}")
                print(f"    cs.y: {cs.y}")
                print(f"    cs.z: {cs.z}")


def main():
    print("=" * 70)
    print("定位真正的问题")
    print("=" * 70)
    
    for angle in [5, 45]:
        analyze_real_problem(angle)


if __name__ == "__main__":
    main()
