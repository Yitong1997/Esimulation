"""
验证出射面修复方案

问题：当出射方向的 N 分量为负时，_direction_to_rotation_angles 计算的
ry = 180°，导致 optiland 的表面法向量与出射方向不一致。

解决方案：修改角度计算方式，确保表面法向量与出射方向一致。

关键洞察：
- optiland 的表面法向量是 (0, 0, 1) 经过旋转后的结果
- 我们需要的是出射面垂直于出射方向
- 但 optiland 的表面法向量应该指向光线来的方向（入射侧）
- 对于出射面，光线从反射镜来，所以法向量应该与出射方向相反！

等等，让我重新思考...

对于透明平面（material='air'），光线直接穿过，不发生反射或折射。
optiland 计算光线与平面的交点时，使用的是平面方程 n · (r - r0) = 0。
法向量的方向不影响交点计算，只影响法向量的符号。

那么问题出在哪里？

让我仔细检查 optiland 的光线追迹逻辑...
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


def test_optiland_plane_intersection():
    """测试 optiland 的平面交点计算"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print("=" * 70)
    print("测试 optiland 的平面交点计算")
    print("=" * 70)
    
    # 测试：光线沿 (0, 0.17, -0.98) 方向传播，与 rx=10° 的平面相交
    # 平面法向量（optiland 定义）：(0, -0.17, 0.98)
    # 注意：optiland 的法向量指向 +Z 方向（旋转前）
    
    # 创建系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.633, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 添加 rx=10° 的透明平面
    rx_deg = 10
    rx_rad = np.radians(rx_deg)
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='air',
        is_stop=True,
        rx=rx_rad,
        ry=0.0,
    )
    
    # 创建沿出射方向传播的光线
    # 出射方向：(0, sin(10°), -cos(10°)) = (0, 0.17, -0.98)
    d_out = np.array([0, np.sin(rx_rad), -np.cos(rx_rad)])
    
    rays = RealRays(
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 5.0]),
        z=np.array([0.0, 0.0]),
        L=np.array([d_out[0], d_out[0]]),
        M=np.array([d_out[1], d_out[1]]),
        N=np.array([d_out[2], d_out[2]]),
        intensity=np.array([1.0, 1.0]),
        wavelength=np.array([0.633, 0.633]),
    )
    
    print(f"\n光线方向: {d_out}")
    print(f"平面 rx: {rx_deg}°")
    
    # 计算平面法向量
    c, s = np.cos(rx_rad), np.sin(rx_rad)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    n_plane = Rx @ np.array([0, 0, 1])
    print(f"平面法向量: {n_plane}")
    
    # 光线方向与平面法向量的点积
    dot = np.dot(d_out, n_plane)
    print(f"光线方向 · 平面法向量 = {dot:.6f}")
    
    if abs(dot) < 1e-6:
        print("⚠️ 光线与平面平行！无法计算交点")
    else:
        # 手动计算交点
        # 光线方程：r = r0 + t * d
        # 平面方程：n · r = 0（平面过原点）
        # 代入：n · (r0 + t * d) = 0
        # t = -n · r0 / (n · d)
        
        for i, (x0, y0, z0) in enumerate([(0, 0, 0), (0, 5, 0)]):
            r0 = np.array([x0, y0, z0])
            t = -np.dot(n_plane, r0) / dot
            intersection = r0 + t * d_out
            print(f"\n光线 {i}:")
            print(f"  起点: {r0}")
            print(f"  t = {t:.6f}")
            print(f"  交点（手动计算）: {intersection}")
    
    # optiland 追迹
    optic.surface_group.trace(rays, skip=1)
    
    print(f"\noptiland 追迹结果:")
    print(f"  光线 0: pos=({rays.x[0]:.6f}, {rays.y[0]:.6f}, {rays.z[0]:.6f})")
    print(f"  光线 1: pos=({rays.x[1]:.6f}, {rays.y[1]:.6f}, {rays.z[1]:.6f})")


def test_correct_exit_surface_definition():
    """测试正确的出射面定义"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print("\n" + "=" * 70)
    print("测试正确的出射面定义")
    print("=" * 70)
    
    # 对于 5° 倾斜的平面镜：
    # - 出射方向：(0, sin(10°), -cos(10°))
    # - 出射面应该垂直于出射方向
    # - 出射面法向量应该与出射方向平行（或反平行）
    
    # 问题：如何定义一个法向量为 (0, 0.17, -0.98) 的平面？
    # 
    # optiland 的平面法向量是 (0, 0, 1) 经过旋转后的结果
    # 我们需要找到 rx, ry 使得旋转后的法向量为 (0, 0.17, -0.98)
    # 
    # 但是，(0, 0, 1) 经过任何旋转都不能变成 (0, 0.17, -0.98)
    # 因为 (0, 0, 1) 的模为 1，旋转后模不变
    # 而 (0, 0.17, -0.98) 的模也是 1，所以理论上可以
    # 
    # 让我们找到正确的旋转角度...
    
    # 目标法向量
    n_target = np.array([0, np.sin(np.radians(10)), -np.cos(np.radians(10))])
    print(f"\n目标法向量: {n_target}")
    
    # 方法 1：使用 rx 和 ry
    # (0, 0, 1) 绕 X 轴旋转 rx 后：(0, -sin(rx), cos(rx))
    # 再绕 Y 轴旋转 ry 后：(sin(ry)*cos(rx), -sin(rx), cos(ry)*cos(rx))
    # 
    # 要使结果为 (0, sin(10°), -cos(10°))：
    # sin(ry)*cos(rx) = 0  => ry = 0 或 rx = 90°
    # -sin(rx) = sin(10°)  => rx = -10°
    # cos(ry)*cos(rx) = -cos(10°)  => cos(ry) = -1 => ry = 180°
    # 
    # 但是 rx = -10° 和 ry = 180° 给出：
    # (sin(180°)*cos(-10°), -sin(-10°), cos(180°)*cos(-10°))
    # = (0, sin(10°), -cos(10°))
    # 
    # 这正是我们想要的！
    
    rx_correct = np.radians(-10)
    ry_correct = np.radians(180)
    
    c, s = np.cos(rx_correct), np.sin(rx_correct)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    c, s = np.cos(ry_correct), np.sin(ry_correct)
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    
    n_computed = Ry @ Rx @ np.array([0, 0, 1])
    print(f"rx={np.degrees(rx_correct):.1f}°, ry={np.degrees(ry_correct):.1f}° 给出的法向量: {n_computed}")
    print(f"与目标匹配: {np.allclose(n_computed, n_target)}")
    
    # 测试这个定义
    print("\n--- 测试正确的出射面定义 ---")
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.633, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='air',
        is_stop=True,
        rx=rx_correct,
        ry=ry_correct,
    )
    
    # 创建沿出射方向传播的光线
    d_out = n_target  # 出射方向与目标法向量相同
    
    rays = RealRays(
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 5.0]),
        z=np.array([0.0, 0.0]),
        L=np.array([d_out[0], d_out[0]]),
        M=np.array([d_out[1], d_out[1]]),
        N=np.array([d_out[2], d_out[2]]),
        intensity=np.array([1.0, 1.0]),
        wavelength=np.array([0.633, 0.633]),
    )
    
    optic.surface_group.trace(rays, skip=1)
    
    print(f"光线方向: {d_out}")
    print(f"optiland 追迹结果:")
    print(f"  光线 0: pos=({rays.x[0]:.6f}, {rays.y[0]:.6f}, {rays.z[0]:.6f})")
    print(f"  光线 1: pos=({rays.x[1]:.6f}, {rays.y[1]:.6f}, {rays.z[1]:.6f})")
    
    # 检查光线是否在出射面上
    # 出射面方程：n_target · r = 0
    for i in range(2):
        r = np.array([rays.x[i], rays.y[i], rays.z[i]])
        distance = np.dot(n_target, r)
        print(f"  光线 {i} 到出射面的距离: {distance:.6f}")


def derive_correct_angles():
    """推导正确的角度计算公式"""
    print("\n" + "=" * 70)
    print("推导正确的角度计算公式")
    print("=" * 70)
    
    print("""
问题：给定出射方向 d = (L, M, N)，找到 rx, ry 使得
optiland 的表面法向量等于 d。

optiland 的表面法向量计算：
n = Ry(ry) @ Rx(rx) @ [0, 0, 1]
  = Ry(ry) @ [0, -sin(rx), cos(rx)]
  = [sin(ry)*cos(rx), -sin(rx), cos(ry)*cos(rx)]

要使 n = (L, M, N)：
  sin(ry)*cos(rx) = L
  -sin(rx) = M
  cos(ry)*cos(rx) = N

从第二个方程：rx = -arcsin(M)

代入第一和第三个方程：
  sin(ry)*cos(-arcsin(M)) = L
  cos(ry)*cos(-arcsin(M)) = N

由于 cos(-arcsin(M)) = sqrt(1 - M²)：
  sin(ry)*sqrt(1 - M²) = L
  cos(ry)*sqrt(1 - M²) = N

如果 sqrt(1 - M²) ≠ 0：
  sin(ry) = L / sqrt(1 - M²)
  cos(ry) = N / sqrt(1 - M²)
  ry = arctan2(L / sqrt(1 - M²), N / sqrt(1 - M²))
     = arctan2(L, N)

所以正确的公式是：
  rx = -arcsin(M)
  ry = arctan2(L, N)

与当前代码的区别：
  当前代码：rx = arcsin(M)
  正确公式：rx = -arcsin(M)

这就是问题所在！rx 的符号错了！
""")
    
    # 验证
    print("\n--- 验证 ---")
    
    for tilt_deg in [5, 10, 22.5, 45]:
        tilt_rad = np.radians(tilt_deg)
        
        # 出射方向
        d_out = np.array([0, np.sin(2*tilt_rad), -np.cos(2*tilt_rad)])
        L, M, N = d_out
        
        # 当前代码的计算
        rx_current = np.arcsin(M)
        ry_current = np.arctan2(L, N)
        
        # 正确的计算
        rx_correct = -np.arcsin(M)
        ry_correct = np.arctan2(L, N)
        
        # 验证当前代码
        c, s = np.cos(rx_current), np.sin(rx_current)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(ry_current), np.sin(ry_current)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        n_current = Ry @ Rx @ np.array([0, 0, 1])
        
        # 验证正确公式
        c, s = np.cos(rx_correct), np.sin(rx_correct)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(ry_correct), np.sin(ry_correct)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        n_correct = Ry @ Rx @ np.array([0, 0, 1])
        
        print(f"\n{tilt_deg}° 倾斜:")
        print(f"  出射方向: {d_out}")
        print(f"  当前代码: rx={np.degrees(rx_current):.1f}°, ry={np.degrees(ry_current):.1f}° -> n={n_current}")
        print(f"  正确公式: rx={np.degrees(rx_correct):.1f}°, ry={np.degrees(ry_correct):.1f}° -> n={n_correct}")
        print(f"  当前代码匹配: {np.allclose(n_current, d_out)}")
        print(f"  正确公式匹配: {np.allclose(n_correct, d_out)}")


def main():
    test_optiland_plane_intersection()
    test_correct_exit_surface_definition()
    derive_correct_angles()
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
问题根源：
  ElementRaytracer._direction_to_rotation_angles 中
  rx = arcsin(M) 应该改为 rx = -arcsin(M)

修复方案：
  在 src/wavefront_to_rays/element_raytracer.py 中
  修改 _direction_to_rotation_angles 方法：
  
  # 当前代码（错误）
  rx = np.arcsin(M_clamped)
  
  # 修复后（正确）
  rx = -np.arcsin(M_clamped)
""")


if __name__ == "__main__":
    main()
