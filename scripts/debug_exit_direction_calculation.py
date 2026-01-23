"""
调试出射方向计算

问题：ElementRaytracer._compute_exit_chief_direction 计算的出射方向
与 optiland 实际追迹的出射方向不一致。

原因分析：
- ElementRaytracer 中的旋转矩阵定义与 optiland 不同
- 需要找出正确的旋转约定
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


def compare_exit_direction_calculations(tilt_deg: float):
    """比较不同方法计算的出射方向"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print(f"\n{'='*70}")
    print(f"比较出射方向计算: {tilt_deg}°")
    print(f"{'='*70}")
    
    tilt_rad = np.radians(tilt_deg)
    
    # 1. optiland 实际追迹的出射方向
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
        material='mirror',
        is_stop=True,
        rx=tilt_rad,
        ry=0.0,
    )
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        material='air',
    )
    
    rays = RealRays(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        L=np.array([0.0]),
        M=np.array([0.0]),
        N=np.array([1.0]),
        intensity=np.array([1.0]),
        wavelength=np.array([0.633]),
    )
    
    optic.surface_group.trace(rays, skip=1)
    
    d_out_optiland = np.array([float(rays.L[0]), float(rays.M[0]), float(rays.N[0])])
    print(f"\n1. optiland 追迹的出射方向: {d_out_optiland}")
    
    # 2. ElementRaytracer._compute_exit_chief_direction 的计算方式
    # 初始法向量沿 -Z
    n_current = np.array([0.0, 0.0, -1.0])
    
    # 绕 X 轴旋转（当前代码的方式）
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    Rx_current = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    n_current = Rx_current @ n_current
    
    d_in = np.array([0, 0, 1])
    d_out_current = d_in - 2 * np.dot(d_in, n_current) * n_current
    d_out_current = d_out_current / np.linalg.norm(d_out_current)
    
    print(f"2. 当前代码计算的出射方向: {d_out_current}")
    print(f"   法向量: {n_current}")
    
    # 3. 正确的计算方式（与 optiland 一致）
    # optiland 的 rx 旋转使得法向量的 Y 分量为正
    # 这意味着旋转矩阵应该是：
    # n_y = sin(rx), n_z = -cos(rx)
    # 即 n = [0, sin(rx), -cos(rx)]
    
    n_correct = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out_correct = d_in - 2 * np.dot(d_in, n_correct) * n_correct
    d_out_correct = d_out_correct / np.linalg.norm(d_out_correct)
    
    print(f"3. 正确的出射方向: {d_out_correct}")
    print(f"   法向量: {n_correct}")
    
    # 4. 分析差异
    print(f"\n差异分析:")
    print(f"  optiland vs 当前代码: {np.allclose(d_out_optiland, d_out_current)}")
    print(f"  optiland vs 正确计算: {np.allclose(d_out_optiland, d_out_correct)}")
    
    # 5. 找出旋转矩阵的问题
    print(f"\n旋转矩阵分析:")
    print(f"  当前代码的 Rx @ [0,0,-1] = {Rx_current @ np.array([0,0,-1])}")
    print(f"  期望的法向量 = {n_correct}")
    
    # 正确的旋转矩阵应该使 [0,0,-1] 变为 [0, sin(rx), -cos(rx)]
    # 这需要绕 X 轴旋转 -rx（负号！）
    Rx_correct = np.array([
        [1, 0, 0],
        [0, np.cos(-tilt_rad), -np.sin(-tilt_rad)],
        [0, np.sin(-tilt_rad), np.cos(-tilt_rad)]
    ])
    n_from_correct_Rx = Rx_correct @ np.array([0, 0, -1])
    print(f"  正确的 Rx(-rx) @ [0,0,-1] = {n_from_correct_Rx}")
    
    return {
        'd_out_optiland': d_out_optiland,
        'd_out_current': d_out_current,
        'd_out_correct': d_out_correct,
    }


def main():
    print("=" * 70)
    print("调试出射方向计算")
    print("=" * 70)
    
    for angle in [5, 10, 22.5, 45]:
        compare_exit_direction_calculations(angle)
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
问题根源：

ElementRaytracer._compute_exit_chief_direction 中的旋转矩阵定义错误。

当前代码：
  n = [0, 0, -1]
  Rx = [[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]]
  n_rotated = Rx @ n = [0, sin(rx), -cos(rx)]  # 这是正确的！

等等，让我重新检查...

实际上，当前代码的旋转矩阵是正确的：
  Rx @ [0, 0, -1] = [0, sin(rx), -cos(rx)]

这与 optiland 的约定一致。

那么问题出在哪里？

让我重新检查之前的调试输出...

在 debug_coordinate_transform_detail.py 中，我们看到：
  期望的反射方向（入射面局部坐标系）: [0, -0.17364818, -0.98480775]
  实际的光线方向（入射面局部坐标系）: [0, +0.17364818, -0.98480775]

但是在这个脚本中，optiland 的输出是 [0, +0.17364818, -0.98480775]，
这与我们的"正确"计算一致。

问题可能出在：
1. debug_coordinate_transform_detail.py 中的"期望"计算有误
2. 或者问题不在出射方向计算，而在其他地方

让我重新检查 debug_coordinate_transform_detail.py 中的法向量计算...
""")


if __name__ == "__main__":
    main()
