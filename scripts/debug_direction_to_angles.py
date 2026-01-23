"""
调试 _direction_to_rotation_angles 函数

问题：
- 出射方向 [0, 0.17, -0.98] 被转换为 rx=10°, ry=180°
- 但这导致出射面法向量为 [0, -0.17, -0.98]（M 分量符号相反）

分析：
- _direction_to_rotation_angles 的目的是找到旋转角度 (rx, ry)
- 使得初始方向 (0, 0, 1) 经过旋转后变为目标方向
- 旋转顺序是 X → Y

问题在于：
- 当 N < 0 时，ry = arctan2(L, N) 会给出 ±180° 附近的值
- 这是正确的数学结果，但 optiland 的表面法向量定义可能不同
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


def analyze_direction_to_angles():
    """分析方向到角度的转换"""
    print("=" * 70)
    print("分析 _direction_to_rotation_angles 函数")
    print("=" * 70)
    
    # 测试几个方向
    test_directions = [
        ([0, 0, 1], "正 Z 方向"),
        ([0, 0, -1], "负 Z 方向"),
        ([0, 0.17364818, -0.98480775], "5° 反射后"),
        ([0, 0.70710678, -0.70710678], "22.5° 反射后"),
        ([0, 1, 0], "正 Y 方向（45° 反射后）"),
    ]
    
    for direction, name in test_directions:
        L, M, N = direction
        
        # _direction_to_rotation_angles 的计算
        M_clamped = np.clip(M, -1.0, 1.0)
        rx = np.arcsin(M_clamped)
        ry = np.arctan2(L, N)
        
        print(f"\n--- {name} ---")
        print(f"  方向: ({L:.6f}, {M:.6f}, {N:.6f})")
        print(f"  rx = arcsin({M:.6f}) = {np.degrees(rx):.2f}°")
        print(f"  ry = arctan2({L:.6f}, {N:.6f}) = {np.degrees(ry):.2f}°")
        
        # 验证：重建方向
        # 初始方向 (0, 0, 1)
        # 绕 X 轴旋转 rx：(0, sin(rx), cos(rx))
        # 绕 Y 轴旋转 ry：(sin(ry)*cos(rx), sin(rx), cos(ry)*cos(rx))
        d_reconstructed = np.array([
            np.sin(ry) * np.cos(rx),
            np.sin(rx),
            np.cos(ry) * np.cos(rx)
        ])
        print(f"  重建方向: ({d_reconstructed[0]:.6f}, {d_reconstructed[1]:.6f}, {d_reconstructed[2]:.6f})")
        print(f"  匹配: {np.allclose(d_reconstructed, direction)}")
        
        # optiland 的表面法向量
        # 初始法向量 (0, 0, 1)
        # 绕 X 轴旋转 rx
        c, s = np.cos(rx), np.sin(rx)
        Rx = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        n_after_rx = Rx @ np.array([0, 0, 1])
        
        # 绕 Y 轴旋转 ry
        c, s = np.cos(ry), np.sin(ry)
        Ry = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        n_after_ry = Ry @ n_after_rx
        
        print(f"  optiland 法向量: ({n_after_ry[0]:.6f}, {n_after_ry[1]:.6f}, {n_after_ry[2]:.6f})")
        
        # 检查法向量与方向的关系
        # 对于出射面，法向量应该与出射方向相同（或相反？）
        dot_product = np.dot(n_after_ry, direction)
        print(f"  法向量 · 方向 = {dot_product:.6f}")
        
        if np.isclose(dot_product, 1.0):
            print(f"  ✓ 法向量与方向相同")
        elif np.isclose(dot_product, -1.0):
            print(f"  ✗ 法向量与方向相反！")
        else:
            print(f"  ? 法向量与方向既不相同也不相反")


def analyze_optiland_surface_convention():
    """分析 optiland 的表面约定"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print("\n" + "=" * 70)
    print("分析 optiland 的表面约定")
    print("=" * 70)
    
    # 创建一个简单的系统，测试光线如何与倾斜表面交互
    
    # 测试：rx = 10°, ry = 0° 的透明平面
    print("\n--- 测试 rx=10°, ry=0° 的透明平面 ---")
    
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
        rx=np.radians(10),
        ry=0.0,
    )
    
    # 创建沿 Z 方向传播的光线
    rays = RealRays(
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 5.0]),
        z=np.array([0.0, 0.0]),
        L=np.array([0.0, 0.0]),
        M=np.array([0.0, 0.0]),
        N=np.array([1.0, 1.0]),
        intensity=np.array([1.0, 1.0]),
        wavelength=np.array([0.633, 0.633]),
    )
    
    optic.surface_group.trace(rays, skip=1)
    
    print(f"  光线 0: pos=({rays.x[0]:.4f}, {rays.y[0]:.4f}, {rays.z[0]:.4f})")
    print(f"  光线 1: pos=({rays.x[1]:.4f}, {rays.y[1]:.4f}, {rays.z[1]:.4f})")
    
    # 分析：对于 rx=10° 的平面，光线应该在哪里与平面相交？
    # 平面方程：n · (r - r0) = 0
    # 其中 n 是法向量，r0 是平面上的一点（原点）
    # 法向量 n = (0, -sin(10°), cos(10°)) = (0, -0.174, 0.985)
    # 
    # 光线方程：r = r0 + t * d
    # 其中 d = (0, 0, 1)
    # 
    # 代入平面方程：n · (r0 + t * d - r0) = 0
    # n · (t * d) = 0
    # t * (n · d) = 0
    # t = 0（如果 n · d ≠ 0）
    # 
    # 但是 n · d = cos(10°) ≈ 0.985 ≠ 0
    # 所以 t = 0，光线与平面在原点相交
    # 
    # 但是对于 y=5 的光线，它应该在哪里与平面相交？
    # 光线方程：r = (0, 5, 0) + t * (0, 0, 1) = (0, 5, t)
    # 平面方程：-0.174 * 5 + 0.985 * t = 0
    # t = 0.174 * 5 / 0.985 = 0.883
    # 
    # 所以光线 1 应该在 (0, 5, 0.883) 与平面相交
    
    t_expected = np.sin(np.radians(10)) * 5 / np.cos(np.radians(10))
    print(f"  期望光线 1 的 z 坐标: {t_expected:.4f}")
    
    # 但是 optiland 的结果是什么？
    # 如果 optiland 的法向量定义不同，结果可能不同


def main():
    analyze_direction_to_angles()
    analyze_optiland_surface_convention()
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
关键发现：

1. _direction_to_rotation_angles 函数的数学是正确的
   - 它正确地将方向转换为旋转角度
   - 重建的方向与原始方向匹配

2. 问题在于 optiland 的表面法向量定义
   - optiland 的表面法向量是 (0, 0, 1) 经过旋转后的结果
   - 但这个法向量与我们期望的出射方向相反（当 N < 0 时）

3. 当 N < 0 时，ry = arctan2(L, N) ≈ 180°
   - 这导致绕 Y 轴旋转 180°
   - 使得法向量的 Y 分量反向

4. 解决方案：
   - 需要修改出射面的定义方式
   - 或者在光线追迹后手动将光线传播到正确的出射面

5. 为什么 45° 时没问题：
   - 45° 时，出射方向是 (0, 1, 0)
   - rx = 90°, ry = 180°
   - 法向量是 (0, -1, 0)
   - 虽然方向相反，但平面是垂直于 Y 轴的
   - 沿 Y 方向传播的光线与这个平面的交点计算仍然正确
   - 因为光线方向与平面法向量平行（或反平行）
""")


if __name__ == "__main__":
    main()
