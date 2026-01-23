"""
调试出射面位置问题

关键发现：
- optiland 追迹后，光线位置不在出射面上
- 5° 倾斜时，光线 3 和 4 到出射面的距离分别是 ±1.82 mm
- 45° 倾斜时，所有光线到出射面的距离都是 0

问题分析：
- optiland 的出射面可能没有正确定位
- 或者光线追迹没有正确传播到出射面
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


def analyze_exit_surface(tilt_deg: float):
    """分析出射面位置"""
    from wavefront_to_rays.element_raytracer import (
        ElementRaytracer, SurfaceDefinition
    )
    from optiland.rays import RealRays
    
    print(f"\n{'='*70}")
    print(f"分析出射面位置: {tilt_deg}°")
    print(f"{'='*70}")
    
    wavelength_um = 0.633
    tilt_rad = np.radians(tilt_deg)
    
    # 计算出射方向
    d_in = np.array([0, 0, 1])
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)
    
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
    
    # 分析 optiland 的表面定义
    surfaces = raytracer.optic.surface_group.surfaces
    
    print(f"\n--- optiland 表面详细分析 ---")
    for i, surf in enumerate(surfaces):
        print(f"\n表面 {i}: {type(surf).__name__}")
        if hasattr(surf, 'geometry'):
            geom = surf.geometry
            cs = geom.cs
            print(f"  坐标系:")
            print(f"    位置: ({cs.x}, {cs.y}, {cs.z})")
            print(f"    旋转: rx={np.degrees(cs.rx):.2f}°, ry={np.degrees(cs.ry):.2f}°")
            
            # 计算表面法向量（在全局坐标系中）
            # 初始法向量是 (0, 0, 1)（指向 +Z）
            # 经过旋转后变为...
            n_local = np.array([0, 0, 1])
            
            # 绕 X 轴旋转 rx
            c, s = np.cos(cs.rx), np.sin(cs.rx)
            Rx = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
            n_rotated = Rx @ n_local
            
            # 绕 Y 轴旋转 ry
            c, s = np.cos(cs.ry), np.sin(cs.ry)
            Ry = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
            n_global = Ry @ n_rotated
            
            print(f"    法向量（全局）: {n_global}")
    
    # 分析问题
    print(f"\n--- 问题分析 ---")
    
    # 对于 5° 倾斜的平面镜：
    # - 镜面法向量：(0, sin(5°), -cos(5°)) = (0, 0.087, -0.996)
    # - 出射方向：(0, sin(10°), -cos(10°)) = (0, 0.174, -0.985)
    # - 出射面法向量应该与出射方向相同
    
    # 但是 optiland 的出射面定义是：
    # - rx = 10°, ry = 180°
    # - 这意味着先绕 X 轴旋转 10°，再绕 Y 轴旋转 180°
    
    # 让我计算这个旋转后的法向量
    rx_exit = np.radians(2 * tilt_deg)  # 出射面 rx
    ry_exit = np.radians(180)  # 出射面 ry
    
    n_local = np.array([0, 0, 1])
    
    c, s = np.cos(rx_exit), np.sin(rx_exit)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    n_rotated = Rx @ n_local
    
    c, s = np.cos(ry_exit), np.sin(ry_exit)
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    n_exit_surface = Ry @ n_rotated
    
    print(f"  出射面法向量（计算）: {n_exit_surface}")
    print(f"  期望的出射方向: {d_out}")
    print(f"  匹配: {np.allclose(n_exit_surface, d_out)}")
    
    # 问题：出射面的法向量与出射方向不匹配！
    # 这是因为 ry = 180° 导致法向量反向
    
    # 让我检查 ElementRaytracer 中出射面的定义
    print(f"\n--- ElementRaytracer 出射面定义分析 ---")
    
    # 从 _create_optic 方法中，出射面的 rx, ry 是这样计算的：
    # exit_dir_local = rotation_matrix.T @ exit_dir_global
    # exit_rx, exit_ry = _direction_to_rotation_angles(exit_dir_local)
    
    exit_dir_global = d_out
    exit_dir_local = raytracer.rotation_matrix.T @ exit_dir_global
    
    print(f"  出射方向（全局）: {exit_dir_global}")
    print(f"  出射方向（入射面局部）: {exit_dir_local}")
    
    # _direction_to_rotation_angles 的计算
    L, M, N = exit_dir_local
    M_clamped = np.clip(M, -1.0, 1.0)
    rx = np.arcsin(M_clamped)
    ry = np.arctan2(L, N)
    
    print(f"  计算的 rx: {np.degrees(rx):.2f}°")
    print(f"  计算的 ry: {np.degrees(ry):.2f}°")
    
    # 验证：这个 rx, ry 是否能正确表示出射方向？
    # 初始方向是 (0, 0, 1)
    # 绕 X 轴旋转 rx 后：(0, sin(rx), cos(rx))
    # 绕 Y 轴旋转 ry 后：(sin(ry)*cos(rx), sin(rx), cos(ry)*cos(rx))
    
    d_reconstructed = np.array([
        np.sin(ry) * np.cos(rx),
        np.sin(rx),
        np.cos(ry) * np.cos(rx)
    ])
    
    print(f"  重建的方向: {d_reconstructed}")
    print(f"  与出射方向匹配: {np.allclose(d_reconstructed, exit_dir_local)}")


def main():
    print("=" * 70)
    print("调试出射面位置问题")
    print("=" * 70)
    
    for angle in [5, 22.5, 45]:
        analyze_exit_surface(angle)
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
问题根源：

optiland 的出射面定义中，ry = 180° 导致法向量反向。

这意味着出射面的法向量与出射方向相反，所以光线不会正确地传播到出射面。

解决方案：
1. 修改出射面的定义，使其法向量与出射方向一致
2. 或者在光线追迹后，手动将光线传播到正确的出射面

但是，这可能不是唯一的问题...

让我进一步分析 optiland 的光线追迹行为。
""")


if __name__ == "__main__":
    main()
