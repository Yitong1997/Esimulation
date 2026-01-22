"""
调试坐标转换问题

分析为什么输出光线的 X 坐标偏移到了 -100 到 -60 mm 的范围
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
    compute_rotation_matrix,
    transform_rays_to_global,
    transform_rays_to_local,
)
from optiland.rays import RealRays


def main():
    """主函数"""
    
    # 模拟表面 4 的情况
    # 入射光轴方向: (0, 1, 0) - 沿 Y 轴
    # 出射光轴方向: (0, 0, 1) - 沿 Z 轴
    # 这是一个 45° 折叠镜
    
    print("=" * 60)
    print("测试 45° 折叠镜的坐标转换")
    print("=" * 60)
    
    # 入射光轴方向（沿 Y 轴）
    entrance_direction = (0.0, 1.0, 0.0)
    
    # 入射面位置
    entrance_position = (0.0, 40.0, 40.0)
    
    print(f"\n入射光轴方向: {entrance_direction}")
    print(f"入射面位置: {entrance_position}")
    
    # 计算入射面旋转矩阵
    R_entrance = compute_rotation_matrix(entrance_direction)
    print(f"\n入射面旋转矩阵 R_entrance:")
    print(R_entrance)
    
    # 创建测试光线（在入射面局部坐标系中）
    # 局部坐标系：Z 轴沿入射光轴方向
    test_rays = RealRays(
        x=np.array([0.0, 10.0, -10.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 10.0, -10.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([0.55, 0.55, 0.55, 0.55, 0.55]),
    )
    test_rays.opd = np.zeros(5)
    
    print(f"\n输入光线（入射面局部坐标系）:")
    print(f"  X: {test_rays.x}")
    print(f"  Y: {test_rays.y}")
    print(f"  Z: {test_rays.z}")
    print(f"  方向: L={test_rays.L}, M={test_rays.M}, N={test_rays.N}")
    
    # 转换到全局坐标系
    rays_global = transform_rays_to_global(test_rays, R_entrance, entrance_position)
    
    print(f"\n光线（全局坐标系）:")
    print(f"  X: {np.asarray(rays_global.x)}")
    print(f"  Y: {np.asarray(rays_global.y)}")
    print(f"  Z: {np.asarray(rays_global.z)}")
    print(f"  方向: L={np.asarray(rays_global.L)}, M={np.asarray(rays_global.M)}, N={np.asarray(rays_global.N)}")
    
    # 出射光轴方向（沿 Z 轴）
    exit_direction = (0.0, 0.0, 1.0)
    exit_position = entrance_position  # 出射面位置与入射面相同
    
    print(f"\n出射光轴方向: {exit_direction}")
    print(f"出射面位置: {exit_position}")
    
    # 计算出射面旋转矩阵
    R_exit = compute_rotation_matrix(exit_direction)
    print(f"\n出射面旋转矩阵 R_exit:")
    print(R_exit)
    
    # 模拟反射后的光线（在全局坐标系中）
    # 反射后，光线方向从 (0, 1, 0) 变为 (0, 0, 1)
    # 位置保持不变（在反射点）
    rays_reflected = RealRays(
        x=np.asarray(rays_global.x).copy(),
        y=np.asarray(rays_global.y).copy(),
        z=np.asarray(rays_global.z).copy(),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([0.55, 0.55, 0.55, 0.55, 0.55]),
    )
    rays_reflected.opd = np.zeros(5)
    
    print(f"\n反射后光线（全局坐标系）:")
    print(f"  X: {np.asarray(rays_reflected.x)}")
    print(f"  Y: {np.asarray(rays_reflected.y)}")
    print(f"  Z: {np.asarray(rays_reflected.z)}")
    
    # 转换到出射面局部坐标系
    rays_exit = transform_rays_to_local(rays_reflected, R_exit, exit_position)
    
    print(f"\n光线（出射面局部坐标系）:")
    print(f"  X: {np.asarray(rays_exit.x)}")
    print(f"  Y: {np.asarray(rays_exit.y)}")
    print(f"  Z: {np.asarray(rays_exit.z)}")
    print(f"  方向: L={np.asarray(rays_exit.L)}, M={np.asarray(rays_exit.M)}, N={np.asarray(rays_exit.N)}")
    
    # 分析问题
    print("\n" + "=" * 60)
    print("问题分析")
    print("=" * 60)
    
    print(f"""
入射面局部坐标系：
  - Z 轴沿入射光轴方向 (0, 1, 0)
  - X 轴和 Y 轴在垂直于入射光轴的平面内
  
出射面局部坐标系：
  - Z 轴沿出射光轴方向 (0, 0, 1)
  - X 轴和 Y 轴在垂直于出射光轴的平面内

问题：
  - 入射面局部坐标系中的 (X, Y) 平面是 XZ 平面（全局）
  - 出射面局部坐标系中的 (X, Y) 平面是 XY 平面（全局）
  - 这两个平面不同，所以坐标转换后位置会发生变化
  
解决方案：
  - 需要确保出射面的网格采样与入射面一致
  - 或者在重建时使用正确的网格范围
""")
    
    # 测试实际的 ElementRaytracer
    print("\n" + "=" * 60)
    print("测试 ElementRaytracer")
    print("=" * 60)
    
    # 创建平面反射镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=30.0,
        tilt_x=np.pi/4,  # 45° 倾斜
    )
    
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=0.55,
        chief_ray_direction=entrance_direction,
        entrance_position=entrance_position,
    )
    
    print(f"\n入射面旋转矩阵:")
    print(raytracer.rotation_matrix)
    
    print(f"\n出射主光线方向: {raytracer.exit_chief_direction}")
    
    print(f"\n出射面旋转矩阵:")
    print(raytracer.exit_rotation_matrix)
    
    # 追迹光线
    output_rays = raytracer.trace(test_rays)
    
    print(f"\n输出光线（出射面局部坐标系）:")
    print(f"  X: {np.asarray(output_rays.x)}")
    print(f"  Y: {np.asarray(output_rays.y)}")
    print(f"  Z: {np.asarray(output_rays.z)}")
    
    print("\n完成")


if __name__ == '__main__':
    main()
