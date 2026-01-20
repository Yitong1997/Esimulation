"""调试光线采样问题 - 深入 ElementRaytracer"""
import sys
import numpy as np

sys.path.insert(0, 'src')

from optiland.rays import RealRays
from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)


def debug_element_raytracer():
    """调试 ElementRaytracer 中的光线采样"""
    
    # 创建简单的凹面镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=200.0,  # 焦距 100mm
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,  # 抛物面
        tilt_x=np.pi/4,  # 45度倾斜
    )
    
    # 创建 ElementRaytracer
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=10.64,
    )
    
    # 创建 5x5 网格的输入光线
    n_rays_1d = 5
    half_size = 10.0
    ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    print("=" * 60)
    print("输入光线 (创建时):")
    print("=" * 60)
    print(f"ray_x unique: {np.unique(ray_x)}")
    print(f"ray_y unique: {np.unique(ray_y)}")
    
    # 创建 RealRays
    input_rays = RealRays(
        x=ray_x,
        y=ray_y,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, 10.64),
    )
    
    print("\n" + "=" * 60)
    print("RealRays 对象 (创建后):")
    print("=" * 60)
    print(f"input_rays.x unique: {np.unique(np.asarray(input_rays.x))}")
    print(f"input_rays.y unique: {np.unique(np.asarray(input_rays.y))}")
    
    # 检查 optiland 的 surface_group
    print("\n" + "=" * 60)
    print("optiland 光学系统信息:")
    print("=" * 60)
    optic = raytracer.optic
    print(f"表面数量: {len(optic.surface_group.surfaces)}")
    for i, surf in enumerate(optic.surface_group.surfaces):
        print(f"  Surface {i}: {type(surf).__name__}")
    
    # 手动模拟 trace 过程中的 localize
    print("\n" + "=" * 60)
    print("模拟 _trace_with_signed_opd 中的 localize:")
    print("=" * 60)
    
    # 复制光线
    test_rays = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    
    print(f"复制后 test_rays.y unique: {np.unique(np.asarray(test_rays.y))}")
    
    # 对每个表面执行 localize
    for i, surface in enumerate(optic.surface_group.surfaces):
        if i < 1:  # skip=1
            continue
        
        print(f"\n--- Surface {i} localize 前 ---")
        print(f"  rays.x unique: {np.unique(np.asarray(test_rays.x))}")
        print(f"  rays.y unique: {np.unique(np.asarray(test_rays.y))}")
        print(f"  rays.z unique: {np.unique(np.asarray(test_rays.z))}")
        
        # 执行 localize
        surface.geometry.localize(test_rays)
        
        print(f"\n--- Surface {i} localize 后 ---")
        print(f"  rays.x unique: {np.unique(np.asarray(test_rays.x))}")
        print(f"  rays.y unique: {np.unique(np.asarray(test_rays.y))}")
        print(f"  rays.z unique: {np.unique(np.asarray(test_rays.z))}")
        
        # 只检查第一个表面
        break


if __name__ == "__main__":
    debug_element_raytracer()
