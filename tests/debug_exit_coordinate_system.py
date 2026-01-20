"""检查 ElementRaytracer 的出射面坐标系

验证 ElementRaytracer 是否正确地将光线转换到了出射面局部坐标系。

关键问题：
- 出射面应该垂直于出射光轴（主光线方向）
- 出射光线的 z 坐标应该接近 0（在出射面上）
- 出射光线的方向应该沿出射光轴（L=0, M=0, N=1 或类似）
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def check_exit_coordinate_system():
    """检查出射面坐标系"""
    
    print("=" * 70)
    print("ElementRaytracer 出射面坐标系检查")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
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
    
    # 创建简单的输入光线（只有中心光线和边缘光线）
    ray_x = np.array([0.0, 5.0, -5.0, 0.0, 0.0])
    ray_y = np.array([0.0, 0.0, 0.0, 5.0, -5.0])
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
    # 检查无倾斜情况
    # =========================================================================
    print("\n1. 无倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_no_tilt = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    
    print(f"   入射主光线方向: {raytracer_no_tilt.chief_ray_direction}")
    print(f"   出射主光线方向: {raytracer_no_tilt.exit_chief_direction}")
    print(f"   入射面旋转矩阵:\n{raytracer_no_tilt.rotation_matrix}")
    print(f"   出射面旋转矩阵:\n{raytracer_no_tilt.exit_rotation_matrix}")
    
    rays_out_no_tilt = raytracer_no_tilt.trace(create_rays())
    
    print("\n   出射光线（在出射面局部坐标系中）：")
    for i in range(n_rays):
        x = float(rays_out_no_tilt.x[i])
        y = float(rays_out_no_tilt.y[i])
        z = float(rays_out_no_tilt.z[i])
        L = float(rays_out_no_tilt.L[i])
        M = float(rays_out_no_tilt.M[i])
        N = float(rays_out_no_tilt.N[i])
        opd = float(rays_out_no_tilt.opd[i])
        print(f"     光线 {i}: pos=({x:.4f}, {y:.4f}, {z:.4f}), dir=({L:.4f}, {M:.4f}, {N:.4f}), OPD={opd:.6f}")
    
    # =========================================================================
    # 检查有倾斜情况
    # =========================================================================
    print("\n2. 45° 倾斜抛物面镜：")
    print("-" * 50)
    
    raytracer_with_tilt = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    
    print(f"   入射主光线方向: {raytracer_with_tilt.chief_ray_direction}")
    print(f"   出射主光线方向: {raytracer_with_tilt.exit_chief_direction}")
    print(f"   入射面旋转矩阵:\n{raytracer_with_tilt.rotation_matrix}")
    print(f"   出射面旋转矩阵:\n{raytracer_with_tilt.exit_rotation_matrix}")
    
    rays_out_with_tilt = raytracer_with_tilt.trace(create_rays())
    
    print("\n   出射光线（在出射面局部坐标系中）：")
    for i in range(n_rays):
        x = float(rays_out_with_tilt.x[i])
        y = float(rays_out_with_tilt.y[i])
        z = float(rays_out_with_tilt.z[i])
        L = float(rays_out_with_tilt.L[i])
        M = float(rays_out_with_tilt.M[i])
        N = float(rays_out_with_tilt.N[i])
        opd = float(rays_out_with_tilt.opd[i])
        print(f"     光线 {i}: pos=({x:.4f}, {y:.4f}, {z:.4f}), dir=({L:.4f}, {M:.4f}, {N:.4f}), OPD={opd:.6f}")
    
    # =========================================================================
    # 分析
    # =========================================================================
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)
    
    # 检查出射光线的 z 坐标是否接近 0
    z_no_tilt = np.array([float(rays_out_no_tilt.z[i]) for i in range(n_rays)])
    z_with_tilt = np.array([float(rays_out_with_tilt.z[i]) for i in range(n_rays)])
    
    print(f"\n无倾斜 - 出射光线 z 坐标范围: [{np.min(z_no_tilt):.6f}, {np.max(z_no_tilt):.6f}]")
    print(f"有倾斜 - 出射光线 z 坐标范围: [{np.min(z_with_tilt):.6f}, {np.max(z_with_tilt):.6f}]")
    
    # 检查出射光线的方向是否沿出射光轴
    N_no_tilt = np.array([float(rays_out_no_tilt.N[i]) for i in range(n_rays)])
    N_with_tilt = np.array([float(rays_out_with_tilt.N[i]) for i in range(n_rays)])
    
    print(f"\n无倾斜 - 出射光线 N 分量范围: [{np.min(N_no_tilt):.6f}, {np.max(N_no_tilt):.6f}]")
    print(f"有倾斜 - 出射光线 N 分量范围: [{np.min(N_with_tilt):.6f}, {np.max(N_with_tilt):.6f}]")


if __name__ == "__main__":
    check_exit_coordinate_system()
