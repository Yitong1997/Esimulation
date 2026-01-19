"""
诊断 45° 倾斜元件的 OPD 计算问题

问题：即使手动创建的光线（OPD=0），45° 倾斜元件的 OPD 也非常大
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.rays import RealRays

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


def test_flat_mirror_45deg():
    """测试 45° 平面镜的 OPD"""
    print("=" * 70)
    print("测试 1: 45° 平面镜的 OPD")
    print("=" * 70)
    
    wavelength = 10.64  # μm
    
    # 创建 45° 平面镜
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,  # 平面
        thickness=0.0,
        material='mirror',
        semi_aperture=50.0,
        conic=0.0,
        tilt_x=np.radians(45.0),
    )
    
    # 创建入射光线
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n入射光线 x: {x}")
    print(f"OPD (waves): {opd_waves}")
    print(f"有效光线: {valid_mask}")
    
    # 理论上，45° 平面镜的 OPD 应该为 0（对于折叠倾斜）
    print(f"\n理论：45° 平面镜（折叠）的 OPD 应该为 0")
    print(f"实际 OPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")


def test_parabolic_mirror_no_tilt():
    """测试无倾斜抛物面镜的 OPD"""
    print("\n" + "=" * 70)
    print("测试 2: 无倾斜抛物面镜的 OPD")
    print("=" * 70)
    
    wavelength = 10.64  # μm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=0.0,  # 无倾斜
    )
    
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    # 理论 OPD
    wavelength_mm = wavelength * 1e-3
    r_sq = x**2
    opd_theory = -r_sq / (2 * f * wavelength_mm)
    
    print(f"\n入射光线 x: {x}")
    print(f"追迹 OPD (waves): {opd_waves}")
    print(f"理论 OPD (waves): {opd_theory}")
    print(f"\n追迹 OPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")
    print(f"理论 OPD 范围: [{np.min(opd_theory):.6f}, {np.max(opd_theory):.6f}]")


def test_parabolic_mirror_45deg():
    """测试 45° 倾斜抛物面镜的 OPD"""
    print("\n" + "=" * 70)
    print("测试 3: 45° 倾斜抛物面镜的 OPD")
    print("=" * 70)
    
    wavelength = 10.64  # μm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.radians(45.0),  # 45° 倾斜
    )
    
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n入射光线 x: {x}")
    print(f"OPD (waves): {opd_waves}")
    print(f"有效光线: {valid_mask}")
    print(f"\nOPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")
    
    # 检查出射光线的原始 OPD
    out_opd_mm = np.asarray(rays_out.opd)
    print(f"\n出射光线原始 OPD (mm): {out_opd_mm}")


def test_with_large_aperture():
    """测试大孔径光线"""
    print("\n" + "=" * 70)
    print("测试 4: 大孔径光线（x 范围 [-80, 80] mm）")
    print("=" * 70)
    
    wavelength = 10.64  # μm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=100.0,  # 大孔径
        conic=-1.0,
        tilt_x=np.radians(45.0),
    )
    
    n_rays = 11
    x = np.linspace(-80, 80, n_rays)  # 大范围
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n入射光线 x: {x}")
    print(f"OPD (waves): {opd_waves}")
    print(f"有效光线: {valid_mask}")
    print(f"\nOPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")
    
    # 检查出射光线的原始 OPD
    out_opd_mm = np.asarray(rays_out.opd)
    print(f"\n出射光线原始 OPD (mm):")
    for i, (xi, opd) in enumerate(zip(x, out_opd_mm)):
        print(f"  x={xi:8.1f} mm: OPD={opd:12.3f} mm")


def main():
    test_flat_mirror_45deg()
    test_parabolic_mirror_no_tilt()
    test_parabolic_mirror_45deg()
    test_with_large_aperture()


if __name__ == "__main__":
    main()
