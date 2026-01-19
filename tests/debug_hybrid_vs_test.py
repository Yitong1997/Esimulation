"""
对比 test_element_raytracer 中的测试方式和 _apply_element_hybrid 中的使用方式

目标：找出为什么测试通过但实际使用时出错
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.rays import RealRays

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


def test_like_element_raytracer_test():
    """模拟 test_element_raytracer.py 中的测试方式"""
    print("=" * 70)
    print("方式 1: 模拟 test_element_raytracer.py 的测试方式")
    print("=" * 70)
    
    # 参数（与 test_element_raytracer 类似）
    wavelength = 0.55  # μm
    f = -50.0  # mm, 凸抛物面镜焦距
    vertex_radius = 2 * f  # -100 mm
    
    # 创建表面定义（无倾斜）
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=0.0,
    )
    
    # 创建入射光线（平行光，沿 z 轴）
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"\n波长: {wavelength} μm")
    print(f"焦距: {f} mm")
    print(f"顶点曲率半径: {vertex_radius} mm")
    
    print(f"\n入射光线 x: {x}")
    print(f"OPD (waves): {opd_waves}")
    print(f"OPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")
    print(f"OPD RMS: {np.nanstd(opd_waves):.6f}")
    
    # 理论 OPD（透镜相位）
    # OPD = -r² / (2f * λ)
    wavelength_mm = wavelength * 1e-3
    r_sq = x**2
    opd_theory = -r_sq / (2 * f * wavelength_mm)
    
    print(f"\n理论 OPD: {opd_theory}")
    print(f"差异: {opd_waves - opd_theory}")
    
    return opd_waves, opd_theory


def test_like_hybrid_propagation():
    """模拟 _apply_element_hybrid 中的使用方式"""
    print("\n" + "=" * 70)
    print("方式 2: 模拟 _apply_element_hybrid 的使用方式")
    print("=" * 70)
    
    # 参数（与 galilean_oap_expander 相同）
    wavelength = 10.64  # μm
    f = -50.0  # mm, 凸抛物面镜焦距
    vertex_radius = 2 * f  # -100 mm
    
    # 创建表面定义（有 45° 倾斜）
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.radians(45.0),  # 45° 倾斜
    )
    
    # 创建入射光线（平行光，沿 z 轴）
    # 注意：_apply_element_hybrid 使用 WavefrontToRaysSampler 生成光线
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"\n波长: {wavelength} μm")
    print(f"焦距: {f} mm")
    print(f"顶点曲率半径: {vertex_radius} mm")
    print(f"倾斜角: 45°")
    
    print(f"\n入射光线 x: {x}")
    print(f"OPD (waves): {opd_waves}")
    print(f"OPD 范围: [{np.nanmin(opd_waves):.6f}, {np.nanmax(opd_waves):.6f}]")
    print(f"OPD RMS: {np.nanstd(opd_waves):.6f}")
    
    return opd_waves


def test_wavefront_sampler_rays():
    """检查 WavefrontToRaysSampler 生成的光线"""
    print("\n" + "=" * 70)
    print("方式 3: 检查 WavefrontToRaysSampler 生成的光线")
    print("=" * 70)
    
    from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
    
    # 创建简单的平面波前
    n = 64
    wavelength = 10.64  # μm
    physical_size = 160.0  # mm
    
    # 平面波前（振幅为高斯，相位为零）
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    w0 = 10.0  # mm
    amplitude = np.exp(-(X**2 + Y**2) / w0**2)
    wavefront = amplitude.astype(np.complex128)  # 相位为零
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        wavelength=wavelength,
        physical_size=physical_size,
        num_rays=100,
    )
    
    rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    ray_opd = sampler.get_ray_opd()
    
    print(f"\n光线数量: {len(ray_x)}")
    print(f"光线 x 范围: [{np.min(ray_x):.3f}, {np.max(ray_x):.3f}] mm")
    print(f"光线 y 范围: [{np.min(ray_y):.3f}, {np.max(ray_y):.3f}] mm")
    
    # 检查光线方向
    L = np.asarray(rays.L)
    M = np.asarray(rays.M)
    N = np.asarray(rays.N)
    
    print(f"\n光线方向余弦:")
    print(f"  L 范围: [{np.min(L):.6f}, {np.max(L):.6f}]")
    print(f"  M 范围: [{np.min(M):.6f}, {np.max(M):.6f}]")
    print(f"  N 范围: [{np.min(N):.6f}, {np.max(N):.6f}]")
    
    print(f"\n入射 OPD 范围: [{np.min(ray_opd):.6f}, {np.max(ray_opd):.6f}] waves")
    
    # 检查光线 z 位置
    z = np.asarray(rays.z)
    print(f"\n光线 z 位置范围: [{np.min(z):.6f}, {np.max(z):.6f}] mm")
    
    return rays, ray_opd


def test_with_wavefront_sampler_rays():
    """使用 WavefrontToRaysSampler 生成的光线进行追迹"""
    print("\n" + "=" * 70)
    print("方式 4: 使用 WavefrontToRaysSampler 光线进行追迹")
    print("=" * 70)
    
    from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
    
    # 创建简单的平面波前
    n = 64
    wavelength = 10.64  # μm
    physical_size = 160.0  # mm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    # 平面波前
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    w0 = 10.0  # mm
    amplitude = np.exp(-(X**2 + Y**2) / w0**2)
    wavefront = amplitude.astype(np.complex128)
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        wavelength=wavelength,
        physical_size=physical_size,
        num_rays=100,
    )
    
    rays_in = sampler.get_output_rays()
    input_opd = sampler.get_ray_opd()
    
    # 创建表面定义（有 45° 倾斜）
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.radians(45.0),
    )
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n有效光线数: {np.sum(valid_mask)}")
    print(f"入射 OPD 范围: [{np.min(input_opd):.6f}, {np.max(input_opd):.6f}] waves")
    print(f"元件 OPD 范围: [{np.nanmin(element_opd):.6f}, {np.nanmax(element_opd):.6f}] waves")
    print(f"元件 OPD RMS: {np.nanstd(element_opd[valid_mask]):.6f} waves")
    
    # 检查入射光线的初始 OPD
    rays_in_opd = np.asarray(rays_in.opd)
    print(f"\n入射光线初始 OPD (mm): [{np.min(rays_in_opd):.6f}, {np.max(rays_in_opd):.6f}]")
    
    # 检查出射光线的 OPD
    rays_out_opd = np.asarray(rays_out.opd)
    print(f"出射光线 OPD (mm): [{np.min(rays_out_opd):.6f}, {np.max(rays_out_opd):.6f}]")
    
    return element_opd, valid_mask


def test_compare_with_without_tilt():
    """对比有无倾斜的 OPD"""
    print("\n" + "=" * 70)
    print("方式 5: 对比有无倾斜的 OPD")
    print("=" * 70)
    
    wavelength = 10.64  # μm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    # 创建入射光线
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    # 无倾斜
    surface_no_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=0.0,
    )
    
    rays_in_1 = RealRays(x.copy(), y.copy(), z.copy(), L.copy(), M.copy(), N.copy(), intensity.copy(), wavelength)
    raytracer_1 = ElementRaytracer(surfaces=[surface_no_tilt], wavelength=wavelength)
    raytracer_1.trace(rays_in_1)
    opd_no_tilt = raytracer_1.get_relative_opd_waves()
    
    # 有 45° 倾斜
    surface_with_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.radians(45.0),
    )
    
    rays_in_2 = RealRays(x.copy(), y.copy(), z.copy(), L.copy(), M.copy(), N.copy(), intensity.copy(), wavelength)
    raytracer_2 = ElementRaytracer(surfaces=[surface_with_tilt], wavelength=wavelength)
    raytracer_2.trace(rays_in_2)
    opd_with_tilt = raytracer_2.get_relative_opd_waves()
    
    print(f"\n无倾斜 OPD: {opd_no_tilt}")
    print(f"有倾斜 OPD: {opd_with_tilt}")
    print(f"\n无倾斜 OPD 范围: [{np.nanmin(opd_no_tilt):.6f}, {np.nanmax(opd_no_tilt):.6f}]")
    print(f"有倾斜 OPD 范围: [{np.nanmin(opd_with_tilt):.6f}, {np.nanmax(opd_with_tilt):.6f}]")
    
    # 理论上，对于折叠倾斜（is_fold=True），倾斜不应该改变 OPD 的形状
    # 只是坐标系发生了变化
    
    return opd_no_tilt, opd_with_tilt


def main():
    test_like_element_raytracer_test()
    test_like_hybrid_propagation()
    test_wavefront_sampler_rays()
    test_with_wavefront_sampler_rays()
    test_compare_with_without_tilt()


if __name__ == "__main__":
    main()
