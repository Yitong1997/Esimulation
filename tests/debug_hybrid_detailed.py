"""
详细诊断混合模式的 OPD 计算
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


def test_wavefront_sampler_opd():
    """测试 WavefrontToRaysSampler 的 OPD 输出"""
    print("=" * 70)
    print("测试 WavefrontToRaysSampler 的 OPD 输出")
    print("=" * 70)
    
    # 参数
    wavelength_um = 10.64
    physical_size = 80.0  # mm
    n = 64
    w0 = 10.0  # mm
    
    # 创建平面波前（相位为零）
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    amplitude = np.exp(-(X**2 + Y**2) / w0**2)
    wavefront = amplitude.astype(np.complex128)  # 相位为零
    
    print(f"\n输入波前：平面波（相位=0），高斯振幅（w0={w0} mm）")
    print(f"物理尺寸：{physical_size} mm")
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        wavelength=wavelength_um,
        physical_size=physical_size,
        num_rays=50,
    )
    
    # 获取光线数据
    ray_x, ray_y = sampler.get_ray_positions()
    input_opd_waves = sampler.get_ray_opd()
    
    print(f"\n采样光线数量：{len(ray_x)}")
    print(f"光线 x 范围：[{np.min(ray_x):.3f}, {np.max(ray_x):.3f}] mm")
    print(f"光线 y 范围：[{np.min(ray_y):.3f}, {np.max(ray_y):.3f}] mm")
    print(f"\n输入 OPD 范围：[{np.min(input_opd_waves):.6f}, {np.max(input_opd_waves):.6f}] waves")
    print(f"输入 OPD 均值：{np.mean(input_opd_waves):.6f} waves")
    print(f"输入 OPD 标准差：{np.std(input_opd_waves):.6f} waves")
    
    # 对于平面波，OPD 应该接近零
    if np.max(np.abs(input_opd_waves)) < 0.01:
        print("\n✓ 平面波的 OPD 接近零")
    else:
        print("\n✗ 平面波的 OPD 不为零！这可能是问题所在")
    
    return input_opd_waves


def test_element_raytracer_opd():
    """测试 ElementRaytracer 的 OPD 输出"""
    print("\n" + "=" * 70)
    print("测试 ElementRaytracer 的 OPD 输出")
    print("=" * 70)
    
    from optiland.rays import RealRays
    
    # 参数
    wavelength_um = 10.64
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    # 创建抛物面镜
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=0.0,
    )
    
    # 创建入射光线（平行光）
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength_um)
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"\n凸抛物面镜：f = {f} mm, R = {vertex_radius} mm")
    print(f"\n元件 OPD（原始）：")
    for xi, opd in zip(x, element_opd_waves):
        print(f"  x={xi:6.1f} mm: OPD={opd:12.6f} waves")
    
    # 取反后的 OPD
    element_opd_corrected = -element_opd_waves
    print(f"\n元件 OPD（取反后）：")
    for xi, opd in zip(x, element_opd_corrected):
        print(f"  x={xi:6.1f} mm: OPD={opd:12.6f} waves")
    
    # PROPER 理论 OPD
    wavelength_mm = wavelength_um * 1e-3
    proper_opd_theory = -x**2 / (2 * f * wavelength_mm)
    print(f"\nPROPER 理论 OPD：")
    for xi, opd in zip(x, proper_opd_theory):
        print(f"  x={xi:6.1f} mm: OPD={opd:12.6f} waves")
    
    # 对比
    print(f"\n对比（取反后 vs PROPER 理论）：")
    for xi, opd_corr, opd_theory in zip(x, element_opd_corrected, proper_opd_theory):
        diff = opd_corr - opd_theory
        print(f"  x={xi:6.1f} mm: 差异={diff:12.6f} waves")


def test_combined_opd():
    """测试组合 OPD（input_opd + element_opd）"""
    print("\n" + "=" * 70)
    print("测试组合 OPD（input_opd + element_opd）")
    print("=" * 70)
    
    from optiland.rays import RealRays
    
    # 参数
    wavelength_um = 10.64
    physical_size = 80.0  # mm
    n = 64
    w0 = 10.0  # mm
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    # 创建平面波前
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    amplitude = np.exp(-(X**2 + Y**2) / w0**2)
    wavefront = amplitude.astype(np.complex128)
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        wavelength=wavelength_um,
        physical_size=physical_size,
        num_rays=50,
    )
    
    rays_in = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    input_opd_waves = sampler.get_ray_opd()
    
    # 计算有效光束半径
    intensity = amplitude**2
    total_intensity = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total_intensity
    y_var = np.sum(Y**2 * intensity) / total_intensity
    beam_radius_mm = 2.0 * np.sqrt(max(x_var, y_var))
    effective_radius_mm = 3.0 * beam_radius_mm
    
    print(f"\n光束半径：{beam_radius_mm:.3f} mm")
    print(f"有效采样半径：{effective_radius_mm:.3f} mm")
    
    # 过滤光束内的光线
    ray_r = np.sqrt(ray_x**2 + ray_y**2)
    in_beam_mask = ray_r <= effective_radius_mm
    
    print(f"\n光束内光线数量：{np.sum(in_beam_mask)}/{len(ray_x)}")
    
    # 创建抛物面镜
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=50.0,
        conic=-1.0,
        tilt_x=0.0,
    )
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    # 综合掩模
    combined_mask = valid_mask & in_beam_mask
    
    # 取反元件 OPD
    element_opd_corrected = -element_opd_waves
    
    # 总 OPD
    total_opd_waves = input_opd_waves + element_opd_corrected
    
    print(f"\n光束内有效光线的 OPD 统计：")
    print(f"  输入 OPD 范围：[{np.min(input_opd_waves[combined_mask]):.6f}, "
          f"{np.max(input_opd_waves[combined_mask]):.6f}] waves")
    print(f"  元件 OPD 范围（取反后）：[{np.min(element_opd_corrected[combined_mask]):.6f}, "
          f"{np.max(element_opd_corrected[combined_mask]):.6f}] waves")
    print(f"  总 OPD 范围：[{np.min(total_opd_waves[combined_mask]):.6f}, "
          f"{np.max(total_opd_waves[combined_mask]):.6f}] waves")
    print(f"  总 OPD RMS：{np.std(total_opd_waves[combined_mask]):.6f} waves")
    
    # PROPER 理论 OPD
    wavelength_mm = wavelength_um * 1e-3
    proper_opd_theory = -ray_r**2 / (2 * f * wavelength_mm)
    
    print(f"\n  PROPER 理论 OPD 范围：[{np.min(proper_opd_theory[combined_mask]):.6f}, "
          f"{np.max(proper_opd_theory[combined_mask]):.6f}] waves")
    
    # 对比
    diff = total_opd_waves - proper_opd_theory
    print(f"\n  差异范围：[{np.min(diff[combined_mask]):.6f}, "
          f"{np.max(diff[combined_mask]):.6f}] waves")
    print(f"  差异 RMS：{np.std(diff[combined_mask]):.6f} waves")


def main():
    test_wavefront_sampler_opd()
    test_element_raytracer_opd()
    test_combined_opd()


if __name__ == "__main__":
    main()
