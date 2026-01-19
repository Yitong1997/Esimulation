"""
诊断 OPD 累积问题

问题：WavefrontToRaysSampler 创建的光线已经有了 OPD，
然后 ElementRaytracer 又累积了更多的 OPD
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.rays import RealRays

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


def main():
    print("=" * 70)
    print("诊断 OPD 累积问题")
    print("=" * 70)
    
    # 参数
    wavelength = 10.64  # μm
    physical_size = 160.0  # mm
    n = 64
    w0 = 10.0  # mm
    
    # 创建平面波前（相位为零）
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    amplitude = np.exp(-(X**2 + Y**2) / w0**2)
    wavefront = amplitude.astype(np.complex128)  # 相位为零
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        wavelength=wavelength,
        physical_size=physical_size,
        num_rays=100,
    )
    
    rays_from_sampler = sampler.get_output_rays()
    
    print("\n1. WavefrontToRaysSampler 输出的光线:")
    print(f"   光线数量: {len(rays_from_sampler.x)}")
    
    # 检查光线的原始 OPD
    opd_raw = np.asarray(rays_from_sampler.opd)
    print(f"   原始 OPD (mm): [{np.min(opd_raw):.6f}, {np.max(opd_raw):.6f}]")
    print(f"   原始 OPD 均值: {np.mean(opd_raw):.6f} mm")
    
    # 检查光线位置
    x = np.asarray(rays_from_sampler.x)
    y = np.asarray(rays_from_sampler.y)
    z = np.asarray(rays_from_sampler.z)
    print(f"   x 范围: [{np.min(x):.3f}, {np.max(x):.3f}] mm")
    print(f"   y 范围: [{np.min(y):.3f}, {np.max(y):.3f}] mm")
    print(f"   z 范围: [{np.min(z):.6f}, {np.max(z):.6f}] mm")
    
    # 检查光线方向
    L = np.asarray(rays_from_sampler.L)
    M = np.asarray(rays_from_sampler.M)
    N = np.asarray(rays_from_sampler.N)
    print(f"   L 范围: [{np.min(L):.6f}, {np.max(L):.6f}]")
    print(f"   M 范围: [{np.min(M):.6f}, {np.max(M):.6f}]")
    print(f"   N 范围: [{np.min(N):.6f}, {np.max(N):.6f}]")
    
    # 获取采样器计算的相对 OPD
    sampler_opd_waves = sampler.get_ray_opd()
    print(f"\n   采样器相对 OPD (waves): [{np.min(sampler_opd_waves):.6f}, {np.max(sampler_opd_waves):.6f}]")
    
    # 现在将这些光线传给 ElementRaytracer
    print("\n2. 将光线传给 ElementRaytracer:")
    
    f = -50.0  # mm
    vertex_radius = 2 * f
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,
        tilt_x=np.radians(45.0),
    )
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    # 追迹
    rays_out = raytracer.trace(rays_from_sampler)
    element_opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"   有效光线数: {np.sum(valid_mask)}")
    print(f"   元件 OPD (waves): [{np.nanmin(element_opd_waves):.6f}, {np.nanmax(element_opd_waves):.6f}]")
    print(f"   元件 OPD RMS: {np.nanstd(element_opd_waves[valid_mask]):.6f} waves")
    
    # 检查出射光线的原始 OPD
    out_opd_raw = np.asarray(rays_out.opd)
    print(f"\n   出射光线原始 OPD (mm): [{np.min(out_opd_raw[valid_mask]):.6f}, {np.max(out_opd_raw[valid_mask]):.6f}]")
    
    # 对比：手动创建光线（OPD=0）
    print("\n3. 对比：手动创建光线（初始 OPD=0）:")
    
    # 使用相同的位置，但 OPD 初始化为 0
    rays_manual = RealRays(
        x.copy(), y.copy(), np.zeros_like(z),  # z=0
        L.copy(), M.copy(), N.copy(),
        np.ones_like(x), wavelength
    )
    
    print(f"   手动光线初始 OPD: {np.asarray(rays_manual.opd)[0]:.6f} mm")
    
    raytracer2 = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out2 = raytracer2.trace(rays_manual)
    element_opd_waves2 = raytracer2.get_relative_opd_waves()
    valid_mask2 = raytracer2.get_valid_ray_mask()
    
    print(f"   有效光线数: {np.sum(valid_mask2)}")
    print(f"   元件 OPD (waves): [{np.nanmin(element_opd_waves2):.6f}, {np.nanmax(element_opd_waves2):.6f}]")
    print(f"   元件 OPD RMS: {np.nanstd(element_opd_waves2[valid_mask2]):.6f} waves")
    
    # 分析
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)
    
    print(f"""
问题分析：
1. WavefrontToRaysSampler 输出的光线已经有了 OPD = {np.mean(opd_raw):.1f} mm
   这是从物面（无穷远）到相位面的光程

2. ElementRaytracer 使用这些光线时，会在现有 OPD 基础上累积
   导致最终 OPD 非常大

3. 手动创建的光线（OPD=0）追迹后的 OPD 是合理的

解决方案：
- 在 _apply_element_hybrid 中，应该重置光线的 OPD 为 0
- 或者使用 sampler.get_ray_opd() 作为入射 OPD，而不是光线的原始 OPD
""")


if __name__ == "__main__":
    main()
