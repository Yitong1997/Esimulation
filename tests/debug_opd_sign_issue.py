"""
诊断 OPD 符号问题

对比 ElementRaytracer 和 PROPER prop_lens 的 OPD 符号
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from optiland.rays import RealRays
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


def test_opd_sign_comparison():
    """对比 ElementRaytracer 和 PROPER prop_lens 的 OPD"""
    print("=" * 70)
    print("对比 ElementRaytracer 和 PROPER prop_lens 的 OPD 符号")
    print("=" * 70)
    
    # 参数
    wavelength_um = 10.64
    wavelength_m = wavelength_um * 1e-6
    f = -50.0  # mm，凸抛物面镜（发散）
    f_m = f * 1e-3
    
    # 测试点
    r_values_mm = np.array([0, 2, 4, 6, 8, 10])
    
    # =========================================================================
    # 1. PROPER prop_lens 的相位变化
    # =========================================================================
    print("\n1. PROPER prop_lens 的相位变化:")
    print("-" * 50)
    
    # PROPER 透镜相位公式：phase = -k * r² / (2f)
    # OPD = phase / (2π) = -r² / (2f * λ)
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm  # 1/mm
    
    proper_opd_waves = -r_values_mm**2 / (2 * f * wavelength_mm)
    
    print(f"焦距 f = {f} mm（凸面镜，发散）")
    print(f"PROPER 透镜相位公式：phase = -k * r² / (2f)")
    print(f"OPD = phase / (2π) = -r² / (2f * λ)")
    print()
    print(f"{'r (mm)':<10} {'PROPER OPD (waves)':<20}")
    print("-" * 30)
    for r, opd in zip(r_values_mm, proper_opd_waves):
        print(f"{r:<10.1f} {opd:<20.6f}")
    
    # =========================================================================
    # 2. ElementRaytracer 的 OPD
    # =========================================================================
    print("\n2. ElementRaytracer 的 OPD:")
    print("-" * 50)
    
    # 创建抛物面镜
    vertex_radius = 2 * f  # -100 mm
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,  # 抛物面
        tilt_x=0.0,
    )
    
    # 创建入射光线
    n_rays = len(r_values_mm)
    x = r_values_mm.copy()
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength_um)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    raytracer_opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"顶点曲率半径 R = {vertex_radius} mm")
    print(f"圆锥常数 k = -1（抛物面）")
    print()
    print(f"{'r (mm)':<10} {'Raytracer OPD (waves)':<20}")
    print("-" * 30)
    for r, opd in zip(r_values_mm, raytracer_opd_waves):
        print(f"{r:<10.1f} {opd:<20.6f}")
    
    # =========================================================================
    # 3. 对比
    # =========================================================================
    print("\n3. 对比:")
    print("-" * 50)
    print(f"{'r (mm)':<10} {'PROPER':<15} {'Raytracer':<15} {'差异':<15} {'比值':<10}")
    print("-" * 65)
    for r, proper_opd, rt_opd in zip(r_values_mm, proper_opd_waves, raytracer_opd_waves):
        diff = rt_opd - proper_opd
        ratio = rt_opd / proper_opd if abs(proper_opd) > 1e-10 else float('nan')
        print(f"{r:<10.1f} {proper_opd:<15.6f} {rt_opd:<15.6f} {diff:<15.6f} {ratio:<10.3f}")
    
    # =========================================================================
    # 4. 分析
    # =========================================================================
    print("\n4. 分析:")
    print("-" * 50)
    
    # 检查符号关系
    if len(proper_opd_waves) > 1 and len(raytracer_opd_waves) > 1:
        # 排除 r=0 的点
        proper_nonzero = proper_opd_waves[r_values_mm > 0]
        rt_nonzero = raytracer_opd_waves[r_values_mm > 0]
        
        if len(proper_nonzero) > 0 and len(rt_nonzero) > 0:
            # 检查符号
            proper_sign = np.sign(proper_nonzero[0])
            rt_sign = np.sign(rt_nonzero[0])
            
            if proper_sign == rt_sign:
                print("符号一致 ✓")
            else:
                print("符号相反 ✗")
                print(f"  PROPER: {'+' if proper_sign > 0 else '-'}")
                print(f"  Raytracer: {'+' if rt_sign > 0 else '-'}")
            
            # 检查比值
            ratios = rt_nonzero / proper_nonzero
            avg_ratio = np.mean(ratios)
            print(f"\n平均比值: {avg_ratio:.3f}")
            
            if np.allclose(ratios, 1.0, rtol=0.1):
                print("比值接近 1.0 ✓")
            elif np.allclose(ratios, -1.0, rtol=0.1):
                print("比值接近 -1.0（符号相反）")
                print("\n建议：在 _apply_element_hybrid 中对 element_opd_waves 取反")
            elif np.allclose(ratios, 2.0, rtol=0.1):
                print("比值接近 2.0（反射镜 OPD 加倍）")
            elif np.allclose(ratios, -2.0, rtol=0.1):
                print("比值接近 -2.0（符号相反 + 反射镜 OPD 加倍）")
                print("\n建议：在 _apply_element_hybrid 中对 element_opd_waves 取反并除以 2")


def test_flat_mirror_opd():
    """测试平面镜的 OPD（应该为 0）"""
    print("\n" + "=" * 70)
    print("测试平面镜的 OPD（应该为 0）")
    print("=" * 70)
    
    wavelength_um = 10.64
    
    # 创建平面镜
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=0.0,
        tilt_x=0.0,
    )
    
    # 创建入射光线
    r_values_mm = np.array([0, 2, 4, 6, 8, 10])
    n_rays = len(r_values_mm)
    x = r_values_mm.copy()
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength_um)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"\n{'r (mm)':<10} {'OPD (waves)':<20}")
    print("-" * 30)
    for r, opd in zip(r_values_mm, opd_waves):
        print(f"{r:<10.1f} {opd:<20.10f}")
    
    max_opd = np.max(np.abs(opd_waves))
    if max_opd < 1e-6:
        print(f"\n平面镜 OPD ≈ 0 ✓ (max = {max_opd:.2e} waves)")
    else:
        print(f"\n平面镜 OPD ≠ 0 ✗ (max = {max_opd:.6f} waves)")


def test_45deg_mirror_opd():
    """测试 45° 平面镜的 OPD（应该为 0）"""
    print("\n" + "=" * 70)
    print("测试 45° 平面镜的 OPD（应该为 0）")
    print("=" * 70)
    
    wavelength_um = 10.64
    
    # 创建 45° 平面镜
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=0.0,
        tilt_x=np.radians(45.0),
    )
    
    # 创建入射光线
    r_values_mm = np.array([0, 2, 4, 6, 8, 10])
    n_rays = len(r_values_mm)
    x = r_values_mm.copy()
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength_um)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    print(f"\n{'r (mm)':<10} {'OPD (waves)':<20}")
    print("-" * 30)
    for r, opd in zip(r_values_mm, opd_waves):
        print(f"{r:<10.1f} {opd:<20.10f}")
    
    max_opd = np.max(np.abs(opd_waves))
    if max_opd < 1e-6:
        print(f"\n45° 平面镜 OPD ≈ 0 ✓ (max = {max_opd:.2e} waves)")
    else:
        print(f"\n45° 平面镜 OPD ≠ 0 ✗ (max = {max_opd:.6f} waves)")


def main():
    test_opd_sign_comparison()
    test_flat_mirror_opd()
    test_45deg_mirror_opd()


if __name__ == "__main__":
    main()
