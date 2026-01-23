"""
调试 OPD 和相位的关系

关键问题：光线追迹 OPD 和波前相位之间的关系是什么？

对于反射镜：
- 几何 OPD = 2 × sag（边缘光线多走的距离，正值）
- 波前相位变化 = -k × 几何 OPD（反射导致相位反转）

这意味着：
- 边缘光线走更长路径（正 OPD）
- 但反射后边缘相位超前（负相位延迟）
- 所以：相位 = -k × OPD

验证这个关系是否正确。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_mirror_opd_phase():
    """分析反射镜的 OPD 和相位关系"""
    
    print("=" * 70)
    print("反射镜 OPD 和相位关系分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    R_mirror = 200.0  # 凹面镜曲率半径（正值）
    r = 10.0  # 边缘位置
    
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} um = {wavelength_mm} mm")
    print(f"  镜面曲率半径: {R_mirror} mm（凹面镜）")
    print(f"  边缘位置 r: {r} mm")
    print(f"  波数 k: {k:.2f} rad/mm")
    
    # 1. 几何分析
    print(f"\n" + "=" * 70)
    print("1. 几何分析")
    print("=" * 70)
    
    # 镜面 sag
    sag = r**2 / (2 * R_mirror)
    print(f"\n镜面 sag = r²/(2R) = {sag:.6f} mm")
    
    # 边缘光线多走的距离（反射，来回两次）
    geometric_opd = 2 * sag
    print(f"几何 OPD = 2 × sag = {geometric_opd:.6f} mm")
    print(f"         = {geometric_opd / wavelength_mm:.2f} waves")
    
    # 2. 相位分析
    print(f"\n" + "=" * 70)
    print("2. 相位分析")
    print("=" * 70)
    
    # 入射平面波：相位 = 0
    phase_in = 0
    print(f"\n入射波前（平面波）:")
    print(f"  边缘相位 = {phase_in} rad")
    
    # 反射后的相位变化
    # 关键：反射导致相位反转
    # 边缘光线多走 geometric_opd，但反射后相位变化是 -k × geometric_opd
    phase_change = -k * geometric_opd
    print(f"\n反射导致的相位变化:")
    print(f"  Δφ = -k × OPD = -{k:.2f} × {geometric_opd:.6f}")
    print(f"     = {phase_change:.2f} rad = {phase_change/(2*np.pi):.2f} waves")
    
    # 出射波前相位
    phase_out = phase_in + phase_change
    print(f"\n出射波前相位:")
    print(f"  φ_out = φ_in + Δφ = {phase_out:.2f} rad = {phase_out/(2*np.pi):.2f} waves")
    print(f"  负值表示边缘相位超前于主光线")
    
    # 3. Pilot Beam 分析
    print(f"\n" + "=" * 70)
    print("3. Pilot Beam 分析")
    print("=" * 70)
    
    # 出射波前曲率半径
    # 入射平面波（R_in = inf）经过凹面镜（f = R/2）反射后
    # 1/R_out = 1/R_in + 2/R_mirror = 0 + 2/R_mirror = 2/R_mirror
    # R_out = R_mirror/2 = 100 mm
    # 但是会聚波的曲率半径是负的（曲率中心在波前前方）
    R_out = -R_mirror / 2
    print(f"\n出射波前曲率半径:")
    print(f"  R_out = -R_mirror/2 = {R_out} mm（会聚波）")
    
    # Pilot Beam 相位公式
    pilot_phase = k * r**2 / (2 * R_out)
    print(f"\nPilot Beam 相位公式:")
    print(f"  φ_pilot = k × r²/(2R_out)")
    print(f"          = {k:.2f} × {r}²/(2×{R_out})")
    print(f"          = {pilot_phase:.2f} rad = {pilot_phase/(2*np.pi):.2f} waves")
    
    # 4. 比较
    print(f"\n" + "=" * 70)
    print("4. 比较")
    print("=" * 70)
    
    print(f"\n出射波前相位（从几何分析）: {phase_out:.2f} rad = {phase_out/(2*np.pi):.2f} waves")
    print(f"Pilot Beam 相位:            {pilot_phase:.2f} rad = {pilot_phase/(2*np.pi):.2f} waves")
    print(f"差异:                       {phase_out - pilot_phase:.6f} rad = {(phase_out - pilot_phase)/(2*np.pi):.6f} waves")
    
    if np.isclose(phase_out, pilot_phase, rtol=1e-6):
        print(f"\n[OK] 两者一致！")
    else:
        print(f"\n[ERROR] 两者不一致！")
    
    # 5. 结论
    print(f"\n" + "=" * 70)
    print("5. 结论")
    print("=" * 70)
    
    print("""
关键发现：

1. 几何 OPD 和波前相位的关系：
   - 几何 OPD = 2 × sag（正值，边缘光线多走的距离）
   - 波前相位变化 = -k × 几何 OPD（负值，反射导致相位反转）

2. Pilot Beam 相位公式验证：
   - φ_pilot = k × r²/(2R_out)
   - 当 R_out < 0（会聚波）时，φ_pilot < 0
   - 这与几何分析一致！

3. 光线追迹 OPD 和 Pilot Beam 相位的关系：
   - 光线追迹 OPD = 几何 OPD = 2 × sag（正值）
   - Pilot Beam 相位 = -k × 几何 OPD = k × r²/(2R_out)（负值）
   - 所以：Pilot Beam 相位 = -k × 光线追迹 OPD

4. 正确的残差计算：
   - 光线追迹相位 = -k × 光线追迹 OPD（转换为相位）
   - 残差相位 = 光线追迹相位 - Pilot Beam 相位
   - 对于理想球面镜，残差应该为 0
""")
    
    # 6. 验证正确的转换
    print(f"\n" + "=" * 70)
    print("6. 验证正确的转换")
    print("=" * 70)
    
    # 光线追迹 OPD（正值）
    raytracing_opd_mm = geometric_opd
    raytracing_opd_waves = raytracing_opd_mm / wavelength_mm
    print(f"\n光线追迹 OPD: {raytracing_opd_waves:.2f} waves（正值）")
    
    # 转换为相位（注意负号！）
    raytracing_phase = -k * raytracing_opd_mm
    print(f"光线追迹相位 = -k × OPD = {raytracing_phase:.2f} rad = {raytracing_phase/(2*np.pi):.2f} waves")
    
    # Pilot Beam 相位
    print(f"Pilot Beam 相位: {pilot_phase:.2f} rad = {pilot_phase/(2*np.pi):.2f} waves")
    
    # 残差
    residual_phase = raytracing_phase - pilot_phase
    print(f"残差相位: {residual_phase:.6f} rad = {residual_phase/(2*np.pi)*1000:.4f} milli-waves")
    
    if abs(residual_phase) < 1e-6:
        print(f"\n[OK] 残差接近 0，转换正确！")
    else:
        print(f"\n[ERROR] 残差不为 0，需要检查！")


def verify_with_actual_raytracing():
    """使用实际光线追迹验证"""
    
    print("\n" + "=" * 70)
    print("使用实际光线追迹验证")
    print("=" * 70)
    
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from hybrid_optical_propagation import PilotBeamParams
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 64
    physical_size_mm = 20.0
    num_rays = 50
    R_mirror = 200.0
    
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    # 创建入射平面波
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    amplitude = np.exp(-r_sq / w0_mm**2)
    phase = np.zeros_like(r_sq)  # 平面波
    
    # 光线采样
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size_mm,
        wavelength=wavelength_um,
        num_rays=num_rays,
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    
    # 光线追迹
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R_mirror,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
    )
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=(0, 0, -1),
    )
    
    output_rays = raytracer.trace(input_rays)
    
    # 获取结果
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    # 出射 Pilot Beam
    pilot_in = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    pilot_out = pilot_in.apply_mirror(R_mirror)
    R_out = pilot_out.curvature_radius_mm
    
    print(f"\n出射 Pilot Beam 曲率半径: {R_out:.2f} mm")
    
    # 对每条有效光线进行验证
    print(f"\n光线验证（选取几条代表性光线）:")
    print(f"{'r (mm)':<10} {'OPD (waves)':<15} {'理论OPD':<15} {'Pilot相位':<15} {'残差(mW)':<15}")
    print("-" * 70)
    
    r_out = np.sqrt(x_out**2 + y_out**2)
    
    # 选取几条不同位置的光线
    indices = np.where(valid_mask)[0]
    sorted_indices = indices[np.argsort(r_out[indices])]
    selected = sorted_indices[::len(sorted_indices)//5][:5]  # 选5条
    
    for idx in selected:
        r = r_out[idx]
        opd = opd_waves[idx]
        
        # 理论 OPD（几何）
        sag = r**2 / (2 * R_mirror)
        theoretical_opd_waves = 2 * sag / wavelength_mm
        
        # Pilot Beam 相位（转换为 waves）
        pilot_phase_rad = k * r**2 / (2 * R_out)
        pilot_phase_waves = pilot_phase_rad / (2 * np.pi)
        
        # 光线追迹相位（转换为 waves）
        # 相位 = -2π × OPD（注意负号！）
        raytracing_phase_waves = -opd
        
        # 残差
        residual_mw = (raytracing_phase_waves - pilot_phase_waves) * 1000
        
        print(f"{r:<10.4f} {opd:<15.2f} {theoretical_opd_waves:<15.2f} {pilot_phase_waves:<15.2f} {residual_mw:<15.4f}")
    
    print(f"\n说明:")
    print(f"  - OPD (waves): 光线追迹的相对 OPD（正值）")
    print(f"  - 理论OPD: 2×sag/λ（正值）")
    print(f"  - Pilot相位: k×r²/(2R_out)/(2π)（负值，因为 R_out < 0）")
    print(f"  - 残差: (-OPD) - Pilot相位（应该接近 0）")


if __name__ == "__main__":
    analyze_mirror_opd_phase()
    verify_with_actual_raytracing()
