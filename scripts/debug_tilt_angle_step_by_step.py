"""
逐步调试倾斜角度误差来源

对比 0°（高精度）和 5°（低精度）两种情况，
追踪数据流中每一步的数值，定位误差首次出现的位置。

数据流：
1. 入射面波前（amplitude, phase）
2. WavefrontToRaysSampler 采样光线
3. ElementRaytracer 光线追迹
4. 出射光线位置和 OPD
5. RayToWavefrontReconstructor 重建
6. 出射面波前

不修改任何内部代码。
"""

import sys
from pathlib import Path
import numpy as np

# 设置路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import warnings
warnings.filterwarnings('ignore')


def analyze_single_angle(tilt_deg: float):
    """详细分析单个角度的数据流"""
    from hybrid_simulation import HybridSimulator
    
    print(f"\n{'='*70}")
    print(f"分析倾斜角度: {tilt_deg}°")
    print(f"{'='*70}")
    
    # 创建仿真器
    sim = HybridSimulator(verbose=False)
    sim.add_flat_mirror(z=50.0, tilt_x=tilt_deg, aperture=30.0)
    sim.set_source(wavelength_um=0.633, w0_mm=5.0, grid_size=256, physical_size_mm=40.0)
    
    # 运行仿真
    result = sim.run()
    
    if not result.success:
        print(f"仿真失败: {result.error_message}")
        return None
    
    # 获取入射面和出射面数据
    entrance_wf = None
    exit_wf = None
    for surface in result.surfaces:
        if surface.entrance is not None:
            entrance_wf = surface.entrance
        if surface.exit is not None:
            exit_wf = surface.exit
    
    if entrance_wf is None or exit_wf is None:
        print("无法获取入射面或出射面数据")
        return None
    
    # 分析入射面
    print(f"\n--- 入射面分析 ---")
    print(f"振幅范围: [{np.min(entrance_wf.amplitude):.6f}, {np.max(entrance_wf.amplitude):.6f}]")
    print(f"相位范围: [{np.min(entrance_wf.phase):.6f}, {np.max(entrance_wf.phase):.6f}] rad")
    
    entrance_residual = entrance_wf.get_residual_phase()
    valid_mask = entrance_wf.amplitude > 0.01 * np.max(entrance_wf.amplitude)
    entrance_rms = np.std(entrance_residual[valid_mask]) / (2 * np.pi) * 1000
    print(f"入射面残差 RMS: {entrance_rms:.3f} milli-waves")
    
    # 分析出射面
    print(f"\n--- 出射面分析 ---")
    print(f"振幅范围: [{np.min(exit_wf.amplitude):.6f}, {np.max(exit_wf.amplitude):.6f}]")
    print(f"相位范围: [{np.min(exit_wf.phase):.6f}, {np.max(exit_wf.phase):.6f}] rad")
    
    exit_residual = exit_wf.get_residual_phase()
    valid_mask = exit_wf.amplitude > 0.01 * np.max(exit_wf.amplitude)
    exit_rms = np.std(exit_residual[valid_mask]) / (2 * np.pi) * 1000
    print(f"出射面残差 RMS: {exit_rms:.3f} milli-waves")
    
    # 分析 Pilot Beam 参数
    print(f"\n--- Pilot Beam 参数 ---")
    print(f"入射面曲率半径: {entrance_wf.pilot_beam.curvature_radius_mm:.2f} mm")
    print(f"出射面曲率半径: {exit_wf.pilot_beam.curvature_radius_mm:.2f} mm")
    
    return {
        'angle_deg': tilt_deg,
        'entrance_rms': entrance_rms,
        'exit_rms': exit_rms,
        'entrance_wf': entrance_wf,
        'exit_wf': exit_wf,
    }


def deep_debug_raytracing(tilt_deg: float):
    """深入调试光线追迹过程"""
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
    
    print(f"\n{'='*70}")
    print(f"深入调试光线追迹: {tilt_deg}°")
    print(f"{'='*70}")
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 256
    physical_size_mm = 40.0
    z_mm = 50.0
    
    # 创建理想高斯光束入射波前
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    # 高斯振幅
    w_z = w0_mm * np.sqrt(1 + (z_mm / z_R)**2)
    amplitude = np.exp(-r_sq / w_z**2)
    
    # 高斯相位（近场，曲率半径很大）
    R = z_mm * (1 + (z_R / z_mm)**2) if z_mm != 0 else np.inf
    k = 2 * np.pi / wavelength_mm
    if np.isinf(R):
        phase = np.zeros_like(r_sq)
    else:
        phase = k * r_sq / (2 * R)
    
    print(f"\n--- 输入波前 ---")
    print(f"振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
    print(f"相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"Pilot Beam 曲率半径: {R:.2f} mm")
    
    # 采样光线
    print(f"\n--- 光线采样 ---")
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size_mm,
        wavelength=wavelength_um,
        num_rays=200,
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    ray_opd = sampler.get_ray_opd()
    
    x_in = np.asarray(input_rays.x)
    y_in = np.asarray(input_rays.y)
    z_in = np.asarray(input_rays.z)
    L_in = np.asarray(input_rays.L)
    M_in = np.asarray(input_rays.M)
    N_in = np.asarray(input_rays.N)
    
    print(f"光线数量: {len(x_in)}")
    print(f"x 范围: [{np.min(x_in):.4f}, {np.max(x_in):.4f}] mm")
    print(f"y 范围: [{np.min(y_in):.4f}, {np.max(y_in):.4f}] mm")
    print(f"z 范围: [{np.min(z_in):.4f}, {np.max(z_in):.4f}] mm")
    print(f"方向余弦 L 范围: [{np.min(L_in):.6f}, {np.max(L_in):.6f}]")
    print(f"方向余弦 M 范围: [{np.min(M_in):.6f}, {np.max(M_in):.6f}]")
    print(f"方向余弦 N 范围: [{np.min(N_in):.6f}, {np.max(N_in):.6f}]")
    print(f"OPD 范围: [{np.min(ray_opd):.6f}, {np.max(ray_opd):.6f}] waves")
    
    # 创建表面定义
    tilt_rad = np.radians(tilt_deg)
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
    
    print(f"\n--- 表面定义 ---")
    print(f"类型: {surface_def.surface_type}")
    print(f"倾斜角 X: {np.degrees(surface_def.tilt_x):.2f}°")
    print(f"倾斜角 Y: {np.degrees(surface_def.tilt_y):.2f}°")
    
    # 计算出射方向（用于 ElementRaytracer）
    # 反射后的主光线方向
    # 入射方向: (0, 0, 1)
    # 表面法向量: 绕 X 轴旋转 tilt_rad 后的 (0, 0, -1)
    # 即 (0, sin(tilt_rad), -cos(tilt_rad))
    # 反射方向: d_out = d_in - 2*(d_in·n)*n
    
    d_in = np.array([0, 0, 1])
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    
    print(f"\n--- 光轴方向 ---")
    print(f"入射方向: {d_in}")
    print(f"表面法向量: {n}")
    print(f"出射方向: {d_out}")
    
    # 光线追迹
    print(f"\n--- 光线追迹 ---")
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=tuple(d_out),
    )
    
    output_rays = raytracer.trace(input_rays)
    
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    opd_out = np.asarray(output_rays.opd)
    
    print(f"出射 x 范围: [{np.min(x_out):.4f}, {np.max(x_out):.4f}] mm")
    print(f"出射 y 范围: [{np.min(y_out):.4f}, {np.max(y_out):.4f}] mm")
    print(f"出射 z 范围: [{np.min(z_out):.4f}, {np.max(z_out):.4f}] mm")
    print(f"出射 L 范围: [{np.min(L_out):.6f}, {np.max(L_out):.6f}]")
    print(f"出射 M 范围: [{np.min(M_out):.6f}, {np.max(M_out):.6f}]")
    print(f"出射 N 范围: [{np.min(N_out):.6f}, {np.max(N_out):.6f}]")
    print(f"出射 OPD 范围: [{np.min(opd_out):.6f}, {np.max(opd_out):.6f}] mm")
    
    # 分析坐标变换
    print(f"\n--- 坐标变换分析 ---")
    
    # 入射面到出射面的位置变化
    dx = x_out - x_in
    dy = y_out - y_in
    dz = z_out - z_in
    
    print(f"Δx 范围: [{np.min(dx):.6f}, {np.max(dx):.6f}] mm")
    print(f"Δy 范围: [{np.min(dy):.6f}, {np.max(dy):.6f}] mm")
    print(f"Δz 范围: [{np.min(dz):.6f}, {np.max(dz):.6f}] mm")
    
    # 对于平面镜，理想情况下：
    # - 0° 倾斜：出射位置 = 入射位置
    # - 45° 倾斜：出射位置在 Y-Z 平面内旋转
    
    # 计算出射位置相对于理论值的偏差
    # 理论出射位置：入射点在镜面上的反射
    
    # 镜面方程：y*sin(tilt) - z*cos(tilt) = 0（通过原点）
    # 入射光线：(x_in, y_in, 0) + t*(L_in, M_in, N_in)
    # 交点：t = (y_in*sin(tilt)) / (N_in*cos(tilt) - M_in*sin(tilt))
    
    # 简化：对于近轴光线，入射点近似在 z=0 平面
    # 反射后，出射点应该在出射面上（垂直于出射方向）
    
    # 检查出射光线是否平行于出射主光线方向
    L_expected = d_out[0]
    M_expected = d_out[1]
    N_expected = d_out[2]
    
    dL = L_out - L_expected
    dM = M_out - M_expected
    dN = N_out - N_expected
    
    print(f"\n--- 出射方向偏差 ---")
    print(f"期望出射方向: ({L_expected:.6f}, {M_expected:.6f}, {N_expected:.6f})")
    print(f"ΔL 范围: [{np.min(dL):.6f}, {np.max(dL):.6f}]")
    print(f"ΔM 范围: [{np.min(dM):.6f}, {np.max(dM):.6f}]")
    print(f"ΔN 范围: [{np.min(dN):.6f}, {np.max(dN):.6f}]")
    
    # OPD 分析
    print(f"\n--- OPD 分析 ---")
    
    # 对于平面镜，理想 OPD = 0（所有光线光程相同）
    # 但由于入射波前有曲率，OPD 应该反映这个曲率
    
    # 计算理论 OPD（基于入射波前曲率）
    # 入射 OPD = r²/(2R)
    r_sq_in = x_in**2 + y_in**2
    if np.isinf(R):
        theoretical_opd_in = np.zeros_like(r_sq_in)
    else:
        theoretical_opd_in = r_sq_in / (2 * R)
    
    # 出射 OPD 应该等于入射 OPD（平面镜不改变 OPD）
    # 但需要考虑坐标变换
    
    opd_out_waves = opd_out / wavelength_mm
    theoretical_opd_waves = theoretical_opd_in / wavelength_mm
    
    print(f"理论入射 OPD 范围: [{np.min(theoretical_opd_waves):.6f}, {np.max(theoretical_opd_waves):.6f}] waves")
    print(f"实际出射 OPD 范围: [{np.min(opd_out_waves):.6f}, {np.max(opd_out_waves):.6f}] waves")
    
    opd_diff = opd_out_waves - theoretical_opd_waves
    print(f"OPD 差异范围: [{np.min(opd_diff):.6f}, {np.max(opd_diff):.6f}] waves")
    print(f"OPD 差异 RMS: {np.std(opd_diff):.6f} waves = {np.std(opd_diff)*1000:.3f} milli-waves")
    
    # 重建波前
    print(f"\n--- 波前重建 ---")
    
    sampling_mm = physical_size_mm / grid_size
    reconstructor = RayToWavefrontReconstructor(
        grid_size=grid_size,
        sampling_mm=sampling_mm,
        wavelength_um=wavelength_um,
    )
    
    valid_mask = np.ones(len(x_in), dtype=bool)
    
    # 计算残差 OPD（相对于出射面 Pilot Beam）
    # 出射面 Pilot Beam 曲率半径 = 入射面曲率半径（平面镜不改变曲率）
    r_sq_out = x_out**2 + y_out**2
    if np.isinf(R):
        pilot_opd_out = np.zeros_like(r_sq_out)
    else:
        pilot_opd_out = r_sq_out / (2 * R)
    
    pilot_opd_out_waves = pilot_opd_out / wavelength_mm
    
    # 残差 OPD = 实际 OPD - Pilot Beam OPD
    residual_opd_waves = opd_out_waves - pilot_opd_out_waves
    
    print(f"Pilot Beam OPD 范围: [{np.min(pilot_opd_out_waves):.6f}, {np.max(pilot_opd_out_waves):.6f}] waves")
    print(f"残差 OPD 范围: [{np.min(residual_opd_waves):.6f}, {np.max(residual_opd_waves):.6f}] waves")
    print(f"残差 OPD RMS: {np.std(residual_opd_waves):.6f} waves = {np.std(residual_opd_waves)*1000:.3f} milli-waves")
    
    # 在光线位置处插值输入振幅
    from scipy.interpolate import RegularGridInterpolator
    amp_interp = RegularGridInterpolator(
        (coords, coords),
        amplitude,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    ray_points = np.column_stack([y_in, x_in])
    input_amplitude_at_rays = amp_interp(ray_points)
    
    # 重建
    exit_amplitude, residual_phase = reconstructor.reconstruct_amplitude_phase(
        ray_x_in=x_in,
        ray_y_in=y_in,
        ray_x_out=x_out,
        ray_y_out=y_out,
        opd_waves=residual_opd_waves,
        valid_mask=valid_mask,
        input_amplitude=input_amplitude_at_rays,
    )
    
    print(f"\n重建振幅范围: [{np.min(exit_amplitude):.6f}, {np.max(exit_amplitude):.6f}]")
    print(f"重建残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")
    
    # 加回 Pilot Beam 相位
    if np.isinf(R):
        pilot_phase_grid = np.zeros((grid_size, grid_size))
    else:
        pilot_phase_grid = k * (X**2 + Y**2) / (2 * R)
    
    exit_phase = residual_phase + pilot_phase_grid
    
    # 计算最终残差
    valid_mask_grid = exit_amplitude > 0.01 * np.max(exit_amplitude)
    final_residual = exit_phase - pilot_phase_grid
    final_residual_valid = final_residual[valid_mask_grid]
    final_residual_valid = final_residual_valid - np.mean(final_residual_valid)
    
    final_rms_waves = np.std(final_residual_valid) / (2 * np.pi)
    print(f"\n最终残差 RMS: {final_rms_waves:.6f} waves = {final_rms_waves*1000:.3f} milli-waves")
    
    return {
        'angle_deg': tilt_deg,
        'opd_diff_rms': np.std(opd_diff) * 1000,
        'residual_opd_rms': np.std(residual_opd_waves) * 1000,
        'final_rms': final_rms_waves * 1000,
        'x_in': x_in,
        'y_in': y_in,
        'x_out': x_out,
        'y_out': y_out,
        'opd_out_waves': opd_out_waves,
        'theoretical_opd_waves': theoretical_opd_waves,
        'residual_opd_waves': residual_opd_waves,
    }


def main():
    print("=" * 70)
    print("逐步调试倾斜角度误差来源")
    print("=" * 70)
    
    # 对比 0° 和 5°
    print("\n" + "=" * 70)
    print("第一部分：高层分析")
    print("=" * 70)
    
    result_0 = analyze_single_angle(0.0)
    result_5 = analyze_single_angle(5.0)
    result_45 = analyze_single_angle(45.0)
    
    print("\n" + "=" * 70)
    print("第二部分：深入光线追迹调试")
    print("=" * 70)
    
    debug_0 = deep_debug_raytracing(0.0)
    debug_5 = deep_debug_raytracing(5.0)
    debug_45 = deep_debug_raytracing(45.0)
    
    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    print(f"\n{'角度':>8} | {'OPD差异RMS':>15} | {'残差OPD RMS':>15} | {'最终RMS':>15}")
    print("-" * 60)
    for d in [debug_0, debug_5, debug_45]:
        if d:
            print(f"{d['angle_deg']:>8.1f}° | "
                  f"{d['opd_diff_rms']:>12.3f} mw | "
                  f"{d['residual_opd_rms']:>12.3f} mw | "
                  f"{d['final_rms']:>12.3f} mw")
    
    print("\n关键发现：")
    if debug_0 and debug_5:
        print(f"  0° OPD 差异 RMS: {debug_0['opd_diff_rms']:.3f} milli-waves")
        print(f"  5° OPD 差异 RMS: {debug_5['opd_diff_rms']:.3f} milli-waves")
        print(f"  差异: {debug_5['opd_diff_rms'] - debug_0['opd_diff_rms']:.3f} milli-waves")
        
        if debug_5['opd_diff_rms'] > debug_0['opd_diff_rms'] + 1:
            print("\n  → 误差首次出现在 ElementRaytracer 的 OPD 计算中")
        else:
            print("\n  → OPD 计算正确，误差可能在重建过程中")


if __name__ == "__main__":
    main()
