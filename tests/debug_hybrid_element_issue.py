"""
深入诊断 _apply_element_hybrid 方法的问题

问题已定位：元件追迹出错
本脚本分析具体原因
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from sequential_system import GaussianBeamSource, ParabolicMirror


def analyze_hybrid_element_step_by_step():
    """逐步分析 _apply_element_hybrid 的每个步骤"""
    print("=" * 70)
    print("逐步分析 _apply_element_hybrid 方法")
    print("=" * 70)
    
    # 参数
    wavelength_um = 10.64
    wavelength_m = wavelength_um * 1e-6
    w0 = 10.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3
    grid_size = 64
    beam_ratio = 0.25
    f1 = -50.0  # mm, OAP1 焦距
    
    # 创建 PROPER 波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 应用初始高斯光束
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # 高斯振幅
    amplitude = np.exp(-R_sq / w0**2)
    gaussian_field = amplitude.astype(np.complex128)
    gaussian_field_fft = proper.prop_shift_center(gaussian_field)
    wfo.wfarr = wfo.wfarr * gaussian_field_fft
    
    print(f"\n1. 初始波前状态:")
    print(f"   网格大小: {n}")
    print(f"   采样: {sampling_mm:.4f} mm/pixel")
    print(f"   物理尺寸: {2*half_size:.1f} mm")
    print(f"   reference_surface: {wfo.reference_surface}")
    print(f"   z: {wfo.z * 1e3:.3f} mm")
    print(f"   z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"   w0: {wfo.w0 * 1e3:.3f} mm")
    
    # 获取初始相位
    phase_before = proper.prop_get_phase(wfo).copy()
    amplitude_before = proper.prop_get_amplitude(wfo).copy()
    
    print(f"\n2. 初始波前相位统计:")
    print(f"   相位范围: [{np.min(phase_before):.4f}, {np.max(phase_before):.4f}] rad")
    print(f"   相位 RMS: {np.std(phase_before):.4f} rad")
    
    # =========================================================================
    # 模拟 _apply_element_hybrid 的步骤
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("模拟 _apply_element_hybrid 步骤")
    print("=" * 70)
    
    # 创建坐标网格（单位 m）
    half_size_m = sampling_m * n / 2
    coords_m = np.linspace(-half_size_m, half_size_m, n)
    X_m, Y_m = np.meshgrid(coords_m, coords_m)
    rho_sq_m = X_m**2 + Y_m**2
    
    k = 2 * np.pi / wfo.lamda  # 波数，单位 1/m
    
    # 步骤 1: 读取存储的复振幅，加回参考相位
    print("\n步骤 1: 读取复振幅，加回参考相位")
    
    reference_surface_old = wfo.reference_surface
    z_w0_old = wfo.z_w0
    
    stored_amplitude = proper.prop_shift_center(wfo.wfarr.copy())
    
    # 计算旧参考相位
    if reference_surface_old == "SPHERI" and abs(wfo.z - z_w0_old) > 1e-10:
        R_ref_old_m = wfo.z - z_w0_old
        old_reference_phase = -k * rho_sq_m / (2 * R_ref_old_m)
        print(f"   参考面类型: SPHERI")
        print(f"   参考球面半径: {R_ref_old_m * 1e3:.3f} mm")
    else:
        old_reference_phase = np.zeros((n, n))
        print(f"   参考面类型: PLANAR")
    
    stored_phase = np.angle(stored_amplitude)
    absolute_phase_in = stored_phase + old_reference_phase
    absolute_amplitude_in = np.abs(stored_amplitude)
    
    print(f"   存储相位范围: [{np.min(stored_phase):.4f}, {np.max(stored_phase):.4f}] rad")
    print(f"   参考相位范围: [{np.min(old_reference_phase):.4f}, {np.max(old_reference_phase):.4f}] rad")
    print(f"   绝对相位范围: [{np.min(absolute_phase_in):.4f}, {np.max(absolute_phase_in):.4f}] rad")
    
    # 步骤 2: 采样为光线
    print("\n步骤 2: 采样为光线")
    
    from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
    
    complex_amplitude_in = absolute_amplitude_in * np.exp(1j * absolute_phase_in)
    physical_size = sampling_mm * n
    
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=complex_amplitude_in,
        wavelength=wavelength_um,
        physical_size=physical_size,
        num_rays=100,
    )
    
    rays_in = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    input_opd_waves = sampler.get_ray_opd()
    
    print(f"   光线数量: {len(ray_x)}")
    print(f"   光线 x 范围: [{np.min(ray_x):.3f}, {np.max(ray_x):.3f}] mm")
    print(f"   入射 OPD 范围: [{np.min(input_opd_waves):.6f}, {np.max(input_opd_waves):.6f}] waves")
    print(f"   入射 OPD RMS: {np.std(input_opd_waves):.6f} waves")
    
    # 步骤 3: 光线追迹
    print("\n步骤 3: 光线追迹")
    
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    
    # 创建 OAP1 的 SurfaceDefinition
    # 注意：这里需要模拟 ParabolicMirror.get_surface_definition() 的行为
    vertex_radius = 2 * f1  # -100 mm
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=vertex_radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,  # 抛物面
        tilt_x=np.radians(45.0),
    )
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"   有效光线数: {np.sum(valid_mask)}")
    print(f"   元件 OPD 范围: [{np.min(element_opd_waves[valid_mask]):.6f}, {np.max(element_opd_waves[valid_mask]):.6f}] waves")
    print(f"   元件 OPD RMS: {np.std(element_opd_waves[valid_mask]):.6f} waves")
    
    # 总 OPD
    total_opd_waves = input_opd_waves + element_opd_waves
    print(f"\n   总 OPD 范围: [{np.min(total_opd_waves[valid_mask]):.6f}, {np.max(total_opd_waves[valid_mask]):.6f}] waves")
    print(f"   总 OPD RMS: {np.std(total_opd_waves[valid_mask]):.6f} waves")
    
    # 步骤 4: 对比 PROPER prop_lens 的结果
    print("\n" + "=" * 70)
    print("对比：PROPER prop_lens 的相位变化")
    print("=" * 70)
    
    # 创建新的波前用于对比
    wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    wfo2.wfarr = wfo2.wfarr * gaussian_field_fft
    
    phase_before_lens = proper.prop_get_phase(wfo2).copy()
    proper.prop_lens(wfo2, f1 * 1e-3)
    phase_after_lens = proper.prop_get_phase(wfo2).copy()
    
    phase_change_proper = phase_after_lens - phase_before_lens
    opd_proper_waves = phase_change_proper / (2 * np.pi)
    
    print(f"   PROPER 相位变化范围: [{np.min(phase_change_proper):.4f}, {np.max(phase_change_proper):.4f}] rad")
    print(f"   PROPER OPD 范围: [{np.min(opd_proper_waves):.6f}, {np.max(opd_proper_waves):.6f}] waves")
    
    # 理论透镜相位
    # phase = -k * r² / (2f)
    # OPD = -r² / (2f * λ)
    wavelength_mm = wavelength_um * 1e-3
    opd_theory_waves = -R_sq / (2 * f1 * wavelength_mm)
    
    print(f"   理论 OPD 范围: [{np.min(opd_theory_waves):.6f}, {np.max(opd_theory_waves):.6f}] waves")
    
    # 关键对比：光线追迹的元件 OPD vs PROPER 的透镜 OPD
    print("\n" + "=" * 70)
    print("关键对比：光线追迹 OPD vs PROPER OPD")
    print("=" * 70)
    
    # 在光线位置采样 PROPER OPD
    from scipy.interpolate import RectBivariateSpline
    
    interp = RectBivariateSpline(coords, coords, opd_proper_waves)
    proper_opd_at_rays = interp(ray_y, ray_x, grid=False)
    
    print(f"\n光线位置处的 OPD 对比:")
    print(f"{'光线':<6} {'x (mm)':<10} {'y (mm)':<10} {'追迹 OPD':<15} {'PROPER OPD':<15} {'差异':<15}")
    print("-" * 75)
    
    for i in range(min(10, len(ray_x))):
        if valid_mask[i]:
            diff = element_opd_waves[i] - proper_opd_at_rays[i]
            print(f"{i:<6} {ray_x[i]:<10.3f} {ray_y[i]:<10.3f} {element_opd_waves[i]:<15.6f} {proper_opd_at_rays[i]:<15.6f} {diff:<15.6f}")
    
    # 统计差异
    valid_diff = element_opd_waves[valid_mask] - proper_opd_at_rays[valid_mask]
    print(f"\n差异统计:")
    print(f"   平均差异: {np.mean(valid_diff):.6f} waves")
    print(f"   差异 RMS: {np.std(valid_diff):.6f} waves")
    print(f"   最大差异: {np.max(np.abs(valid_diff)):.6f} waves")
    
    # 分析问题
    print("\n" + "=" * 70)
    print("问题分析")
    print("=" * 70)
    
    ratio = np.mean(np.abs(element_opd_waves[valid_mask])) / np.mean(np.abs(proper_opd_at_rays[valid_mask]))
    print(f"\n光线追迹 OPD / PROPER OPD 比值: {ratio:.2f}")
    
    if ratio > 10:
        print("\n⚠️ 光线追迹的 OPD 比 PROPER 大很多倍！")
        print("   可能原因:")
        print("   1. OPD 单位不一致（mm vs waves）")
        print("   2. 参考面处理错误")
        print("   3. 倾斜元件的 OPD 计算错误")
    elif ratio < 0.1:
        print("\n⚠️ 光线追迹的 OPD 比 PROPER 小很多倍！")
    else:
        print("\n✓ OPD 量级基本一致")
        
        if np.std(valid_diff) > 0.1:
            print("   但存在显著的形状差异")


def analyze_element_opd_calculation():
    """分析 ElementRaytracer 的 OPD 计算"""
    print("\n" + "=" * 70)
    print("分析 ElementRaytracer 的 OPD 计算")
    print("=" * 70)
    
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
    # 简单测试：平行光入射凸抛物面镜（无倾斜）
    f1 = -50.0  # mm
    vertex_radius = 2 * f1  # -100 mm
    wavelength = 10.64  # μm
    
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
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    
    # 理论 OPD（透镜相位）
    # 对于凸抛物面镜，f = R/2 = -50 mm
    # OPD = -r² / (2f * λ)
    wavelength_mm = wavelength * 1e-3
    r_sq = x**2
    opd_theory = -r_sq / (2 * f1 * wavelength_mm)
    
    print(f"\n无倾斜凸抛物面镜 OPD 对比:")
    print(f"{'x (mm)':<10} {'追迹 OPD':<15} {'理论 OPD':<15} {'差异':<15}")
    print("-" * 55)
    
    for i in range(n_rays):
        diff = opd_waves[i] - opd_theory[i]
        print(f"{x[i]:<10.2f} {opd_waves[i]:<15.6f} {opd_theory[i]:<15.6f} {diff:<15.6f}")
    
    # 检查 OPD 的符号和量级
    print(f"\n追迹 OPD 范围: [{np.min(opd_waves):.6f}, {np.max(opd_waves):.6f}] waves")
    print(f"理论 OPD 范围: [{np.min(opd_theory):.6f}, {np.max(opd_theory):.6f}] waves")
    
    ratio = np.mean(np.abs(opd_waves)) / np.mean(np.abs(opd_theory)) if np.mean(np.abs(opd_theory)) > 1e-10 else float('inf')
    print(f"比值: {ratio:.4f}")


def main():
    analyze_hybrid_element_step_by_step()
    analyze_element_opd_calculation()


if __name__ == "__main__":
    main()
