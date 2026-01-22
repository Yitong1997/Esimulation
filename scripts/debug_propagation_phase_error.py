"""
调试传播相位误差的真正来源
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper

from hybrid_optical_propagation import PilotBeamParams, GridSampling, SourceDefinition
from hybrid_optical_propagation.state_converter import StateConverter


def analyze_propagation_error():
    """分析传播误差的真正来源"""
    print("=" * 80)
    print("分析传播误差的真正来源")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 创建初始波前
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    grid_sampling = source.get_grid_sampling()
    
    print(f"\n初始状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  初始相位范围: [{np.min(phase):.4f}, {np.max(phase):.4f}] rad")
    
    # 计算初始 Pilot Beam 相位
    pilot_phase_initial = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase_initial):.4f}, {np.max(pilot_phase_initial):.4f}] rad")
    
    # 初始相位与 Pilot Beam 的差异
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    initial_diff = phase - pilot_phase_initial
    print(f"  初始相位与 Pilot Beam 差异 RMS: {np.sqrt(np.mean(initial_diff[valid_mask]**2)):.6f} rad")
    
    # 传播
    propagation_distance_mm = 3 * z_R_mm
    print(f"\n传播距离: {propagation_distance_mm:.1f} mm (3×z_R)")
    
    proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
    
    # 更新 Pilot Beam
    new_pilot_params = pilot_params.propagate(propagation_distance_mm)
    
    print(f"\n传播后状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  wfo.z: {wfo.z * 1e3:.1f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.1f} mm")
    
    # 从 PROPER 提取原始数据
    proper_amplitude = proper.prop_get_amplitude(wfo)
    proper_phase = proper.prop_get_phase(wfo)  # 残差相位，折叠的
    
    print(f"\nPROPER 提取的残差相位:")
    print(f"  范围: [{np.min(proper_phase):.4f}, {np.max(proper_phase):.4f}] rad")
    print(f"  RMS: {np.sqrt(np.mean(proper_phase**2)):.4f} rad")
    
    # 计算 PROPER 参考面相位
    converter = StateConverter(wavelength_um)
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    print(f"\nPROPER 参考面相位:")
    print(f"  范围: [{np.min(proper_ref_phase):.4f}, {np.max(proper_ref_phase):.4f}] rad")
    
    # 重建绝对相位（加回参考面相位）
    absolute_phase = proper_ref_phase + proper_phase
    
    print(f"\n重建的绝对相位（PROPER 参考面 + 残差）:")
    print(f"  范围: [{np.min(absolute_phase):.4f}, {np.max(absolute_phase):.4f}] rad")
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = new_pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    print(f"\nPilot Beam 参考相位:")
    print(f"  范围: [{np.min(pilot_phase):.4f}, {np.max(pilot_phase):.4f}] rad")
    print(f"  曲率半径: {new_pilot_params.curvature_radius_mm:.1f} mm")
    
    # 比较绝对相位与 Pilot Beam 相位
    valid_mask = proper_amplitude > 0.01 * np.max(proper_amplitude)
    
    # 直接比较（不解包裹）
    direct_diff = absolute_phase - pilot_phase
    print(f"\n绝对相位与 Pilot Beam 相位的直接差异:")
    print(f"  RMS: {np.sqrt(np.mean(direct_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(direct_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    print(f"  MAX: {np.max(np.abs(direct_diff[valid_mask])):.4f} rad")
    
    # 使用 angle(exp(1j * diff)) 处理 2π 周期性
    wrapped_diff = np.angle(np.exp(1j * direct_diff))
    print(f"\n使用 angle(exp(1j*diff)) 处理后的差异:")
    print(f"  RMS: {np.sqrt(np.mean(wrapped_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(wrapped_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    
    # 分析：理想高斯光束传播后的相位应该是什么？
    print("\n" + "=" * 80)
    print("理论分析：理想高斯光束传播后的相位")
    print("=" * 80)
    
    # 理想高斯光束在 z 处的相位（相对于主光线）
    # φ(r, z) = k * r² / (2 * R(z))
    # 其中 R(z) = z * (1 + (z_R/z)²) 是严格公式
    
    z_mm = propagation_distance_mm
    R_strict_mm = z_mm * (1 + (z_R_mm / z_mm)**2)
    
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_mm = X_mm**2 + Y_mm**2
    k = 2 * np.pi / wavelength_mm
    
    # 理想相位（严格公式）
    ideal_phase_strict = k * r_sq_mm / (2 * R_strict_mm)
    
    print(f"\n理想相位（严格公式，R = {R_strict_mm:.1f} mm）:")
    print(f"  范围: [{np.min(ideal_phase_strict):.4f}, {np.max(ideal_phase_strict):.4f}] rad")
    
    # PROPER 使用的曲率半径
    R_proper_mm = z_mm  # 远场近似
    ideal_phase_proper = k * r_sq_mm / (2 * R_proper_mm)
    
    print(f"\n理想相位（PROPER 远场近似，R = {R_proper_mm:.1f} mm）:")
    print(f"  范围: [{np.min(ideal_phase_proper):.4f}, {np.max(ideal_phase_proper):.4f}] rad")
    
    # 两种理想相位的差异
    ideal_diff = ideal_phase_strict - ideal_phase_proper
    print(f"\n两种理想相位的差异:")
    print(f"  RMS: {np.sqrt(np.mean(ideal_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(ideal_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    
    # 比较 PROPER 重建的绝对相位与 PROPER 理想相位
    proper_ideal_diff = absolute_phase - ideal_phase_proper
    print(f"\nPROPER 重建相位与 PROPER 理想相位的差异:")
    print(f"  RMS: {np.sqrt(np.mean(proper_ideal_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(proper_ideal_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
误差来源分析：

1. PROPER 内部使用远场近似曲率半径 R_proper = z
2. Pilot Beam 使用严格公式 R_strict = z * (1 + (z_R/z)²)
3. 在 z = 3×z_R 时，两者差异约 10%

4. PROPER 传播后的相位是基于 R_proper 的
5. 我们用 Pilot Beam 相位（基于 R_strict）来比较
6. 这导致了系统性的相位差异

解决方案：
- 方案 A：接受这个差异，因为它是 PROPER 内部的近似
- 方案 B：在比较时使用 PROPER 的曲率半径，而不是 Pilot Beam 的
- 方案 C：只在真正的远场（z >> z_R）使用 PROPER 传播
""")


if __name__ == "__main__":
    analyze_propagation_error()
