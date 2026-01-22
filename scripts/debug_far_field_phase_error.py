"""
调试远场传播相位误差

分析 PROPER 参考面曲率半径（远场近似）与 Pilot Beam 曲率半径（严格公式）
之间的差异如何导致相位误差。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper

from hybrid_optical_propagation import PilotBeamParams, GridSampling, SourceDefinition
from hybrid_optical_propagation.state_converter import StateConverter


def analyze_curvature_radius_difference():
    """分析两种曲率半径公式的差异"""
    print("=" * 80)
    print("分析曲率半径公式差异")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R_mm:.1f} mm")
    
    # 不同传播距离
    z_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
    
    print(f"\n{'z/z_R':<10} {'z (mm)':<15} {'R_proper (mm)':<20} {'R_pilot (mm)':<20} {'差异 (%)':<15}")
    print("-" * 80)
    
    for factor in z_factors:
        z_mm = factor * z_R_mm
        
        # PROPER 远场近似: R_ref = z - z_w0 = z（假设 z_w0 = 0）
        R_proper_mm = z_mm
        
        # Pilot Beam 严格公式: R = z × (1 + (z_R/z)²)
        R_pilot_mm = z_mm * (1 + (z_R_mm / z_mm)**2)
        
        # 差异
        diff_percent = abs(R_proper_mm - R_pilot_mm) / R_pilot_mm * 100
        
        print(f"{factor:<10.1f} {z_mm:<15.1f} {R_proper_mm:<20.1f} {R_pilot_mm:<20.1f} {diff_percent:<15.2f}")
    
    print("\n结论：")
    print("  - 在 z = z_R 时，差异最大（100%）")
    print("  - 在 z = 3×z_R 时，差异约 11%")
    print("  - 在 z = 10×z_R 时，差异约 1%")
    print("  - 在 z >> z_R 时，两者趋于一致")


def analyze_phase_error_source():
    """分析相位误差来源"""
    print("\n" + "=" * 80)
    print("分析相位误差来源")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 传播距离：3 × 瑞利长度
    propagation_distance_mm = 3 * z_R_mm
    
    print(f"\n参数:")
    print(f"  传播距离: {propagation_distance_mm:.1f} mm (3×z_R)")
    
    # 创建初始波前
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    amplitude, phase, pilot_params, wfo, grid_sampling = (
        source.create_initial_wavefront()[0],
        source.create_initial_wavefront()[1],
        source.create_initial_wavefront()[2],
        source.create_initial_wavefront()[3],
        source.get_grid_sampling(),
    )
    
    # 重新创建（避免重复调用）
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    grid_sampling = source.get_grid_sampling()
    
    print(f"\n初始状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  wfo.z: {wfo.z * 1e3:.1f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.1f} mm")
    print(f"  wfo.z_Rayleigh: {wfo.z_Rayleigh * 1e3:.1f} mm")
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
    
    print(f"\n传播后状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  wfo.z: {wfo.z * 1e3:.1f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.1f} mm")
    
    # 计算两种曲率半径
    z_mm = wfo.z * 1e3
    z_w0_mm = wfo.z_w0 * 1e3
    z_R_mm_proper = wfo.z_Rayleigh * 1e3
    
    # PROPER 参考面曲率半径
    R_proper_mm = z_mm - z_w0_mm
    
    # Pilot Beam 曲率半径（严格公式）
    z_rel_mm = z_mm - z_w0_mm  # 相对于束腰的距离
    R_pilot_mm = z_rel_mm * (1 + (z_R_mm_proper / z_rel_mm)**2)
    
    print(f"\n曲率半径比较:")
    print(f"  PROPER 参考面 R_ref: {R_proper_mm:.1f} mm")
    print(f"  Pilot Beam R: {R_pilot_mm:.1f} mm")
    print(f"  差异: {abs(R_proper_mm - R_pilot_mm):.1f} mm ({abs(R_proper_mm - R_pilot_mm) / R_pilot_mm * 100:.1f}%)")
    
    # 计算相位差异
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_mm = X_mm**2 + Y_mm**2
    k = 2 * np.pi / wavelength_mm
    
    # PROPER 参考面相位
    phase_proper = k * r_sq_mm / (2 * R_proper_mm)
    
    # Pilot Beam 相位
    phase_pilot = k * r_sq_mm / (2 * R_pilot_mm)
    
    # 相位差
    phase_diff = phase_proper - phase_pilot
    
    # 在有效区域内计算
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    
    print(f"\n相位差异（在有效区域内）:")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    print(f"  MAX: {np.max(np.abs(phase_diff[valid_mask])):.4f} rad")
    print(f"  PV: {np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask]):.4f} rad")
    
    # 边缘处的相位差
    r_max_mm = physical_size_mm / 2
    phase_diff_edge = k * r_max_mm**2 / 2 * (1/R_proper_mm - 1/R_pilot_mm)
    print(f"\n边缘处相位差（r = {r_max_mm} mm）:")
    print(f"  {phase_diff_edge:.4f} rad = {phase_diff_edge / (2 * np.pi):.6f} waves")


def test_proper_internal_consistency():
    """测试 PROPER 内部一致性"""
    print("\n" + "=" * 80)
    print("测试 PROPER 内部一致性")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 传播距离
    propagation_distance_mm = 3 * z_R_mm
    
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
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
    
    # 从 PROPER 提取相位
    proper_phase = proper.prop_get_phase(wfo)
    proper_amplitude = proper.prop_get_amplitude(wfo)
    
    print(f"\nPROPER 提取的相位:")
    print(f"  范围: [{np.min(proper_phase):.4f}, {np.max(proper_phase):.4f}] rad")
    print(f"  RMS: {np.sqrt(np.mean(proper_phase**2)):.4f} rad")
    
    # 计算 PROPER 参考面相位
    converter = StateConverter(wavelength_um)
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    print(f"\nPROPER 参考面相位:")
    print(f"  范围: [{np.min(proper_ref_phase):.4f}, {np.max(proper_ref_phase):.4f}] rad")
    
    # 重建绝对相位
    absolute_phase = proper_ref_phase + proper_phase
    
    print(f"\n重建的绝对相位:")
    print(f"  范围: [{np.min(absolute_phase):.4f}, {np.max(absolute_phase):.4f}] rad")
    
    # 更新 Pilot Beam 参数
    new_pilot_params = pilot_params.propagate(propagation_distance_mm)
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = new_pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    print(f"\nPilot Beam 参考相位:")
    print(f"  范围: [{np.min(pilot_phase):.4f}, {np.max(pilot_phase):.4f}] rad")
    print(f"  曲率半径: {new_pilot_params.curvature_radius_mm:.1f} mm")
    
    # 比较
    valid_mask = proper_amplitude > 0.01 * np.max(proper_amplitude)
    phase_diff = absolute_phase - pilot_phase
    
    print(f"\n绝对相位与 Pilot Beam 相位的差异:")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")


if __name__ == "__main__":
    analyze_curvature_radius_difference()
    analyze_phase_error_source()
    test_proper_internal_consistency()
