"""
调试写入/读取 PROPER 的一致性

验证：写入 PROPER 后再读取，相位是否一致
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper

from hybrid_optical_propagation import PilotBeamParams, GridSampling, SourceDefinition
from hybrid_optical_propagation.state_converter import StateConverter


def test_write_read_consistency():
    """测试写入/读取一致性"""
    print("=" * 80)
    print("测试写入/读取 PROPER 的一致性")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 创建 Pilot Beam 参数（在远场位置）
    z_mm = 3 * z_R_mm  # 3 倍瑞利长度
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,  # 束腰在当前位置之前
    )
    
    print(f"\nPilot Beam 参数:")
    print(f"  曲率半径: {pilot_params.curvature_radius_mm:.1f} mm")
    print(f"  光斑大小: {pilot_params.spot_size_mm:.2f} mm")
    print(f"  束腰位置: {pilot_params.waist_position_mm:.1f} mm")
    
    # 创建网格采样
    grid_sampling = GridSampling.create(
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    # 创建测试振幅和相位
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_mm = X_mm**2 + Y_mm**2
    
    # 振幅：高斯分布
    w = pilot_params.spot_size_mm
    amplitude = np.exp(-r_sq_mm / w**2)
    
    # 相位：使用 Pilot Beam 严格公式
    k = 2 * np.pi / wavelength_mm
    R = pilot_params.curvature_radius_mm
    phase = k * r_sq_mm / (2 * R)
    
    print(f"\n输入相位:")
    print(f"  范围: [{np.min(phase):.4f}, {np.max(phase):.4f}] rad")
    print(f"  RMS: {np.sqrt(np.mean(phase**2)):.4f} rad")
    
    # 写入 PROPER
    converter = StateConverter(wavelength_um)
    wfo = converter.amplitude_phase_to_proper(
        amplitude, phase, grid_sampling, pilot_params
    )
    
    print(f"\nPROPER 状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  wfo.z: {wfo.z * 1e3:.1f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.1f} mm")
    
    # 计算 PROPER 参考面曲率半径
    R_proper_mm = (wfo.z - wfo.z_w0) * 1e3
    print(f"  PROPER 参考面曲率半径: {R_proper_mm:.1f} mm")
    print(f"  Pilot Beam 曲率半径: {R:.1f} mm")
    print(f"  差异: {abs(R_proper_mm - R):.1f} mm ({abs(R_proper_mm - R) / R * 100:.1f}%)")
    
    # 读取 PROPER
    amplitude_out, phase_out = converter.proper_to_amplitude_phase(
        wfo, grid_sampling, pilot_params
    )
    
    print(f"\n输出相位:")
    print(f"  范围: [{np.min(phase_out):.4f}, {np.max(phase_out):.4f}] rad")
    print(f"  RMS: {np.sqrt(np.mean(phase_out**2)):.4f} rad")
    
    # 比较
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    phase_diff = phase_out - phase
    
    print(f"\n相位差异（输出 - 输入）:")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)):.6f} rad")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi):.8f} waves")
    print(f"  MAX: {np.max(np.abs(phase_diff[valid_mask])):.6f} rad")
    
    # 振幅比较
    amp_diff = amplitude_out - amplitude
    print(f"\n振幅差异:")
    print(f"  RMS: {np.sqrt(np.mean(amp_diff[valid_mask]**2)):.8f}")
    print(f"  MAX: {np.max(np.abs(amp_diff[valid_mask])):.8f}")


def test_propagation_then_read():
    """测试传播后读取"""
    print("\n" + "=" * 80)
    print("测试传播后读取")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 创建初始波前（在束腰处）
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
    print(f"  Pilot Beam 曲率半径: {pilot_params.curvature_radius_mm}")
    
    # 传播到远场
    propagation_distance_mm = 3 * z_R_mm
    proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
    
    # 更新 Pilot Beam
    new_pilot_params = pilot_params.propagate(propagation_distance_mm)
    
    print(f"\n传播后状态:")
    print(f"  参考面: {wfo.reference_surface}")
    print(f"  wfo.z: {wfo.z * 1e3:.1f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.1f} mm")
    print(f"  PROPER 参考面曲率半径: {(wfo.z - wfo.z_w0) * 1e3:.1f} mm")
    print(f"  Pilot Beam 曲率半径: {new_pilot_params.curvature_radius_mm:.1f} mm")
    
    # 从 PROPER 提取
    converter = StateConverter(wavelength_um)
    amplitude_out, phase_out = converter.proper_to_amplitude_phase(
        wfo, grid_sampling, new_pilot_params
    )
    
    # 计算理想 Pilot Beam 相位
    pilot_phase = new_pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 比较
    valid_mask = amplitude_out > 0.01 * np.max(amplitude_out)
    phase_diff = phase_out - pilot_phase
    
    print(f"\n提取相位与 Pilot Beam 相位的差异:")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)):.4f} rad")
    print(f"  RMS: {np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")
    
    # 分析误差来源
    print("\n误差来源分析:")
    
    # PROPER 参考面相位
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    # 两种参考相位的差异
    ref_diff = proper_ref_phase - pilot_phase
    print(f"  PROPER 参考面与 Pilot Beam 相位差异:")
    print(f"    RMS: {np.sqrt(np.mean(ref_diff[valid_mask]**2)):.4f} rad")
    print(f"    RMS: {np.sqrt(np.mean(ref_diff[valid_mask]**2)) / (2 * np.pi):.6f} waves")


if __name__ == "__main__":
    test_write_read_consistency()
    test_propagation_then_read()
