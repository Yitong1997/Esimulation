"""
定位远场传播误差的来源

分离测试：
1. PROPER 纯自由空间传播（不经过 StateConverter）
2. StateConverter 写入/读取（不经过传播）
3. 完整流程（StateConverter + PROPER 传播 + StateConverter）

目标：确定误差是来自 PROPER 传播还是 StateConverter 的读写操作
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper

from hybrid_optical_propagation import PilotBeamParams, GridSampling, SourceDefinition
from hybrid_optical_propagation.state_converter import StateConverter


def test_pure_proper_propagation():
    """测试 1: PROPER 纯自由空间传播（不经过 StateConverter）
    
    直接使用 PROPER 创建高斯光束，传播，然后与理论 Pilot Beam 比较
    """
    print("=" * 80)
    print("测试 1: PROPER 纯自由空间传播")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    wavelength_m = wavelength_um * 1e-6
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 测试不同传播距离
    z_factors = [1.0, 3.0, 10.0, 100.0]
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R_mm:.1f} mm")
    
    print(f"\n{'z/z_R':<10} {'参考面':<10} {'相位RMS误差(waves)':<20}")
    print("-" * 45)
    
    for z_factor in z_factors:
        propagation_distance_mm = z_factor * z_R_mm
        
        # 使用 PROPER 创建初始高斯光束
        beam_diameter_m = 2 * w0_mm * 1e-3
        grid_width_m = physical_size_mm * 1e-3
        beam_diam_fraction = beam_diameter_m / grid_width_m
        
        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            grid_size,
            beam_diam_fraction,
        )
        
        # 传播
        proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
        
        # 从 PROPER 提取振幅和相位
        amplitude = proper.prop_get_amplitude(wfo)
        proper_phase = proper.prop_get_phase(wfo)  # 残差相位
        
        # 计算 PROPER 参考面相位
        if wfo.reference_surface == "SPHERI":
            R_ref_m = wfo.z - wfo.z_w0
            sampling_m = proper.prop_get_sampling(wfo)
            n = grid_size
            coords_m = (np.arange(n) - n // 2) * sampling_m
            X_m, Y_m = np.meshgrid(coords_m, coords_m)
            r_sq_m = X_m**2 + Y_m**2
            k = 2 * np.pi / wavelength_m
            proper_ref_phase = k * r_sq_m / (2 * R_ref_m)
        else:
            proper_ref_phase = np.zeros((grid_size, grid_size))
        
        # 重建完整相位
        full_phase = proper_ref_phase + proper_phase
        
        # 计算理论 Pilot Beam 相位（严格公式）
        z_mm = propagation_distance_mm
        R_pilot_mm = z_mm * (1 + (z_R_mm / z_mm)**2)
        
        sampling_mm = physical_size_mm / grid_size
        coords_mm = (np.arange(grid_size) - grid_size // 2) * sampling_mm
        X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
        r_sq_mm = X_mm**2 + Y_mm**2
        
        k_mm = 2 * np.pi / wavelength_mm
        pilot_phase = k_mm * r_sq_mm / (2 * R_pilot_mm)
        
        # 计算误差
        valid_mask = amplitude > 0.01 * np.max(amplitude)
        phase_diff = np.angle(np.exp(1j * (full_phase - pilot_phase)))
        rms_error_waves = np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi)
        
        print(f"{z_factor:<10.1f} {wfo.reference_surface:<10} {rms_error_waves:<20.6f}")
    
    return rms_error_waves


def test_state_converter_write_read():
    """测试 2: StateConverter 写入/读取（不经过传播）
    
    创建高斯波前，写入 PROPER，立即读取，比较误差
    """
    print("\n" + "=" * 80)
    print("测试 2: StateConverter 写入/读取（无传播）")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 测试不同的初始位置（模拟远场情况）
    z_factors = [0.0, 1.0, 3.0, 10.0]
    
    print(f"\n{'z/z_R':<10} {'参考面':<10} {'相位RMS误差(waves)':<20}")
    print("-" * 45)
    
    converter = StateConverter(wavelength_um)
    
    for z_factor in z_factors:
        z_mm = z_factor * z_R_mm if z_factor > 0 else 0.001  # 避免除零
        
        # 创建 Pilot Beam 参数
        pilot_params = PilotBeamParams.from_gaussian_source(
            wavelength_um, w0_mm, -z_mm  # z0_mm 是束腰相对于当前位置
        )
        
        # 创建高斯振幅和相位
        sampling_mm = physical_size_mm / grid_size
        coords_mm = (np.arange(grid_size) - grid_size // 2) * sampling_mm
        X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
        r_sq_mm = X_mm**2 + Y_mm**2
        
        # 光斑大小
        w_mm = w0_mm * np.sqrt(1 + (z_mm / z_R_mm)**2) if z_R_mm > 0 else w0_mm
        amplitude = np.exp(-r_sq_mm / w_mm**2)
        
        # 相位（使用严格 Pilot Beam 公式）
        phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
        
        # 创建 GridSampling
        grid_sampling = GridSampling.create(grid_size, physical_size_mm)
        
        # 写入 PROPER
        wfo = converter.amplitude_phase_to_proper(
            amplitude, phase, grid_sampling, pilot_params
        )
        
        # 立即读取
        amplitude_out, phase_out = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, pilot_params
        )
        
        # 计算误差
        valid_mask = amplitude > 0.01 * np.max(amplitude)
        phase_diff = np.angle(np.exp(1j * (phase_out - phase)))
        rms_error_waves = np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi)
        
        print(f"{z_factor:<10.1f} {wfo.reference_surface:<10} {rms_error_waves:<20.6f}")


def test_full_pipeline():
    """测试 3: 完整流程（StateConverter + PROPER 传播 + StateConverter）
    
    这是远场传播测试中使用的完整流程
    """
    print("\n" + "=" * 80)
    print("测试 3: 完整流程（写入 → 传播 → 读取）")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    z_factors = [1.0, 3.0, 10.0, 100.0]
    
    print(f"\n{'z/z_R':<10} {'参考面':<10} {'相位RMS误差(waves)':<20}")
    print("-" * 45)
    
    for z_factor in z_factors:
        propagation_distance_mm = z_factor * z_R_mm
        
        # 使用 SourceDefinition 创建初始波前
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
        
        # 更新 Pilot Beam
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 使用 StateConverter 读取
        converter = StateConverter(wavelength_um)
        amplitude_out, phase_out = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(grid_size, physical_size_mm)
        
        # 计算误差
        valid_mask = amplitude_out > 0.01 * np.max(amplitude_out)
        phase_diff = np.angle(np.exp(1j * (phase_out - pilot_phase)))
        rms_error_waves = np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi)
        
        print(f"{z_factor:<10.1f} {wfo.reference_surface:<10} {rms_error_waves:<20.6f}")


def test_proper_vs_pilot_curvature():
    """测试 4: 比较 PROPER 参考面曲率和 Pilot Beam 曲率
    
    分析曲率半径差异对相位的影响
    """
    print("\n" + "=" * 80)
    print("测试 4: PROPER 参考面曲率 vs Pilot Beam 曲率")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    z_factors = [1.0, 2.0, 3.0, 5.0, 10.0, 100.0]
    
    print(f"\n{'z/z_R':<10} {'R_proper(mm)':<15} {'R_pilot(mm)':<15} {'差异(%)':<12} {'相位差(边缘,waves)':<20}")
    print("-" * 75)
    
    r_edge_mm = 5.0  # 边缘半径
    k = 2 * np.pi / wavelength_mm
    
    for z_factor in z_factors:
        z_mm = z_factor * z_R_mm
        
        # PROPER 远场近似曲率
        R_proper_mm = z_mm  # R = z - z_w0，z_w0 = 0
        
        # Pilot Beam 严格曲率
        R_pilot_mm = z_mm * (1 + (z_R_mm / z_mm)**2)
        
        # 曲率差异
        diff_percent = abs(R_proper_mm - R_pilot_mm) / R_pilot_mm * 100
        
        # 边缘相位差
        phase_proper = k * r_edge_mm**2 / (2 * R_proper_mm)
        phase_pilot = k * r_edge_mm**2 / (2 * R_pilot_mm)
        phase_diff_waves = abs(phase_proper - phase_pilot) / (2 * np.pi)
        
        print(f"{z_factor:<10.1f} {R_proper_mm:<15.1f} {R_pilot_mm:<15.1f} {diff_percent:<12.2f} {phase_diff_waves:<20.6f}")


def test_source_definition_accuracy():
    """测试 5: SourceDefinition 创建的初始波前精度
    
    检查 SourceDefinition 创建的波前是否与 Pilot Beam 一致
    """
    print("\n" + "=" * 80)
    print("测试 5: SourceDefinition 初始波前精度")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    # 创建初始波前
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 计算误差
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    phase_diff = np.angle(np.exp(1j * (phase - pilot_phase)))
    rms_error_waves = np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi)
    
    print(f"\n初始波前精度:")
    print(f"  参考面类型: {wfo.reference_surface}")
    print(f"  相位 RMS 误差: {rms_error_waves:.6f} waves")
    print(f"  Pilot Beam 曲率半径: {pilot_params.curvature_radius_mm}")


if __name__ == "__main__":
    test_pure_proper_propagation()
    test_state_converter_write_read()
    test_full_pipeline()
    test_proper_vs_pilot_curvature()
    test_source_definition_accuracy()
