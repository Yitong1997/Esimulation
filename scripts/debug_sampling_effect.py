"""
调试采样对传播精度的影响
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper

from hybrid_optical_propagation import PilotBeamParams, GridSampling, SourceDefinition
from hybrid_optical_propagation.state_converter import StateConverter


def test_sampling_effect():
    """测试不同采样对传播精度的影响"""
    print("=" * 80)
    print("测试采样对传播精度的影响")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    propagation_distance_mm = 3 * z_R_mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R_mm:.1f} mm")
    print(f"  传播距离: {propagation_distance_mm:.1f} mm (3×z_R)")
    
    # 测试不同的网格大小
    grid_sizes = [64, 128, 256, 512]
    
    print(f"\n{'网格大小':<12} {'采样间隔(mm)':<15} {'相位RMS误差(waves)':<20}")
    print("-" * 50)
    
    for grid_size in grid_sizes:
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
        
        # 更新 Pilot Beam
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 从 PROPER 提取
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
        
        sampling_mm = physical_size_mm / grid_size
        print(f"{grid_size:<12} {sampling_mm:<15.4f} {rms_error_waves:<20.6f}")


def test_physical_size_effect():
    """测试物理尺寸对传播精度的影响"""
    print("\n" + "=" * 80)
    print("测试物理尺寸对传播精度的影响")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 256
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    propagation_distance_mm = 3 * z_R_mm
    
    # 传播后的光斑大小
    w_after = w0_mm * np.sqrt(1 + (propagation_distance_mm / z_R_mm)**2)
    print(f"\n传播后光斑大小: {w_after:.2f} mm")
    
    # 测试不同的物理尺寸
    physical_sizes = [10.0, 20.0, 40.0, 80.0]
    
    print(f"\n{'物理尺寸(mm)':<15} {'尺寸/光斑':<12} {'相位RMS误差(waves)':<20}")
    print("-" * 50)
    
    for physical_size_mm in physical_sizes:
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
        
        # 更新 Pilot Beam
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 从 PROPER 提取
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
        
        size_ratio = physical_size_mm / w_after
        print(f"{physical_size_mm:<15.1f} {size_ratio:<12.2f} {rms_error_waves:<20.6f}")


def test_propagation_distance_effect():
    """测试传播距离对精度的影响"""
    print("\n" + "=" * 80)
    print("测试传播距离对精度的影响")
    print("=" * 80)
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 256
    physical_size_mm = 40.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    # 测试不同的传播距离
    distance_factors = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print(f"\n{'z/z_R':<10} {'传播距离(mm)':<15} {'相位RMS误差(waves)':<20}")
    print("-" * 50)
    
    for factor in distance_factors:
        propagation_distance_mm = factor * z_R_mm
        
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
        
        # 更新 Pilot Beam
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 从 PROPER 提取
        converter = StateConverter(wavelength_um)
        amplitude_out, phase_out = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(grid_size, physical_size_mm)
        
        # 计算误差
        valid_mask = amplitude_out > 0.01 * np.max(amplitude_out)
        if np.sum(valid_mask) > 0:
            phase_diff = np.angle(np.exp(1j * (phase_out - pilot_phase)))
            rms_error_waves = np.sqrt(np.mean(phase_diff[valid_mask]**2)) / (2 * np.pi)
        else:
            rms_error_waves = np.nan
        
        print(f"{factor:<10.1f} {propagation_distance_mm:<15.1f} {rms_error_waves:<20.6f}")


if __name__ == "__main__":
    test_sampling_effect()
    test_physical_size_effect()
    test_propagation_distance_effect()
