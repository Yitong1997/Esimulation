"""调试相位符号问题

问题：相关性是 -0.5，说明相位符号可能有问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def debug_phase_sign():
    """调试相位符号"""
    print("=" * 70)
    print("调试相位符号")
    print("=" * 70)
    
    wavelength_m = 0.633e-6
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 128
    beam_ratio = 0.25
    
    # 初始化波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    print(f"\nPROPER 参数:")
    print(f"  网格大小: {n}x{n}")
    print(f"  采样间隔: {sampling_mm:.4f} mm/pixel")
    
    # 创建坐标网格
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    R_mm = np.sqrt(X_mm**2 + Y_mm**2)
    
    # 创建一个简单的离焦像差
    beam_radius_mm = 5.0
    r_norm = R_mm / beam_radius_mm
    defocus_waves = r_norm**2
    mask = R_mm <= beam_radius_mm
    defocus_waves = np.where(mask, defocus_waves, 0.0)
    
    # =====================================================================
    # 测试不同的相位符号
    # =====================================================================
    for sign in [1, -1]:
        print(f"\n" + "=" * 70)
        print(f"测试相位符号: {'+' if sign > 0 else '-'}2π * aberration")
        print("=" * 70)
        
        wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        # 应用相位
        aberration_phase = sign * 2 * np.pi * defocus_waves
        phase_field = np.exp(1j * aberration_phase)
        phase_field_fft = proper.prop_shift_center(phase_field)
        wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
        
        # 测量
        phase_after = proper.prop_get_phase(wfo_test)
        
        # 在光束范围内测量
        valid_phase = phase_after[mask]
        valid_expected = aberration_phase[mask]
        
        # 去除平均值
        valid_phase_centered = valid_phase - np.mean(valid_phase)
        valid_expected_centered = valid_expected - np.mean(valid_expected)
        
        # 计算相关性
        correlation = np.corrcoef(valid_phase_centered, valid_expected_centered)[0, 1]
        
        # 计算 RMS
        measured_rms = np.std(valid_phase_centered) / (2 * np.pi)
        expected_rms = np.std(valid_expected_centered) / (2 * np.pi)
        
        print(f"  测量 WFE RMS: {measured_rms:.4f} waves")
        print(f"  预期 WFE RMS: {expected_rms:.4f} waves")
        print(f"  相关性: {correlation:.4f}")
        
        # 检查中心和边缘的相位值
        center_idx = n // 2
        edge_idx = int(center_idx + beam_radius_mm / sampling_mm)
        
        print(f"\n  相位值检查:")
        print(f"    中心 (r=0): 应用={aberration_phase[center_idx, center_idx]:.4f}, 测量={phase_after[center_idx, center_idx]:.4f}")
        print(f"    边缘 (r=5mm): 应用={aberration_phase[center_idx, edge_idx]:.4f}, 测量={phase_after[center_idx, edge_idx]:.4f}")
    
    # =====================================================================
    # 测试：直接比较应用的相位和测量的相位
    # =====================================================================
    print("\n" + "=" * 70)
    print("直接比较应用的相位和测量的相位")
    print("=" * 70)
    
    wfo_direct = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取初始相位
    phase_initial = proper.prop_get_phase(wfo_direct)
    
    # 应用一个已知的相位（正号）
    known_phase = 2 * np.pi * defocus_waves  # 正号
    phase_field = np.exp(1j * known_phase)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo_direct.wfarr = wfo_direct.wfarr * phase_field_fft
    
    # 测量相位变化
    phase_final = proper.prop_get_phase(wfo_direct)
    phase_change = phase_final - phase_initial
    
    # 比较
    print(f"\n  应用的相位（中心）: {known_phase[center_idx, center_idx]:.4f} rad")
    print(f"  测量的相位变化（中心）: {phase_change[center_idx, center_idx]:.4f} rad")
    print(f"  应用的相位（边缘）: {known_phase[center_idx, edge_idx]:.4f} rad")
    print(f"  测量的相位变化（边缘）: {phase_change[center_idx, edge_idx]:.4f} rad")
    
    # 检查相位变化是否等于应用的相位
    valid_known = known_phase[mask]
    valid_change = phase_change[mask]
    
    diff = valid_change - valid_known
    diff_rms = np.std(diff) / (2 * np.pi)
    print(f"\n  相位差异 RMS: {diff_rms:.4f} waves")
    
    # 检查是否是符号相反
    diff_neg = valid_change + valid_known
    diff_neg_rms = np.std(diff_neg) / (2 * np.pi)
    print(f"  相位差异（假设符号相反）RMS: {diff_neg_rms:.4f} waves")


if __name__ == "__main__":
    debug_phase_sign()
