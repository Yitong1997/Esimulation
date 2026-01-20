"""调试相位测量方式

问题：应用像差相位后，测量的 WFE 远小于预期
可能原因：
1. PROPER 的相位是相对于参考球面的
2. 相位测量方式不正确
3. 相位应用方式不正确
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def test_phase_application():
    """测试相位应用和测量"""
    print("=" * 70)
    print("测试相位应用和测量")
    print("=" * 70)
    
    wavelength_m = 0.633e-6
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 512
    beam_ratio = 0.25
    
    # 初始化波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    print(f"\nPROPER 参数:")
    print(f"  网格大小: {n}x{n}")
    print(f"  采样间隔: {sampling_mm:.4f} mm/pixel")
    
    # 获取初始相位
    phase_initial = proper.prop_get_phase(wfo)
    amp_initial = proper.prop_get_amplitude(wfo)
    
    print(f"\n初始波前:")
    print(f"  相位 RMS: {np.std(phase_initial):.4f} rad")
    print(f"  相位 PV: {np.max(phase_initial) - np.min(phase_initial):.4f} rad")
    
    # 创建一个已知的像差相位（例如 1 波长的球差）
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    R_mm = np.sqrt(X_mm**2 + Y_mm**2)
    
    # 创建一个简单的像差：r^4 形式的球差
    # 归一化到光束半径
    beam_radius_mm = 5.0  # mm
    r_norm = R_mm / beam_radius_mm
    
    # 球差系数（波长数）
    spherical_coeff = 1.0  # 1 波长的球差
    aberration_waves = spherical_coeff * r_norm**4
    
    # 只在光束范围内应用
    mask = R_mm <= beam_radius_mm
    aberration_waves = np.where(mask, aberration_waves, 0.0)
    
    # 计算预期的 RMS
    valid_aberration = aberration_waves[mask]
    expected_rms = np.std(valid_aberration - np.mean(valid_aberration))
    
    print(f"\n应用的像差:")
    print(f"  像差 RMS: {expected_rms:.4f} waves")
    print(f"  像差 PV: {np.max(valid_aberration) - np.min(valid_aberration):.4f} waves")
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
    # 应用相位
    phase_field = np.exp(1j * aberration_phase)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo.wfarr = wfo.wfarr * phase_field_fft
    
    # 测量应用后的相位
    phase_after = proper.prop_get_phase(wfo)
    amp_after = proper.prop_get_amplitude(wfo)
    
    # 使用相同的掩模测量
    valid_phase = phase_after[mask]
    measured_rms = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
    
    print(f"\n测量的 WFE:")
    print(f"  WFE RMS: {measured_rms:.4f} waves")
    print(f"  预期 RMS: {expected_rms:.4f} waves")
    print(f"  比值: {measured_rms / expected_rms:.4f}")
    
    # 直接比较相位变化
    phase_change = phase_after - phase_initial
    valid_change = phase_change[mask]
    change_rms = np.std(valid_change - np.mean(valid_change)) / (2 * np.pi)
    
    print(f"\n相位变化:")
    print(f"  变化 RMS: {change_rms:.4f} waves")
    
    # 检查相位是否正确应用
    # 应用的相位应该等于 aberration_phase
    expected_phase_change = aberration_phase[mask]
    actual_phase_change = phase_change[mask]
    
    # 去除平均值后比较
    expected_centered = expected_phase_change - np.mean(expected_phase_change)
    actual_centered = actual_phase_change - np.mean(actual_phase_change)
    
    correlation = np.corrcoef(expected_centered, actual_centered)[0, 1]
    print(f"  相位相关性: {correlation:.4f}")
    
    # 检查差异
    diff = actual_centered - expected_centered
    diff_rms = np.std(diff) / (2 * np.pi)
    print(f"  差异 RMS: {diff_rms:.4f} waves")


if __name__ == "__main__":
    test_phase_application()
