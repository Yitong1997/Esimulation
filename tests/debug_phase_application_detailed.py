"""详细调试相位应用

问题：
1. 测量的 WFE 与预期值有差异
2. 相位相关性是负的（-0.3950）

可能原因：
1. prop_shift_center 的使用方式不正确
2. 相位符号问题
3. PROPER 内部坐标系问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def test_phase_application_detailed():
    """详细测试相位应用"""
    print("=" * 70)
    print("详细测试相位应用")
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
    
    # 创建坐标网格
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    R_mm = np.sqrt(X_mm**2 + Y_mm**2)
    
    # 创建一个简单的像差：r^2 形式的离焦
    beam_radius_mm = 5.0  # mm
    r_norm = R_mm / beam_radius_mm
    
    # 离焦系数（波长数）
    defocus_coeff = 1.0  # 1 波长的离焦
    aberration_waves = defocus_coeff * r_norm**2
    
    # 只在光束范围内应用
    mask = R_mm <= beam_radius_mm
    aberration_waves = np.where(mask, aberration_waves, 0.0)
    
    # 计算预期的 RMS
    valid_aberration = aberration_waves[mask]
    expected_rms = np.std(valid_aberration - np.mean(valid_aberration))
    
    print(f"\n应用的像差:")
    print(f"  像差 RMS: {expected_rms:.4f} waves")
    print(f"  像差 PV: {np.max(valid_aberration) - np.min(valid_aberration):.4f} waves")
    
    # =====================================================================
    # 测试 1：直接修改 wfarr（不使用 prop_shift_center）
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 1：直接修改 wfarr（不使用 prop_shift_center）")
    print("=" * 70)
    
    wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 转换为相位（注意符号）
    aberration_phase = -2 * np.pi * aberration_waves
    
    # 直接应用相位（不使用 prop_shift_center）
    phase_field = np.exp(1j * aberration_phase)
    wfo1.wfarr = wfo1.wfarr * phase_field
    
    # 测量
    phase_after1 = proper.prop_get_phase(wfo1)
    valid_phase1 = phase_after1[mask]
    measured_rms1 = np.std(valid_phase1 - np.mean(valid_phase1)) / (2 * np.pi)
    
    print(f"  测量 WFE RMS: {measured_rms1:.4f} waves")
    print(f"  预期 RMS: {expected_rms:.4f} waves")
    print(f"  比值: {measured_rms1 / expected_rms:.4f}")
    
    # =====================================================================
    # 测试 2：使用 prop_shift_center
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 2：使用 prop_shift_center")
    print("=" * 70)
    
    wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 使用 prop_shift_center
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo2.wfarr = wfo2.wfarr * phase_field_fft
    
    # 测量
    phase_after2 = proper.prop_get_phase(wfo2)
    valid_phase2 = phase_after2[mask]
    measured_rms2 = np.std(valid_phase2 - np.mean(valid_phase2)) / (2 * np.pi)
    
    print(f"  测量 WFE RMS: {measured_rms2:.4f} waves")
    print(f"  预期 RMS: {expected_rms:.4f} waves")
    print(f"  比值: {measured_rms2 / expected_rms:.4f}")
    
    # =====================================================================
    # 测试 3：检查 prop_shift_center 的效果
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 3：检查 prop_shift_center 的效果")
    print("=" * 70)
    
    # 创建一个简单的测试数组
    test_array = np.zeros((8, 8))
    test_array[3, 3] = 1.0  # 中心位置
    test_array[4, 4] = 2.0  # 偏移位置
    
    shifted = proper.prop_shift_center(test_array)
    
    print("  原始数组（中心区域）:")
    print(test_array[2:6, 2:6])
    print("\n  移位后数组（中心区域）:")
    print(shifted[2:6, 2:6])
    
    # =====================================================================
    # 测试 4：使用 prop_add_phase
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 4：使用 prop_add_phase")
    print("=" * 70)
    
    wfo4 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 使用 prop_add_phase（如果存在）
    try:
        # 注意：prop_add_phase 接受的是 OPD（米），不是相位
        opd_m = aberration_waves * wavelength_m
        proper.prop_add_phase(wfo4, opd_m)
        
        # 测量
        phase_after4 = proper.prop_get_phase(wfo4)
        valid_phase4 = phase_after4[mask]
        measured_rms4 = np.std(valid_phase4 - np.mean(valid_phase4)) / (2 * np.pi)
        
        print(f"  测量 WFE RMS: {measured_rms4:.4f} waves")
        print(f"  预期 RMS: {expected_rms:.4f} waves")
        print(f"  比值: {measured_rms4 / expected_rms:.4f}")
    except Exception as e:
        print(f"  prop_add_phase 不可用: {e}")
    
    # =====================================================================
    # 测试 5：检查 PROPER 的初始波前
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 5：检查 PROPER 的初始波前")
    print("=" * 70)
    
    wfo5 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取初始波前
    amp_init = proper.prop_get_amplitude(wfo5)
    phase_init = proper.prop_get_phase(wfo5)
    
    print(f"  初始振幅最大值: {np.max(amp_init):.4f}")
    print(f"  初始振幅最小值: {np.min(amp_init):.4f}")
    print(f"  初始相位 RMS: {np.std(phase_init):.4f} rad")
    
    # 检查 wfarr 的结构
    print(f"\n  wfarr 形状: {wfo5.wfarr.shape}")
    print(f"  wfarr 数据类型: {wfo5.wfarr.dtype}")
    print(f"  wfarr 中心值: {wfo5.wfarr[n//2, n//2]}")
    print(f"  wfarr 左上角值: {wfo5.wfarr[0, 0]}")
    
    # =====================================================================
    # 测试 6：正确的相位应用方式
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 6：正确的相位应用方式")
    print("=" * 70)
    
    wfo6 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取初始波前的振幅和相位
    amp_before = proper.prop_get_amplitude(wfo6)
    phase_before = proper.prop_get_phase(wfo6)
    
    # 创建相位场（以中心为原点）
    aberration_phase_centered = -2 * np.pi * aberration_waves
    
    # 将相位场移到 FFT 坐标系
    phase_field_centered = np.exp(1j * aberration_phase_centered)
    phase_field_fft = proper.prop_shift_center(phase_field_centered)
    
    # 应用相位
    wfo6.wfarr = wfo6.wfarr * phase_field_fft
    
    # 获取应用后的振幅和相位
    amp_after = proper.prop_get_amplitude(wfo6)
    phase_after = proper.prop_get_phase(wfo6)
    
    # 计算相位变化
    phase_change = phase_after - phase_before
    
    # 在掩模区域内比较
    valid_change = phase_change[mask]
    valid_expected = aberration_phase_centered[mask]
    
    # 去除平均值
    valid_change_centered = valid_change - np.mean(valid_change)
    valid_expected_centered = valid_expected - np.mean(valid_expected)
    
    # 计算相关性
    correlation = np.corrcoef(valid_change_centered, valid_expected_centered)[0, 1]
    
    print(f"  相位变化 RMS: {np.std(valid_change_centered) / (2 * np.pi):.4f} waves")
    print(f"  预期相位 RMS: {np.std(valid_expected_centered) / (2 * np.pi):.4f} waves")
    print(f"  相关性: {correlation:.4f}")
    
    # 检查差异
    diff = valid_change_centered - valid_expected_centered
    diff_rms = np.std(diff) / (2 * np.pi)
    print(f"  差异 RMS: {diff_rms:.4f} waves")
    
    # =====================================================================
    # 测试 7：检查 prop_shift_center 是否是 fftshift
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 7：检查 prop_shift_center 是否是 fftshift")
    print("=" * 70)
    
    test_array = np.arange(16).reshape(4, 4)
    shifted_proper = proper.prop_shift_center(test_array)
    shifted_numpy = np.fft.fftshift(test_array)
    
    print("  原始数组:")
    print(test_array)
    print("\n  prop_shift_center 结果:")
    print(shifted_proper)
    print("\n  np.fft.fftshift 结果:")
    print(shifted_numpy)
    print(f"\n  两者相同: {np.allclose(shifted_proper, shifted_numpy)}")


if __name__ == "__main__":
    test_phase_application_detailed()
