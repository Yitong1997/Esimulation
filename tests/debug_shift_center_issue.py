"""调试 prop_shift_center 导致的相位测量问题

问题：
- phase_grid_masked RMS = 1.1119 waves
- 但相位变化 RMS = 0.2842 waves

原因：prop_shift_center 将相位移到了 FFT 坐标系，
但 prop_get_phase 返回的相位是在原始坐标系中的。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def debug_shift_center():
    """调试 prop_shift_center 问题"""
    print("=" * 70)
    print("调试 prop_shift_center 问题")
    print("=" * 70)
    
    wavelength_m = 0.633e-6
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 64  # 使用小网格便于观察
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
    
    # 创建一个简单的像差：只在一个象限有值
    # 这样可以清楚地看到移位效果
    aberration_waves = np.zeros((n, n))
    aberration_waves[n//2:, n//2:] = 1.0  # 右上象限
    
    print(f"\n原始像差:")
    print(f"  非零元素位置: 右上象限")
    print(f"  非零元素数: {np.sum(aberration_waves != 0)}")
    
    # 转换为相位
    aberration_phase = -2 * np.pi * aberration_waves
    
    # =====================================================================
    # 测试 1：不使用 prop_shift_center
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 1：不使用 prop_shift_center")
    print("=" * 70)
    
    wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    phase_before1 = proper.prop_get_phase(wfo1)
    
    phase_field1 = np.exp(1j * aberration_phase)
    wfo1.wfarr = wfo1.wfarr * phase_field1
    
    phase_after1 = proper.prop_get_phase(wfo1)
    phase_change1 = phase_after1 - phase_before1
    
    # 检查相位变化的位置
    nonzero_mask1 = np.abs(phase_change1) > 0.1
    print(f"  相位变化非零位置:")
    print(f"    左上象限: {np.sum(nonzero_mask1[:n//2, :n//2])}")
    print(f"    右上象限: {np.sum(nonzero_mask1[:n//2, n//2:])}")
    print(f"    左下象限: {np.sum(nonzero_mask1[n//2:, :n//2])}")
    print(f"    右下象限: {np.sum(nonzero_mask1[n//2:, n//2:])}")
    
    # =====================================================================
    # 测试 2：使用 prop_shift_center
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 2：使用 prop_shift_center")
    print("=" * 70)
    
    wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    phase_before2 = proper.prop_get_phase(wfo2)
    
    phase_field2 = np.exp(1j * aberration_phase)
    phase_field2_fft = proper.prop_shift_center(phase_field2)
    wfo2.wfarr = wfo2.wfarr * phase_field2_fft
    
    phase_after2 = proper.prop_get_phase(wfo2)
    phase_change2 = phase_after2 - phase_before2
    
    # 检查相位变化的位置
    nonzero_mask2 = np.abs(phase_change2) > 0.1
    print(f"  相位变化非零位置:")
    print(f"    左上象限: {np.sum(nonzero_mask2[:n//2, :n//2])}")
    print(f"    右上象限: {np.sum(nonzero_mask2[:n//2, n//2:])}")
    print(f"    左下象限: {np.sum(nonzero_mask2[n//2:, :n//2])}")
    print(f"    右下象限: {np.sum(nonzero_mask2[n//2:, n//2:])}")
    
    # =====================================================================
    # 测试 3：检查 PROPER 初始波前的结构
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 3：检查 PROPER 初始波前的结构")
    print("=" * 70)
    
    wfo3 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 检查 wfarr 的结构
    print(f"  wfarr[0, 0] (左上角): {wfo3.wfarr[0, 0]}")
    print(f"  wfarr[n//2, n//2] (中心): {wfo3.wfarr[n//2, n//2]}")
    print(f"  wfarr[n-1, n-1] (右下角): {wfo3.wfarr[n-1, n-1]}")
    
    # 检查振幅分布
    amp3 = proper.prop_get_amplitude(wfo3)
    print(f"\n  振幅分布:")
    print(f"    左上角: {amp3[0, 0]:.4f}")
    print(f"    中心: {amp3[n//2, n//2]:.4f}")
    print(f"    右下角: {amp3[n-1, n-1]:.4f}")
    
    # =====================================================================
    # 测试 4：正确的相位应用方式
    # =====================================================================
    print("\n" + "=" * 70)
    print("测试 4：正确的相位应用方式")
    print("=" * 70)
    
    # PROPER 的 wfarr 是在 FFT 坐标系中的
    # prop_get_phase 返回的是移位后的相位（以中心为原点）
    # 所以我们需要：
    # 1. 创建以中心为原点的相位场
    # 2. 使用 prop_shift_center 移到 FFT 坐标系
    # 3. 应用到 wfarr
    # 4. prop_get_phase 会自动移回以中心为原点
    
    wfo4 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 创建一个简单的离焦像差（以中心为原点）
    r_norm = R_mm / 5.0  # 归一化到 5mm
    defocus_waves = r_norm**2
    defocus_waves = np.where(R_mm <= 5.0, defocus_waves, 0.0)
    defocus_phase = -2 * np.pi * defocus_waves
    
    # 应用相位
    phase_field4 = np.exp(1j * defocus_phase)
    phase_field4_fft = proper.prop_shift_center(phase_field4)
    wfo4.wfarr = wfo4.wfarr * phase_field4_fft
    
    # 测量
    phase_after4 = proper.prop_get_phase(wfo4)
    
    # 在光束范围内测量
    mask4 = R_mm <= 5.0
    valid_phase4 = phase_after4[mask4]
    valid_expected4 = defocus_phase[mask4]
    
    measured_rms4 = np.std(valid_phase4 - np.mean(valid_phase4)) / (2 * np.pi)
    expected_rms4 = np.std(valid_expected4 - np.mean(valid_expected4)) / (2 * np.pi)
    
    print(f"  测量 WFE RMS: {measured_rms4:.4f} waves")
    print(f"  预期 WFE RMS: {expected_rms4:.4f} waves")
    print(f"  比值: {measured_rms4 / expected_rms4:.4f}")
    
    # 检查相关性
    valid_phase4_centered = valid_phase4 - np.mean(valid_phase4)
    valid_expected4_centered = valid_expected4 - np.mean(valid_expected4)
    correlation4 = np.corrcoef(valid_phase4_centered, valid_expected4_centered)[0, 1]
    print(f"  相关性: {correlation4:.4f}")


if __name__ == "__main__":
    debug_shift_center()
