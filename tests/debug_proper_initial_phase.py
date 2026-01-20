"""调试 PROPER 波前的初始相位"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def debug_proper_initial_phase():
    """调试 PROPER 波前的初始相位"""
    print("=" * 70)
    print("调试 PROPER 波前的初始相位")
    print("=" * 70)
    
    wavelength_um = 0.633
    w0 = 5.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3
    grid_size = 512
    beam_ratio = 0.25
    
    # 初始化 PROPER 波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_um * 1e-6, grid_size, beam_ratio)
    
    # 获取初始相位
    phase = proper.prop_get_phase(wfo)
    phase_centered = proper.prop_shift_center(phase)
    
    # 获取振幅
    amp = proper.prop_get_amplitude(wfo)
    amp_centered = proper.prop_shift_center(amp)
    
    # 创建掩模
    mask = amp_centered > 0.01 * np.max(amp_centered)
    
    print(f"\n初始相位（在有效区域内）:")
    print(f"  min = {np.min(phase_centered[mask]):.4f} rad")
    print(f"  max = {np.max(phase_centered[mask]):.4f} rad")
    print(f"  RMS = {np.std(phase_centered[mask]):.4f} rad")
    print(f"  RMS (waves) = {np.std(phase_centered[mask]) / (2 * np.pi):.4f}")
    
    # 检查相位是否为平面
    n = grid_size
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    coords = np.linspace(-sampling_mm * n / 2, sampling_mm * n / 2, n)
    X, Y = np.meshgrid(coords, coords)
    
    # 拟合倾斜
    valid_phase = phase_centered[mask]
    valid_x = X[mask]
    valid_y = Y[mask]
    
    A = np.column_stack([np.ones_like(valid_x), valid_x, valid_y])
    coeffs, _, _, _ = np.linalg.lstsq(A, valid_phase, rcond=None)
    
    print(f"\n相位倾斜拟合:")
    print(f"  a0 (piston) = {coeffs[0]:.4f} rad")
    print(f"  a1 (tilt_x) = {coeffs[1]:.6f} rad/mm")
    print(f"  a2 (tilt_y) = {coeffs[2]:.6f} rad/mm")
    
    # 去除倾斜后的相位
    tilt_phase = coeffs[0] + coeffs[1] * X + coeffs[2] * Y
    phase_no_tilt = phase_centered - tilt_phase
    
    print(f"\n去除倾斜后的相位（在有效区域内）:")
    print(f"  RMS = {np.std(phase_no_tilt[mask]):.4f} rad")
    print(f"  RMS (waves) = {np.std(phase_no_tilt[mask]) / (2 * np.pi):.4f}")


if __name__ == "__main__":
    debug_proper_initial_phase()
