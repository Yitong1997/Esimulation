"""
正确验证 PROPER 传播算法

基于对 PROPER 源码的分析：
1. wfarr 存储完整复振幅
2. 高斯光束参数用于选择传播算法和优化计算
3. 远场传播时采样会改变：dx_new = λ*|dz| / (ngrid * dx_old)
4. prop_qphase 添加相位：exp(i*π/(λ*c) * r²)

验证策略：
1. 近场传播（PTP）：采样不变，直接比较
2. 远场传播：需要考虑采样变化和 FFT 缩放
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def gaussian_beam_params(w0, wavelength, z):
    """计算高斯光束参数（严格公式）"""
    z_R = np.pi * w0**2 / wavelength
    
    if abs(z) < 1e-12:
        return w0, np.inf, 0.0, z_R
    
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)
    gouy = np.arctan(z / z_R)
    
    return w, R, gouy, z_R


def create_gaussian_field(grid_size, sampling_m, w0, wavelength, z, include_gouy=True):
    """创建高斯光束复振幅场"""
    w, R, gouy, z_R = gaussian_beam_params(w0, wavelength, z)
    
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_m
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength
    
    # 振幅：(w0/w) * exp(-r²/w²)
    amplitude = (w0 / w) * np.exp(-r_sq / w**2)
    
    # 相位：球面波前 + Gouy 相位
    if np.isinf(R):
        spherical_phase = np.zeros_like(r_sq)
    else:
        spherical_phase = k * r_sq / (2 * R)
    
    if include_gouy:
        phase = spherical_phase - gouy
    else:
        phase = spherical_phase
    
    return amplitude * np.exp(1j * phase), {'w': w, 'R': R, 'gouy': gouy, 'z_R': z_R}


def verify_near_field_ptp():
    """验证近场 PTP 传播"""
    
    print("=" * 70)
    print("验证近场 PTP 传播")
    print("=" * 70)
    
    # 参数
    wavelength = 632.8e-9  # m
    w0 = 1e-3  # 1 mm
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    
    # 传播距离：在瑞利距离内
    propagation_distance = 0.1 * z_R  # 0.1 z_R
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  w0: {w0 * 1e3:.3f} mm")
    print(f"  z_R: {z_R * 1e3:.3f} mm")
    print(f"  传播距离: {propagation_distance * 1e3:.3f} mm ({propagation_distance/z_R:.2f} z_R)")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_m = grid_width / grid_size
    
    print(f"  采样: {sampling_m * 1e6:.3f} μm")
    
    # 创建初始高斯光束（在束腰处）
    initial_field, _ = create_gaussian_field(grid_size, sampling_m, w0, wavelength, 0.0)
    
    # 设置 PROPER
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(initial_field.astype(np.complex128))
    
    print(f"\n初始状态:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  reference_surface: {wfo.reference_surface}")
    print(f"  beam_type_old: {wfo.beam_type_old}")
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance)
    
    print(f"\n传播后:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 读取结果
    read_amp = proper.prop_get_amplitude(wfo)
    read_phase = proper.prop_get_phase(wfo)
    
    # 理论值（使用相同采样）
    z_final = propagation_distance
    theory_field, theory_params = create_gaussian_field(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    theory_amp = np.abs(theory_field)
    theory_phase = np.angle(theory_field)
    
    print(f"\n理论参数:")
    print(f"  w(z): {theory_params['w'] * 1e3:.6f} mm")
    print(f"  R(z): {theory_params['R'] * 1e3:.3f} mm")
    print(f"  Gouy: {theory_params['gouy']:.6f} rad")
    
    # 比较
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    amp_error = np.sqrt(np.mean((read_amp[mask] - theory_amp[mask])**2))
    phase_diff = np.angle(np.exp(1j * (read_phase - theory_phase)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n误差:")
    print(f"  振幅 RMS: {amp_error:.6f}")
    print(f"  相位 RMS: {phase_error:.6f} rad = {phase_error/(2*np.pi):.6f} waves")
    
    passed = amp_error < 0.001 and phase_error < 0.01
    print(f"\n结果: {'PASS' if passed else 'FAIL'}")
    
    return passed


def verify_inside_to_outside():
    """验证 INSIDE_to_OUTSIDE 传播"""
    
    print("\n" + "=" * 70)
    print("验证 INSIDE_to_OUTSIDE 传播")
    print("=" * 70)
    
    # 参数
    wavelength = 632.8e-9
    w0 = 0.5e-3
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    
    # 传播距离：从束腰到远场
    propagation_distance = 5 * z_R
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  w0: {w0 * 1e3:.3f} mm")
    print(f"  z_R: {z_R * 1e3:.3f} mm")
    print(f"  传播距离: {propagation_distance * 1e3:.1f} mm ({propagation_distance/z_R:.1f} z_R)")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_initial = grid_width / grid_size
    
    print(f"  初始采样: {sampling_initial * 1e6:.3f} μm")
    
    # 创建初始高斯光束
    initial_field, _ = create_gaussian_field(grid_size, sampling_initial, w0, wavelength, 0.0)
    
    # 设置 PROPER
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(initial_field.astype(np.complex128))
    
    print(f"\n初始状态:")
    print(f"  reference_surface: {wfo.reference_surface}")
    print(f"  beam_type_old: {wfo.beam_type_old}")
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance)
    
    # 计算预期的新采样
    # INSIDE_to_OUTSIDE: 先 PTP 到 z_w0，再 WTS 到目标
    # WTS 的采样变化：dx_new = λ*|dz| / (ngrid * dx_old)
    # 这里 dz = propagation_distance - 0 = propagation_distance
    expected_sampling = wavelength * propagation_distance / (grid_size * sampling_initial)
    
    print(f"\n传播后:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  预期采样: {expected_sampling * 1e6:.3f} μm")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 读取结果
    read_amp = proper.prop_get_amplitude(wfo)
    read_phase = proper.prop_get_phase(wfo)
    
    # 理论值（使用新采样）
    z_final = propagation_distance
    theory_field, theory_params = create_gaussian_field(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    theory_amp = np.abs(theory_field)
    theory_phase = np.angle(theory_field)
    
    print(f"\n理论参数:")
    print(f"  w(z): {theory_params['w'] * 1e3:.3f} mm")
    print(f"  R(z): {theory_params['R'] * 1e3:.1f} mm")
    print(f"  Gouy: {theory_params['gouy']:.6f} rad")
    
    # 比较
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    # 振幅比较
    amp_ratio = read_amp[mask] / theory_amp[mask]
    print(f"\n振幅比较:")
    print(f"  比值范围: [{np.min(amp_ratio):.6f}, {np.max(amp_ratio):.6f}]")
    print(f"  比值均值: {np.mean(amp_ratio):.6f}")
    
    # 检查是否有缩放因子
    # FFT 会引入缩放，需要归一化
    scale_factor = np.mean(amp_ratio)
    
    amp_error = np.sqrt(np.mean((read_amp[mask]/scale_factor - theory_amp[mask])**2))
    phase_diff = np.angle(np.exp(1j * (read_phase - theory_phase)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n归一化后误差:")
    print(f"  振幅 RMS: {amp_error:.6f}")
    print(f"  相位 RMS: {phase_error:.6f} rad = {phase_error/(2*np.pi):.6f} waves")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    im = ax.imshow(read_amp, cmap='hot')
    ax.set_title('PROPER Amplitude')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(theory_amp, cmap='hot')
    ax.set_title('Theory Amplitude')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow((read_amp/scale_factor - theory_amp) * mask, cmap='RdBu')
    ax.set_title('Amplitude Diff (normalized)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    im = ax.imshow(read_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(theory_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(phase_diff * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Phase Diff (rad)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'INSIDE_to_OUTSIDE: z = {propagation_distance/z_R:.1f} z_R')
    plt.tight_layout()
    plt.savefig('verify_inside_to_outside.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: verify_inside_to_outside.png")
    
    passed = amp_error < 0.01 and phase_error < 0.1
    print(f"\n结果: {'PASS' if passed else 'FAIL'}")
    
    return passed


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    result1 = verify_near_field_ptp()
    result2 = verify_inside_to_outside()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  近场 PTP: {'PASS' if result1 else 'FAIL'}")
    print(f"  INSIDE_to_OUTSIDE: {'PASS' if result2 else 'FAIL'}")
