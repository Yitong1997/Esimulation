"""
验证 PROPER 的正确理解：wfarr 存储的是完整复振幅

核心理解：
1. wfarr 存储的是实际的完整复振幅，不是残差
2. 高斯光束参数只用于决定传播算法和优化计算
3. 读取时直接读取，写入时直接写入
4. 读取的相位是包裹相位，需要用 pilot beam 解包裹

验证方法：
1. 创建一个已知的高斯光束复振幅
2. 直接写入 wfarr
3. 传播
4. 直接读取并与理论值比较
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


def create_gaussian_amplitude(grid_size, sampling_m, w0, wavelength, z, include_gouy=True):
    """创建高斯光束复振幅
    
    注意：PROPER 传播会包含 Gouy 相位，所以理论公式也需要包含
    """
    w, R, gouy, z_R = gaussian_beam_params(w0, wavelength, z)
    
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_m
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength
    
    # 振幅
    amplitude = (w0 / w) * np.exp(-r_sq / w**2)
    
    # 相位
    # 球面波前相位（相对于主光线）
    if np.isinf(R):
        spherical_phase = np.zeros_like(r_sq)
    else:
        spherical_phase = k * r_sq / (2 * R)
    
    # Gouy 相位（空间常数，但 PROPER 会包含它）
    if include_gouy:
        phase = spherical_phase - gouy  # 负号因为 Gouy 相位是相位延迟
    else:
        phase = spherical_phase
    
    return amplitude * np.exp(1j * phase), {'w': w, 'R': R, 'gouy': gouy, 'z_R': z_R}


def setup_proper_direct(complex_amplitude, w0, wavelength, z, grid_size, sampling_m):
    """直接设置 PROPER：wfarr = 完整复振幅"""
    beam_diameter = 2 * w0
    z_R = np.pi * w0**2 / wavelength
    
    grid_width = grid_size * sampling_m
    beam_diam_fraction = beam_diameter / grid_width
    
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    # 设置位置和参数
    wfo.z = z
    wfo.z_w0 = 0.0
    wfo.w0 = w0
    wfo.z_Rayleigh = z_R
    wfo.dx = sampling_m
    
    # 确定参考面类型
    rayleigh_factor = proper.rayleigh_factor
    if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
        wfo.reference_surface = "PLANAR"
        wfo.beam_type_old = "INSIDE_"
    else:
        wfo.reference_surface = "SPHERI"
        wfo.beam_type_old = "OUTSIDE"
    
    # 直接写入完整复振幅
    wfo.wfarr = proper.prop_shift_center(complex_amplitude.astype(np.complex128))
    
    return wfo


def read_proper_direct(wfo):
    """直接读取 PROPER：返回完整复振幅"""
    amplitude = proper.prop_get_amplitude(wfo)
    phase = proper.prop_get_phase(wfo)  # 包裹相位
    return amplitude, phase


def unwrap_with_pilot_beam(wrapped_phase, pilot_phase):
    """使用 pilot beam 解包裹"""
    phase_diff = wrapped_phase - pilot_phase
    return pilot_phase + np.angle(np.exp(1j * phase_diff))


def verify_gaussian_propagation():
    """验证高斯光束传播"""
    
    print("=" * 70)
    print("验证：wfarr 存储完整复振幅")
    print("=" * 70)
    
    # 参数
    wavelength = 632.8e-9  # m
    w0 = 1e-3  # 1 mm
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  束腰半径 w0: {w0 * 1e3:.3f} mm")
    print(f"  瑞利距离 z_R: {z_R * 1e3:.3f} mm")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_m = grid_width / grid_size
    
    print(f"  采样间距: {sampling_m * 1e6:.3f} μm")
    
    # ========== 测试 1：在束腰处创建高斯光束 ==========
    print("\n" + "=" * 70)
    print("测试 1：在束腰处 (z=0) 创建高斯光束")
    print("=" * 70)
    
    z_initial = 0.0
    initial_amp, initial_params = create_gaussian_amplitude(
        grid_size, sampling_m, w0, wavelength, z_initial
    )
    
    print(f"  理论光斑半径: {initial_params['w'] * 1e3:.6f} mm")
    R_str = 'inf' if np.isinf(initial_params['R']) else f"{initial_params['R']:.3f} m"
    print(f"  理论曲率半径: {R_str}")
    
    # 设置 PROPER
    wfo = setup_proper_direct(initial_amp, w0, wavelength, z_initial, grid_size, sampling_m)
    
    # 验证写入
    read_amp, read_phase = read_proper_direct(wfo)
    
    write_amp_error = np.max(np.abs(read_amp - np.abs(initial_amp)))
    write_phase_error = np.max(np.abs(read_phase - np.angle(initial_amp)))
    
    print(f"\n  写入验证:")
    print(f"    振幅最大误差: {write_amp_error:.2e}")
    print(f"    相位最大误差: {write_phase_error:.6f} rad")
    
    # ========== 测试 2：传播 ==========
    print("\n" + "=" * 70)
    print("测试 2：传播 0.5 m")
    print("=" * 70)
    
    propagation_distance = 0.5  # m
    proper.prop_propagate(wfo, propagation_distance)
    
    z_final = z_initial + propagation_distance
    new_sampling = wfo.dx
    
    print(f"  传播后 z: {wfo.z * 1e3:.3f} mm")
    print(f"  传播后采样: {new_sampling * 1e6:.3f} μm")
    print(f"  参考面类型: {wfo.reference_surface}")
    
    # ========== 测试 3：读取并与理论比较 ==========
    print("\n" + "=" * 70)
    print("测试 3：读取并与理论比较")
    print("=" * 70)
    
    # 读取
    read_amp_final, read_phase_final = read_proper_direct(wfo)
    
    # 理论值
    theory_amp, theory_params = create_gaussian_amplitude(
        grid_size, new_sampling, w0, wavelength, z_final
    )
    theory_amplitude = np.abs(theory_amp)
    theory_phase = np.angle(theory_amp)
    
    print(f"  理论光斑半径: {theory_params['w'] * 1e3:.6f} mm")
    print(f"  理论曲率半径: {theory_params['R'] * 1e3:.3f} mm")
    
    # 计算 pilot beam 相位用于解包裹
    n = grid_size
    x = (np.arange(n) - n // 2) * new_sampling
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    k = 2 * np.pi / wavelength
    R = theory_params['R']
    pilot_phase = k * r_sq / (2 * R)
    
    # 解包裹
    unwrapped_phase = unwrap_with_pilot_beam(read_phase_final, pilot_phase)
    
    # 比较（有效区域）
    mask = theory_amplitude > 0.01 * np.max(theory_amplitude)
    
    amp_error_rms = np.sqrt(np.mean((read_amp_final[mask] - theory_amplitude[mask])**2))
    amp_error_max = np.max(np.abs(read_amp_final[mask] - theory_amplitude[mask]))
    
    # 相位误差（使用解包裹后的相位）
    phase_diff = np.angle(np.exp(1j * (unwrapped_phase - theory_phase)))
    phase_error_rms = np.sqrt(np.mean(phase_diff[mask]**2))
    phase_error_max = np.max(np.abs(phase_diff[mask]))
    
    print(f"\n  误差分析:")
    print(f"    振幅误差 RMS: {amp_error_rms:.6f}")
    print(f"    振幅误差 Max: {amp_error_max:.6f}")
    print(f"    相位误差 RMS: {phase_error_rms:.6f} rad = {phase_error_rms/(2*np.pi):.6f} waves")
    print(f"    相位误差 Max: {phase_error_max:.6f} rad = {phase_error_max/(2*np.pi):.6f} waves")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行：PROPER 读取结果
    ax = axes[0, 0]
    im = ax.imshow(read_amp_final, cmap='hot')
    ax.set_title('PROPER Amplitude')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(read_phase_final, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER Phase (wrapped)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(pilot_phase, cmap='twilight')
    ax.set_title('Pilot Beam Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    im = ax.imshow(unwrapped_phase, cmap='twilight')
    ax.set_title('Unwrapped Phase')
    plt.colorbar(im, ax=ax)
    
    # 第二行：理论值和误差
    ax = axes[1, 0]
    im = ax.imshow(theory_amplitude, cmap='hot')
    ax.set_title('Theory Amplitude')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(theory_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow((read_amp_final - theory_amplitude) * mask, cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title('Amplitude Error')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 3]
    im = ax.imshow(phase_diff * mask, cmap='RdBu', vmin=-0.1, vmax=0.1)
    ax.set_title('Phase Error (rad)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Gaussian Beam Propagation: z = {propagation_distance*1e3:.1f} mm (z/z_R = {propagation_distance/z_R:.2f})')
    plt.tight_layout()
    plt.savefig('verify_proper_wfarr_full_amplitude.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: verify_proper_wfarr_full_amplitude.png")
    
    # 判断
    passed = amp_error_rms < 0.01 and phase_error_rms < 0.1
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    
    return passed


def verify_far_field():
    """验证远场传播"""
    
    print("\n" + "=" * 70)
    print("验证远场传播 (z >> z_R)")
    print("=" * 70)
    
    wavelength = 632.8e-9
    w0 = 0.5e-3
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    propagation_distance = 10 * z_R
    
    print(f"\n参数:")
    print(f"  z/z_R = {propagation_distance/z_R:.1f}")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_m = grid_width / grid_size
    
    # 创建初始高斯光束
    initial_amp, _ = create_gaussian_amplitude(grid_size, sampling_m, w0, wavelength, 0.0)
    
    # 设置 PROPER
    wfo = setup_proper_direct(initial_amp, w0, wavelength, 0.0, grid_size, sampling_m)
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance)
    
    new_sampling = wfo.dx
    z_final = propagation_distance
    
    print(f"  传播后采样: {new_sampling * 1e6:.3f} μm")
    print(f"  参考面类型: {wfo.reference_surface}")
    
    # 读取
    read_amp, read_phase = read_proper_direct(wfo)
    
    # 理论值
    theory_amp, theory_params = create_gaussian_amplitude(
        grid_size, new_sampling, w0, wavelength, z_final
    )
    
    # Pilot beam 相位
    n = grid_size
    x = (np.arange(n) - n // 2) * new_sampling
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    k = 2 * np.pi / wavelength
    R = theory_params['R']
    pilot_phase = k * r_sq / (2 * R)
    
    # 解包裹
    unwrapped_phase = unwrap_with_pilot_beam(read_phase, pilot_phase)
    
    # 比较
    theory_amplitude = np.abs(theory_amp)
    theory_phase = np.angle(theory_amp)
    mask = theory_amplitude > 0.01 * np.max(theory_amplitude)
    
    amp_error_rms = np.sqrt(np.mean((read_amp[mask] - theory_amplitude[mask])**2))
    phase_diff = np.angle(np.exp(1j * (unwrapped_phase - theory_phase)))
    phase_error_rms = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n  误差:")
    print(f"    振幅误差 RMS: {amp_error_rms:.6f}")
    print(f"    相位误差 RMS: {phase_error_rms:.6f} rad = {phase_error_rms/(2*np.pi):.6f} waves")
    
    passed = amp_error_rms < 0.01 and phase_error_rms < 0.1
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    
    return passed


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    result1 = verify_gaussian_propagation()
    result2 = verify_far_field()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Near-field: {'PASS' if result1 else 'FAIL'}")
    print(f"  Far-field: {'PASS' if result2 else 'FAIL'}")
