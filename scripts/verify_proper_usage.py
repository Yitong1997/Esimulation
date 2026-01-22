"""
验证 PROPER 库的正确使用方式

核心逻辑：
1. 写入 PROPER：根据仿真复振幅和 pilot beam 参数，正确设置 wfo 的所有属性
2. 读取 PROPER：直接读取，不需要减去参考波面，但需要用 pilot beam 解包裹

验证方法：理论高斯光束的传播
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def gaussian_beam_params(w0, wavelength, z):
    """计算高斯光束在位置 z 处的参数（严格公式）
    
    参数:
        w0: 束腰半径 (m)
        wavelength: 波长 (m)
        z: 距离束腰的传播距离 (m)
    
    返回:
        w: 光斑半径 (m)
        R: 曲率半径 (m)，平面波时返回 inf
        gouy: Gouy 相位 (rad)
        z_R: 瑞利距离 (m)
    """
    z_R = np.pi * w0**2 / wavelength
    
    if abs(z) < 1e-12:
        # 在束腰处
        return w0, np.inf, 0.0, z_R
    
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)  # 严格公式
    gouy = np.arctan(z / z_R)
    
    return w, R, gouy, z_R


def create_theoretical_gaussian(grid_size, sampling_m, w0, wavelength, z):
    """创建理论高斯光束复振幅
    
    参数:
        grid_size: 网格大小
        sampling_m: 采样间距 (m)
        w0: 束腰半径 (m)
        wavelength: 波长 (m)
        z: 距离束腰的传播距离 (m)
    
    返回:
        complex_amplitude: 复振幅数组
        params: 高斯光束参数字典
    """
    w, R, gouy, z_R = gaussian_beam_params(w0, wavelength, z)
    
    # 创建坐标网格
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_m
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength
    
    # 高斯光束复振幅（相对于主光线，主光线处相位为 0）
    # E(r,z) = (w0/w) * exp(-r²/w²) * exp(-i*k*r²/(2R)) * exp(i*gouy)
    # 
    # 注意：Gouy 相位是空间常数，在计算相对于主光线的相位时会被抵消
    # 所以我们只需要考虑球面波前相位
    
    amplitude = (w0 / w) * np.exp(-r_sq / w**2)
    
    if np.isinf(R):
        phase = np.zeros_like(r_sq)
    else:
        # 球面波前相位（相对于主光线的相位延迟）
        phase = k * r_sq / (2 * R)
    
    complex_amplitude = amplitude * np.exp(1j * phase)
    
    params = {
        'w0': w0,
        'w': w,
        'R': R,
        'gouy': gouy,
        'z_R': z_R,
        'z': z
    }
    
    return complex_amplitude, params


def setup_proper_from_simulation(simulation_amplitude, pilot_params, grid_size, sampling_m, wavelength):
    """从仿真复振幅设置 PROPER wfo 对象
    
    正确的逻辑：
    1. 使用 prop_begin 初始化 wfo
    2. 设置 wfo 的所有属性以匹配 pilot beam 参数
    3. 将仿真复振幅写入 wfarr
    
    参数:
        simulation_amplitude: 仿真复振幅（绝对相位，非折叠）
        pilot_params: pilot beam 参数字典
        grid_size: 网格大小
        sampling_m: 采样间距 (m)
        wavelength: 波长 (m)
    
    返回:
        wfo: 设置好的 PROPER wfo 对象
    """
    # 从 pilot beam 参数获取 w0
    w0 = pilot_params['w0']
    beam_diameter = 2 * w0
    
    # 计算 beam_diam_fraction
    grid_width = grid_size * sampling_m
    beam_diam_fraction = beam_diameter / grid_width
    
    # 初始化 wfo
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    # 设置所有属性以匹配 pilot beam 参数
    wfo.z = pilot_params['z']
    wfo.z_w0 = 0.0  # 假设束腰在 z=0
    wfo.w0 = w0
    wfo.z_Rayleigh = pilot_params['z_R']
    wfo.dx = sampling_m
    
    # 确定参考面类型
    rayleigh_factor = proper.rayleigh_factor
    if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
        wfo.reference_surface = "PLANAR"
        wfo.beam_type_old = "INSIDE_"
    else:
        wfo.reference_surface = "SPHERI"
        wfo.beam_type_old = "OUTSIDE"
    
    # 将仿真复振幅写入 wfarr
    # 注意：需要使用 prop_shift_center 转换到 FFT 坐标系
    wfo.wfarr = proper.prop_shift_center(simulation_amplitude.astype(np.complex128))
    
    return wfo


def read_simulation_from_proper(wfo, pilot_params, grid_size, sampling_m, wavelength):
    """从 PROPER wfo 读取仿真复振幅
    
    正确的逻辑：
    1. 直接读取振幅和相位
    2. 相位是包裹相位，需要用 pilot beam 解包裹
    
    参数:
        wfo: PROPER wfo 对象
        pilot_params: pilot beam 参数字典
        grid_size: 网格大小
        sampling_m: 采样间距 (m)
        wavelength: 波长 (m)
    
    返回:
        simulation_amplitude: 仿真复振幅（绝对相位，非折叠）
    """
    # 直接读取振幅和相位
    amplitude = proper.prop_get_amplitude(wfo)
    wrapped_phase = proper.prop_get_phase(wfo)
    
    # 计算 pilot beam 参考相位（非折叠）
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_m
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength
    R = pilot_params['R']
    
    if np.isinf(R):
        pilot_phase = np.zeros_like(r_sq)
    else:
        pilot_phase = k * r_sq / (2 * R)
    
    # 使用 pilot beam 解包裹
    # T_unwrapped = T_pilot + angle(exp(i * (T - T_pilot)))
    phase_diff = wrapped_phase - pilot_phase
    unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
    
    # 组合为仿真复振幅
    simulation_amplitude = amplitude * np.exp(1j * unwrapped_phase)
    
    return simulation_amplitude, wrapped_phase, pilot_phase, unwrapped_phase


def verify_gaussian_propagation():
    """验证高斯光束传播的正确性"""
    
    print("=" * 70)
    print("验证 PROPER 使用方式：高斯光束传播")
    print("=" * 70)
    
    # 参数设置
    wavelength = 632.8e-9  # m
    w0 = 1e-3  # 束腰半径 1 mm
    grid_size = 256
    
    # 初始位置（在束腰处）
    z_initial = 0.0
    
    # 传播距离
    propagation_distance = 0.5  # 0.5 m
    
    # 计算瑞利距离
    z_R = np.pi * w0**2 / wavelength
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  束腰半径 w0: {w0 * 1e3:.3f} mm")
    print(f"  瑞利距离 z_R: {z_R * 1e3:.3f} mm")
    print(f"  传播距离: {propagation_distance * 1e3:.1f} mm")
    print(f"  传播距离/瑞利距离: {propagation_distance / z_R:.2f}")
    
    # 计算初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3  # 光束占网格的 30%
    grid_width = beam_diameter / beam_diam_fraction
    sampling_m = grid_width / grid_size
    
    print(f"  网格大小: {grid_size}")
    print(f"  采样间距: {sampling_m * 1e6:.3f} μm")
    print(f"  网格宽度: {grid_width * 1e3:.3f} mm")
    
    # ========== 步骤 1：创建初始高斯光束 ==========
    print("\n" + "=" * 70)
    print("步骤 1：创建初始高斯光束（在束腰处）")
    print("=" * 70)
    
    initial_amplitude, initial_params = create_theoretical_gaussian(
        grid_size, sampling_m, w0, wavelength, z_initial
    )
    
    print(f"  初始光斑半径 w: {initial_params['w'] * 1e3:.6f} mm")
    print(f"  初始曲率半径 R: {'inf' if np.isinf(initial_params['R']) else f'{initial_params["R"] * 1e3:.3f} mm'}")
    
    # ========== 步骤 2：设置 PROPER ==========
    print("\n" + "=" * 70)
    print("步骤 2：从仿真复振幅设置 PROPER")
    print("=" * 70)
    
    wfo = setup_proper_from_simulation(
        initial_amplitude, initial_params, grid_size, sampling_m, wavelength
    )
    
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  wfo.w0: {wfo.w0 * 1e3:.6f} mm")
    print(f"  wfo.z_Rayleigh: {wfo.z_Rayleigh * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  wfo.reference_surface: {wfo.reference_surface}")
    print(f"  wfo.beam_type_old: {wfo.beam_type_old}")
    
    # 验证写入是否正确
    read_amp = proper.prop_get_amplitude(wfo)
    read_phase = proper.prop_get_phase(wfo)
    
    amp_error = np.max(np.abs(read_amp - np.abs(initial_amplitude)))
    print(f"\n  写入验证:")
    print(f"    振幅最大误差: {amp_error:.2e}")
    
    # ========== 步骤 3：PROPER 传播 ==========
    print("\n" + "=" * 70)
    print("步骤 3：PROPER 传播")
    print("=" * 70)
    
    proper.prop_propagate(wfo, propagation_distance)
    
    print(f"  传播后 wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  传播后 wfo.z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  传播后 wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  传播后 wfo.reference_surface: {wfo.reference_surface}")
    
    # ========== 步骤 4：计算传播后的 pilot beam 参数 ==========
    print("\n" + "=" * 70)
    print("步骤 4：计算传播后的 pilot beam 参数")
    print("=" * 70)
    
    z_final = z_initial + propagation_distance
    _, final_params = create_theoretical_gaussian(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    
    print(f"  理论光斑半径 w: {final_params['w'] * 1e3:.6f} mm")
    print(f"  理论曲率半径 R: {final_params['R'] * 1e3:.3f} mm")
    
    # ========== 步骤 5：从 PROPER 读取仿真复振幅 ==========
    print("\n" + "=" * 70)
    print("步骤 5：从 PROPER 读取仿真复振幅")
    print("=" * 70)
    
    final_amplitude, wrapped_phase, pilot_phase, unwrapped_phase = read_simulation_from_proper(
        wfo, final_params, grid_size, wfo.dx, wavelength
    )
    
    # ========== 步骤 6：与理论值比较 ==========
    print("\n" + "=" * 70)
    print("步骤 6：与理论值比较")
    print("=" * 70)
    
    # 创建理论传播后的高斯光束
    theoretical_amplitude, _ = create_theoretical_gaussian(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    
    # 计算误差
    amp_read = np.abs(final_amplitude)
    amp_theory = np.abs(theoretical_amplitude)
    phase_read = np.angle(final_amplitude)
    phase_theory = np.angle(theoretical_amplitude)
    
    # 只在有效区域（振幅 > 1% 峰值）比较
    mask = amp_theory > 0.01 * np.max(amp_theory)
    
    amp_error_rms = np.sqrt(np.mean((amp_read[mask] - amp_theory[mask])**2))
    amp_error_max = np.max(np.abs(amp_read[mask] - amp_theory[mask]))
    
    # 相位误差（考虑 2π 周期）
    phase_diff = np.angle(np.exp(1j * (phase_read - phase_theory)))
    phase_error_rms = np.sqrt(np.mean(phase_diff[mask]**2))
    phase_error_max = np.max(np.abs(phase_diff[mask]))
    
    print(f"  振幅误差 (RMS): {amp_error_rms:.6f}")
    print(f"  振幅误差 (Max): {amp_error_max:.6f}")
    print(f"  相位误差 (RMS): {phase_error_rms:.6f} rad = {phase_error_rms / (2*np.pi):.6f} waves")
    print(f"  相位误差 (Max): {phase_error_max:.6f} rad = {phase_error_max / (2*np.pi):.6f} waves")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 第一行：初始状态
    ax = axes[0, 0]
    im = ax.imshow(np.abs(initial_amplitude), cmap='hot')
    ax.set_title('初始振幅')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(np.angle(initial_amplitude), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('初始相位')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    ax.axis('off')
    ax.text(0.5, 0.5, f'初始参数:\nw0 = {w0*1e3:.3f} mm\nz = {z_initial*1e3:.1f} mm\nR = inf',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    ax = axes[0, 3]
    ax.axis('off')
    
    # 第二行：PROPER 读取结果
    ax = axes[1, 0]
    im = ax.imshow(amp_read, cmap='hot')
    ax.set_title('PROPER 读取振幅')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(wrapped_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER 包裹相位')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(pilot_phase, cmap='twilight')
    ax.set_title('Pilot Beam 参考相位')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 3]
    im = ax.imshow(unwrapped_phase, cmap='twilight')
    ax.set_title('解包裹后相位')
    plt.colorbar(im, ax=ax)
    
    # 第三行：与理论比较
    ax = axes[2, 0]
    im = ax.imshow(amp_theory, cmap='hot')
    ax.set_title('理论振幅')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 1]
    im = ax.imshow(phase_theory, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('理论相位')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 2]
    im = ax.imshow((amp_read - amp_theory) * mask, cmap='RdBu', 
                   vmin=-0.01, vmax=0.01)
    ax.set_title('振幅误差')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 3]
    im = ax.imshow(phase_diff * mask, cmap='RdBu',
                   vmin=-0.1, vmax=0.1)
    ax.set_title('相位误差 (rad)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'高斯光束传播验证: z = {propagation_distance*1e3:.1f} mm (z/z_R = {propagation_distance/z_R:.2f})')
    plt.tight_layout()
    plt.savefig('verify_proper_usage_gaussian.png', dpi=150)
    plt.close()
    
    print(f"\n图像已保存到: verify_proper_usage_gaussian.png")
    
    # 判断是否通过
    passed = amp_error_rms < 0.01 and phase_error_rms < 0.1
    print(f"\n验证结果: {'通过 ✓' if passed else '失败 ✗'}")
    
    return passed


def verify_far_field_propagation():
    """验证远场传播（z >> z_R）"""
    
    print("\n" + "=" * 70)
    print("验证远场传播 (z >> z_R)")
    print("=" * 70)
    
    # 参数设置
    wavelength = 632.8e-9  # m
    w0 = 0.5e-3  # 束腰半径 0.5 mm
    grid_size = 256
    
    # 计算瑞利距离
    z_R = np.pi * w0**2 / wavelength
    
    # 传播到远场（10 倍瑞利距离）
    propagation_distance = 10 * z_R
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  束腰半径 w0: {w0 * 1e3:.3f} mm")
    print(f"  瑞利距离 z_R: {z_R * 1e3:.3f} mm")
    print(f"  传播距离: {propagation_distance * 1e3:.1f} mm")
    print(f"  传播距离/瑞利距离: {propagation_distance / z_R:.1f}")
    
    # 计算初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_m = grid_width / grid_size
    
    # 创建初始高斯光束
    initial_amplitude, initial_params = create_theoretical_gaussian(
        grid_size, sampling_m, w0, wavelength, 0.0
    )
    
    # 设置 PROPER
    wfo = setup_proper_from_simulation(
        initial_amplitude, initial_params, grid_size, sampling_m, wavelength
    )
    
    # 传播
    proper.prop_propagate(wfo, propagation_distance)
    
    # 计算传播后的 pilot beam 参数
    z_final = propagation_distance
    _, final_params = create_theoretical_gaussian(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    
    print(f"\n传播后:")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  wfo.reference_surface: {wfo.reference_surface}")
    print(f"  理论光斑半径 w: {final_params['w'] * 1e3:.3f} mm")
    print(f"  理论曲率半径 R: {final_params['R'] * 1e3:.1f} mm")
    
    # 从 PROPER 读取
    final_amplitude, wrapped_phase, pilot_phase, unwrapped_phase = read_simulation_from_proper(
        wfo, final_params, grid_size, wfo.dx, wavelength
    )
    
    # 创建理论值
    theoretical_amplitude, _ = create_theoretical_gaussian(
        grid_size, wfo.dx, w0, wavelength, z_final
    )
    
    # 计算误差
    amp_read = np.abs(final_amplitude)
    amp_theory = np.abs(theoretical_amplitude)
    phase_read = np.angle(final_amplitude)
    phase_theory = np.angle(theoretical_amplitude)
    
    mask = amp_theory > 0.01 * np.max(amp_theory)
    
    amp_error_rms = np.sqrt(np.mean((amp_read[mask] - amp_theory[mask])**2))
    phase_diff = np.angle(np.exp(1j * (phase_read - phase_theory)))
    phase_error_rms = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n误差:")
    print(f"  振幅误差 (RMS): {amp_error_rms:.6f}")
    print(f"  相位误差 (RMS): {phase_error_rms:.6f} rad = {phase_error_rms / (2*np.pi):.6f} waves")
    
    passed = amp_error_rms < 0.01 and phase_error_rms < 0.1
    print(f"\n验证结果: {'通过 ✓' if passed else '失败 ✗'}")
    
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROPER 使用方式验证")
    print("=" * 70)
    
    # 关闭 PROPER 的打印输出
    proper.print_it = False
    proper.verbose = False
    
    # 验证近场传播
    result1 = verify_gaussian_propagation()
    
    # 验证远场传播
    result2 = verify_far_field_propagation()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  近场传播: {'通过 ✓' if result1 else '失败 ✗'}")
    print(f"  远场传播: {'通过 ✓' if result2 else '失败 ✗'}")
    
    if result1 and result2:
        print("\n所有验证通过！PROPER 使用方式正确。")
    else:
        print("\n部分验证失败，需要进一步调查。")
