"""
调试远场传播问题
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def gaussian_beam_params(w0, wavelength, z):
    """计算高斯光束参数"""
    z_R = np.pi * w0**2 / wavelength
    
    if abs(z) < 1e-12:
        return w0, np.inf, 0.0, z_R
    
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)
    gouy = np.arctan(z / z_R)
    
    return w, R, gouy, z_R


def debug_far_field():
    """调试远场传播"""
    
    print("=" * 70)
    print("调试远场传播")
    print("=" * 70)
    
    wavelength = 632.8e-9
    w0 = 0.5e-3
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  w0: {w0 * 1e3:.3f} mm")
    print(f"  z_R: {z_R * 1e3:.3f} mm")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    sampling_initial = grid_width / grid_size
    
    print(f"  初始采样: {sampling_initial * 1e6:.3f} μm")
    print(f"  初始网格宽度: {grid_width * 1e3:.3f} mm")
    
    # 创建初始高斯光束（在束腰处）
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_initial
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    initial_amp = np.exp(-r_sq / w0**2)
    initial_phase = np.zeros_like(r_sq)
    initial_field = initial_amp * np.exp(1j * initial_phase)
    
    # 设置 PROPER
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(initial_field.astype(np.complex128))
    
    print(f"\n初始状态:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 传播到远场
    propagation_distance = 10 * z_R
    proper.prop_propagate(wfo, propagation_distance)
    
    z_final = propagation_distance
    new_sampling = wfo.dx
    
    print(f"\n传播后:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.dx: {new_sampling * 1e6:.3f} μm")
    print(f"  reference_surface: {wfo.reference_surface}")
    print(f"  新网格宽度: {new_sampling * grid_size * 1e3:.3f} mm")
    
    # 理论参数
    w_final, R_final, gouy_final, _ = gaussian_beam_params(w0, wavelength, z_final)
    
    print(f"\n理论参数:")
    print(f"  w(z): {w_final * 1e3:.3f} mm")
    print(f"  R(z): {R_final * 1e3:.3f} mm")
    print(f"  Gouy: {gouy_final:.6f} rad")
    
    # 读取 PROPER 结果
    read_amp = proper.prop_get_amplitude(wfo)
    read_phase = proper.prop_get_phase(wfo)
    
    # 创建新坐标网格（使用新采样）
    x_new = (np.arange(n) - n // 2) * new_sampling
    X_new, Y_new = np.meshgrid(x_new, x_new)
    r_sq_new = X_new**2 + Y_new**2
    
    # 理论振幅
    theory_amp = (w0 / w_final) * np.exp(-r_sq_new / w_final**2)
    
    # 理论相位（包含 Gouy 相位）
    k = 2 * np.pi / wavelength
    theory_spherical = k * r_sq_new / (2 * R_final)
    theory_phase = theory_spherical - gouy_final
    
    # 比较振幅
    print(f"\n振幅比较:")
    print(f"  PROPER 振幅范围: [{np.min(read_amp):.6f}, {np.max(read_amp):.6f}]")
    print(f"  理论振幅范围: [{np.min(theory_amp):.6f}, {np.max(theory_amp):.6f}]")
    print(f"  PROPER 振幅峰值位置: {np.unravel_index(np.argmax(read_amp), read_amp.shape)}")
    print(f"  理论振幅峰值位置: {np.unravel_index(np.argmax(theory_amp), theory_amp.shape)}")
    
    # 检查振幅分布
    center = n // 2
    print(f"\n中心线振幅:")
    print(f"  PROPER: {read_amp[center, center-5:center+6]}")
    print(f"  理论:   {theory_amp[center, center-5:center+6]}")
    
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
    im = ax.imshow(read_amp - theory_amp, cmap='RdBu')
    ax.set_title('Amplitude Difference')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    im = ax.imshow(read_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER Phase (wrapped)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(np.mod(theory_phase + np.pi, 2*np.pi) - np.pi, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory Phase (wrapped)')
    plt.colorbar(im, ax=ax)
    
    # 中心线剖面
    ax = axes[1, 2]
    ax.plot(x_new * 1e3, read_amp[center, :], 'b-', label='PROPER')
    ax.plot(x_new * 1e3, theory_amp[center, :], 'r--', label='Theory')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.set_title('Center Line Profile')
    
    plt.suptitle(f'Far-field Propagation: z = {z_final/z_R:.1f} z_R')
    plt.tight_layout()
    plt.savefig('debug_far_field_propagation.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: debug_far_field_propagation.png")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    debug_far_field()
