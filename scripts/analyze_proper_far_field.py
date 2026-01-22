"""
仔细分析 PROPER 远场传播的数学

WTS (Waist-to-Spherical) 传播流程：
1. 添加二次相位：wfarr *= exp(i*π/(λ*dz) * r²)
2. FFT
3. 更新采样：dx_new = λ*|dz| / (ngrid * dx_old)

理论高斯光束传播：
- 在束腰处：E(r,0) = exp(-r²/w0²)
- 传播到 z：E(r,z) = (w0/w) * exp(-r²/w²) * exp(-i*k*r²/(2R)) * exp(i*ψ)
  其中 ψ = arctan(z/z_R) 是 Gouy 相位

关键问题：PROPER 的 qphase 和理论公式的关系是什么？
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def analyze_wts_math():
    """分析 WTS 传播的数学"""
    
    print("=" * 70)
    print("分析 PROPER WTS 传播数学")
    print("=" * 70)
    
    # 参数
    wavelength = 632.8e-9
    w0 = 0.5e-3
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    k = 2 * np.pi / wavelength
    
    # 传播距离
    dz = 5 * z_R
    
    print(f"\n参数:")
    print(f"  λ = {wavelength * 1e9:.1f} nm")
    print(f"  w0 = {w0 * 1e3:.3f} mm")
    print(f"  z_R = {z_R * 1e3:.3f} mm")
    print(f"  dz = {dz * 1e3:.1f} mm = {dz/z_R:.1f} z_R")
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    dx_old = grid_width / grid_size
    
    # 新采样
    dx_new = wavelength * dz / (grid_size * dx_old)
    
    print(f"\n采样:")
    print(f"  dx_old = {dx_old * 1e6:.3f} μm")
    print(f"  dx_new = {dx_new * 1e6:.3f} μm")
    
    # ========== 分析 qphase ==========
    print("\n" + "=" * 70)
    print("分析 prop_qphase")
    print("=" * 70)
    
    # qphase 添加的相位：exp(i*π/(λ*dz) * r²)
    # 这等价于：exp(i * k * r² / (2*dz))
    # 因为 π/(λ*dz) = (2π/λ) / (2*dz) = k / (2*dz)
    
    print(f"\nqphase 添加的相位:")
    print(f"  exp(i*π/(λ*dz) * r²) = exp(i * k * r² / (2*dz))")
    print(f"  这是曲率半径为 dz 的球面波前")
    
    # ========== 理论高斯光束 ==========
    print("\n" + "=" * 70)
    print("理论高斯光束")
    print("=" * 70)
    
    # 在 z=dz 处的参数
    w = w0 * np.sqrt(1 + (dz / z_R)**2)
    R = dz * (1 + (z_R / dz)**2)
    gouy = np.arctan(dz / z_R)
    
    print(f"\n在 z = {dz/z_R:.1f} z_R 处:")
    print(f"  w(z) = {w * 1e3:.3f} mm")
    print(f"  R(z) = {R * 1e3:.1f} mm")
    print(f"  Gouy = {gouy:.6f} rad")
    
    # 理论相位（相对于主光线）：k*r²/(2R) - gouy
    # 但 gouy 是常数，可以忽略
    
    print(f"\n理论相位（相对于主光线）:")
    print(f"  φ(r) = k*r²/(2R) - gouy")
    print(f"  其中 k*r²/(2R) 是球面波前相位")
    
    # ========== 比较 qphase 和理论 ==========
    print("\n" + "=" * 70)
    print("比较 qphase 和理论")
    print("=" * 70)
    
    # qphase 的曲率半径是 dz
    # 理论的曲率半径是 R = dz * (1 + (z_R/dz)²)
    
    print(f"\n曲率半径比较:")
    print(f"  qphase 曲率: dz = {dz * 1e3:.1f} mm")
    print(f"  理论曲率: R = {R * 1e3:.1f} mm")
    print(f"  差异: {(R - dz) * 1e3:.1f} mm")
    print(f"  相对差异: {(R - dz) / R * 100:.2f}%")
    
    # 这说明 qphase 添加的不是最终的球面波前！
    # 而是 Fresnel 传播的一部分
    
    print("\n关键发现：")
    print("  qphase 添加的是 Fresnel 传播核的一部分，不是最终的球面波前！")
    print("  PROPER 使用的是 Fresnel 近似传播，不是直接计算高斯光束公式。")


def verify_fresnel_propagation():
    """验证 Fresnel 传播"""
    
    print("\n" + "=" * 70)
    print("验证 Fresnel 传播")
    print("=" * 70)
    
    # Fresnel 传播公式：
    # E(x',y',z) = exp(ikz)/(iλz) * ∫∫ E(x,y,0) * exp(ik/(2z)*[(x'-x)² + (y'-y)²]) dx dy
    #
    # 这可以写成卷积形式，用 FFT 计算：
    # E(x',y',z) = exp(ikz)/(iλz) * exp(ik/(2z)*(x'²+y'²)) * FFT[E(x,y,0) * exp(ik/(2z)*(x²+y²))]
    #
    # PROPER 的 WTS 做的就是这个：
    # 1. qphase: 乘以 exp(ik/(2z)*(x²+y²))
    # 2. FFT
    # 3. 输出坐标变换（采样变化）
    
    # 参数
    wavelength = 632.8e-9
    w0 = 0.5e-3
    grid_size = 256
    
    z_R = np.pi * w0**2 / wavelength
    k = 2 * np.pi / wavelength
    dz = 5 * z_R
    
    # 初始采样
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.3
    grid_width = beam_diameter / beam_diam_fraction
    dx_old = grid_width / grid_size
    
    # 创建初始高斯光束
    n = grid_size
    x_old = (np.arange(n) - n // 2) * dx_old
    X_old, Y_old = np.meshgrid(x_old, x_old)
    r_sq_old = X_old**2 + Y_old**2
    
    E_initial = np.exp(-r_sq_old / w0**2)
    
    # ========== 手动 Fresnel 传播 ==========
    # 1. 乘以二次相位
    E_with_qphase = E_initial * np.exp(1j * k / (2 * dz) * r_sq_old)
    
    # 2. FFT（shift 到中心）
    E_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_with_qphase)))
    
    # 3. 新坐标
    dx_new = wavelength * dz / (grid_size * dx_old)
    x_new = (np.arange(n) - n // 2) * dx_new
    X_new, Y_new = np.meshgrid(x_new, x_new)
    r_sq_new = X_new**2 + Y_new**2
    
    # 4. 乘以输出端的二次相位（Fresnel 传播的另一部分）
    # 注意：PROPER 的 WTS 没有显式乘这个，因为它被吸收到参考面中了
    E_fresnel = E_fft * np.exp(1j * k / (2 * dz) * r_sq_new)
    
    # 归一化
    E_fresnel = E_fresnel / (1j * wavelength * dz) * (dx_old**2)
    
    # ========== 理论高斯光束 ==========
    w = w0 * np.sqrt(1 + (dz / z_R)**2)
    R = dz * (1 + (z_R / dz)**2)
    gouy = np.arctan(dz / z_R)
    
    E_theory = (w0 / w) * np.exp(-r_sq_new / w**2) * np.exp(-1j * k * r_sq_new / (2 * R)) * np.exp(1j * gouy)
    
    # ========== 比较 ==========
    print("\n比较手动 Fresnel 传播和理论高斯光束:")
    
    # 振幅
    amp_fresnel = np.abs(E_fresnel)
    amp_theory = np.abs(E_theory)
    
    # 归一化到峰值
    amp_fresnel = amp_fresnel / np.max(amp_fresnel)
    amp_theory = amp_theory / np.max(amp_theory)
    
    mask = amp_theory > 0.01
    amp_error = np.sqrt(np.mean((amp_fresnel[mask] - amp_theory[mask])**2))
    
    print(f"  振幅误差 RMS: {amp_error:.6f}")
    
    # 相位（减去中心相位）
    phase_fresnel = np.angle(E_fresnel)
    phase_theory = np.angle(E_theory)
    
    center = n // 2
    phase_fresnel_rel = phase_fresnel - phase_fresnel[center, center]
    phase_theory_rel = phase_theory - phase_theory[center, center]
    
    phase_diff = np.angle(np.exp(1j * (phase_fresnel_rel - phase_theory_rel)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"  相位误差 RMS: {phase_error:.6f} rad")
    
    # ========== 现在用 PROPER ==========
    print("\n" + "=" * 70)
    print("使用 PROPER 传播")
    print("=" * 70)
    
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    
    proper.prop_propagate(wfo, dz)
    
    E_proper_amp = proper.prop_get_amplitude(wfo)
    E_proper_phase = proper.prop_get_phase(wfo)
    
    # 归一化振幅
    E_proper_amp_norm = E_proper_amp / np.max(E_proper_amp)
    
    amp_error_proper = np.sqrt(np.mean((E_proper_amp_norm[mask] - amp_theory[mask])**2))
    print(f"  PROPER 振幅误差 RMS: {amp_error_proper:.6f}")
    
    # 相位（减去中心相位）
    E_proper_phase_rel = E_proper_phase - E_proper_phase[center, center]
    
    phase_diff_proper = np.angle(np.exp(1j * (E_proper_phase_rel - phase_theory_rel)))
    phase_error_proper = np.sqrt(np.mean(phase_diff_proper[mask]**2))
    
    print(f"  PROPER 相位误差 RMS: {phase_error_proper:.6f} rad")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    ax = axes[0, 0]
    im = ax.imshow(amp_fresnel, cmap='hot')
    ax.set_title('Manual Fresnel Amp')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(E_proper_amp_norm, cmap='hot')
    ax.set_title('PROPER Amp')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(amp_theory, cmap='hot')
    ax.set_title('Theory Amp')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    im = ax.imshow((E_proper_amp_norm - amp_theory) * mask, cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title('PROPER - Theory')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    im = ax.imshow(phase_fresnel_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Manual Fresnel Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(E_proper_phase_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(phase_theory_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 3]
    im = ax.imshow(phase_diff_proper * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('PROPER - Theory Phase')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Fresnel Propagation Analysis')
    plt.tight_layout()
    plt.savefig('analyze_proper_far_field.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: analyze_proper_far_field.png")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    analyze_wts_math()
    verify_fresnel_propagation()
