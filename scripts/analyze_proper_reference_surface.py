"""
深入分析 PROPER 的参考面机制

关键问题：当 reference_surface = "SPHERI" 时，PROPER 存储的是什么？

根据 PROPER 手册和代码：
- PLANAR：wfarr 存储完整复振幅
- SPHERI：wfarr 存储相对于参考球面的残差？还是完整复振幅？

让我通过实验来验证。
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def test_reference_surface():
    """测试参考面机制"""
    
    print("=" * 70)
    print("测试 PROPER 参考面机制")
    print("=" * 70)
    
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
    
    # ========== PROPER 传播 ==========
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    
    print(f"\n传播前:")
    print(f"  z = {wfo.z * 1e3:.3f} mm")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
    
    proper.prop_propagate(wfo, dz)
    
    print(f"\n传播后:")
    print(f"  z = {wfo.z * 1e3:.3f} mm")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
    print(f"  dx = {wfo.dx * 1e6:.3f} μm")
    
    # 读取 PROPER 结果
    E_proper_amp = proper.prop_get_amplitude(wfo)
    E_proper_phase = proper.prop_get_phase(wfo)
    
    # 新坐标
    dx_new = wfo.dx
    x_new = (np.arange(n) - n // 2) * dx_new
    X_new, Y_new = np.meshgrid(x_new, x_new)
    r_sq_new = X_new**2 + Y_new**2
    
    # ========== 理论高斯光束 ==========
    w = w0 * np.sqrt(1 + (dz / z_R)**2)
    R = dz * (1 + (z_R / dz)**2)
    gouy = np.arctan(dz / z_R)
    
    print(f"\n理论高斯光束参数:")
    print(f"  w(z) = {w * 1e3:.3f} mm")
    print(f"  R(z) = {R * 1e3:.1f} mm")
    print(f"  Gouy = {gouy:.6f} rad")
    
    # 理论振幅
    theory_amp = (w0 / w) * np.exp(-r_sq_new / w**2)
    
    # 理论相位（完整，包含球面波前和 Gouy）
    theory_phase_full = -k * r_sq_new / (2 * R) - gouy
    
    # ========== 检查 PROPER 是否存储了参考球面 ==========
    print("\n" + "=" * 70)
    print("检查 PROPER 相位")
    print("=" * 70)
    
    center = n // 2
    
    # PROPER 相位（减去中心）
    proper_phase_rel = E_proper_phase - E_proper_phase[center, center]
    
    # 理论相位（减去中心）
    theory_phase_rel = theory_phase_full - theory_phase_full[center, center]
    
    # 如果 PROPER 存储的是完整复振幅，那么 proper_phase_rel 应该等于 theory_phase_rel
    
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    phase_diff = np.angle(np.exp(1j * (proper_phase_rel - theory_phase_rel)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n假设 PROPER 存储完整复振幅:")
    print(f"  相位误差 RMS: {phase_error:.6f} rad")
    
    # ========== 检查是否存储了相对于参考球面的残差 ==========
    # 如果 reference_surface = "SPHERI"，参考球面曲率半径是 z - z_w0
    R_ref = wfo.z - wfo.z_w0
    print(f"\n参考球面曲率半径 R_ref = z - z_w0 = {R_ref * 1e3:.1f} mm")
    print(f"理论曲率半径 R = {R * 1e3:.1f} mm")
    
    # 参考球面相位
    ref_phase = -k * r_sq_new / (2 * R_ref)
    ref_phase_rel = ref_phase - ref_phase[center, center]
    
    # 如果 PROPER 存储的是残差，那么：
    # proper_phase = 完整相位 - 参考球面相位
    # 所以：完整相位 = proper_phase + 参考球面相位
    
    reconstructed_phase = proper_phase_rel + ref_phase_rel
    
    phase_diff2 = np.angle(np.exp(1j * (reconstructed_phase - theory_phase_rel)))
    phase_error2 = np.sqrt(np.mean(phase_diff2[mask]**2))
    
    print(f"\n假设 PROPER 存储残差（需要加回参考球面）:")
    print(f"  相位误差 RMS: {phase_error2:.6f} rad")
    
    # ========== 检查 PROPER 相位的实际形状 ==========
    print("\n" + "=" * 70)
    print("分析 PROPER 相位的实际形状")
    print("=" * 70)
    
    # 沿 x 轴的相位剖面
    x_mm = x_new * 1e3
    proper_profile = proper_phase_rel[center, :]
    theory_profile = theory_phase_rel[center, :]
    ref_profile = ref_phase_rel[center, :]
    
    # 拟合 PROPER 相位的曲率
    # φ = a * r² + b
    # 在有效区域内拟合
    valid = mask[center, :]
    r_valid = x_new[valid]
    phase_valid = E_proper_phase[center, valid] - E_proper_phase[center, center]
    
    # 最小二乘拟合
    A = np.column_stack([r_valid**2, np.ones_like(r_valid)])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_valid, rcond=None)
    a_fit = coeffs[0]
    
    # 从拟合系数计算曲率半径
    # φ = -k * r² / (2R) => a = -k / (2R) => R = -k / (2a)
    R_fit = -k / (2 * a_fit)
    
    print(f"\n从 PROPER 相位拟合的曲率半径:")
    print(f"  R_fit = {R_fit * 1e3:.1f} mm")
    print(f"  理论 R = {R * 1e3:.1f} mm")
    print(f"  参考 R_ref = {R_ref * 1e3:.1f} mm")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.plot(x_mm, proper_profile, 'b-', label='PROPER')
    ax.plot(x_mm, theory_profile, 'r--', label='Theory')
    ax.plot(x_mm, ref_profile, 'g:', label='Ref sphere')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.set_title('Phase profiles (relative to center)')
    ax.set_xlim([-10, 10])
    
    ax = axes[0, 1]
    ax.plot(x_mm, proper_profile - theory_profile, 'b-')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase diff (rad)')
    ax.set_title('PROPER - Theory')
    ax.set_xlim([-10, 10])
    
    ax = axes[0, 2]
    ax.plot(x_mm, proper_profile - ref_profile, 'b-')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase diff (rad)')
    ax.set_title('PROPER - Ref sphere')
    ax.set_xlim([-10, 10])
    
    ax = axes[1, 0]
    im = ax.imshow(proper_phase_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(theory_phase_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(phase_diff * mask, cmap='RdBu', vmin=-2, vmax=2)
    ax.set_title('Phase diff (rad)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Reference Surface Analysis: R_fit={R_fit*1e3:.0f}mm, R_theory={R*1e3:.0f}mm, R_ref={R_ref*1e3:.0f}mm')
    plt.tight_layout()
    plt.savefig('analyze_proper_reference_surface.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: analyze_proper_reference_surface.png")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    test_reference_surface()
