"""
最终分析 PROPER 的存储机制

基于之前的测试，让我系统地分析 PROPER 在远场传播后存储的内容。
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def final_analysis():
    """最终分析"""
    
    print("=" * 70)
    print("最终分析 PROPER 存储机制")
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
    
    # PROPER 传播
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    proper.prop_propagate(wfo, dz)
    
    # 读取结果
    E_proper_amp = proper.prop_get_amplitude(wfo)
    E_proper_phase = proper.prop_get_phase(wfo)
    
    # 新坐标
    dx_new = wfo.dx
    x_new = (np.arange(n) - n // 2) * dx_new
    X_new, Y_new = np.meshgrid(x_new, x_new)
    r_sq_new = X_new**2 + Y_new**2
    
    # 理论参数
    w = w0 * np.sqrt(1 + (dz / z_R)**2)
    R = dz * (1 + (z_R / dz)**2)
    gouy = np.arctan(dz / z_R)
    R_ref = wfo.z - wfo.z_w0
    
    center = n // 2
    mask = E_proper_amp > 0.01 * np.max(E_proper_amp)
    
    print(f"\n关键参数:")
    print(f"  z = {wfo.z * 1e3:.1f} mm")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.1f} mm")
    print(f"  R_ref = z - z_w0 = {R_ref * 1e3:.1f} mm")
    print(f"  R_theory = {R * 1e3:.1f} mm")
    print(f"  Gouy = {gouy:.6f} rad")
    
    # ========== 分析 PROPER 相位的结构 ==========
    print("\n" + "=" * 70)
    print("分析 PROPER 相位结构")
    print("=" * 70)
    
    # 拟合 PROPER 相位：φ = a*r² + b
    r_valid = np.sqrt(r_sq_new[mask])
    phase_valid = E_proper_phase[mask]
    
    # 使用中心相位作为参考
    phase_valid_rel = phase_valid - E_proper_phase[center, center]
    r_sq_valid = r_sq_new[mask]
    
    # 线性拟合 phase vs r²
    A = np.column_stack([r_sq_valid, np.ones_like(r_sq_valid)])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_valid_rel, rcond=None)
    a_fit = coeffs[0]
    b_fit = coeffs[1]
    
    # 从 a 计算等效曲率半径
    # φ = -k*r²/(2R) => a = -k/(2R) => R = -k/(2a)
    R_fit = -k / (2 * a_fit)
    
    print(f"\n拟合结果:")
    print(f"  a = {a_fit:.6e} rad/m²")
    print(f"  b = {b_fit:.6f} rad")
    print(f"  R_fit = {R_fit * 1e3:.1f} mm")
    
    # 比较
    print(f"\n曲率半径比较:")
    print(f"  R_fit = {R_fit * 1e3:.1f} mm")
    print(f"  R_ref = {R_ref * 1e3:.1f} mm")
    print(f"  R_theory = {R * 1e3:.1f} mm")
    print(f"  R_fit / R_ref = {R_fit / R_ref:.6f}")
    
    # ========== 检查 PROPER 是否存储了相对于 R_ref 的残差 ==========
    print("\n" + "=" * 70)
    print("验证：PROPER 存储相对于 R_ref 的残差")
    print("=" * 70)
    
    # 如果 PROPER 存储的是相对于 R_ref 的残差：
    # PROPER_phase = full_phase - ref_phase
    #              = -k*r²/(2R) - gouy - (-k*r²/(2*R_ref))
    #              = k*r²/2 * (1/R_ref - 1/R) - gouy
    
    # 残差的等效曲率：
    # 1/R_residual = 1/R_ref - 1/R
    R_residual = R * R_ref / (R - R_ref)
    
    print(f"\n预期残差曲率 R_residual = {R_residual * 1e3:.1f} mm")
    print(f"拟合曲率 R_fit = {R_fit * 1e3:.1f} mm")
    print(f"比值 R_fit / R_residual = {R_fit / R_residual:.6f}")
    
    # 如果 R_fit ≈ R_residual，说明 PROPER 存储的是残差（不含 Gouy）
    # 如果 R_fit ≈ ∞，说明 PROPER 存储的是残差（含 Gouy 作为常数偏移）
    
    # ========== 直接验证 ==========
    print("\n" + "=" * 70)
    print("直接验证")
    print("=" * 70)
    
    # 假设 PROPER 存储：残差相位 = k*r²/2 * (1/R_ref - 1/R) + 常数
    # 其中常数可能包含 Gouy 相位
    
    residual_phase = k * r_sq_new / 2 * (1/R_ref - 1/R)
    residual_phase_rel = residual_phase - residual_phase[center, center]
    
    proper_phase_rel = E_proper_phase - E_proper_phase[center, center]
    
    diff = np.angle(np.exp(1j * (proper_phase_rel - residual_phase_rel)))
    error = np.sqrt(np.mean(diff[mask]**2))
    
    print(f"\n假设 PROPER = 残差（不含 Gouy 常数）:")
    print(f"  误差 RMS = {error:.6f} rad")
    
    # 检查常数偏移
    const_offset = np.mean((E_proper_phase[mask] - residual_phase[mask]))
    print(f"  常数偏移 = {const_offset:.6f} rad")
    print(f"  -Gouy = {-gouy:.6f} rad")
    print(f"  差值 = {const_offset + gouy:.6f} rad")
    
    # ========== 最终结论 ==========
    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)
    
    # 完整验证：PROPER_phase = k*r²/2 * (1/R_ref - 1/R) + const
    # 其中 const 是某个常数（可能与 Gouy 相关）
    
    # 重建完整相位
    # full_phase = PROPER_phase + ref_phase
    #            = k*r²/2 * (1/R_ref - 1/R) + const + (-k*r²/(2*R_ref))
    #            = -k*r²/(2*R) + const
    
    ref_phase = -k * r_sq_new / (2 * R_ref)
    reconstructed = E_proper_phase + ref_phase
    
    # 理论完整相位（不含 Gouy，因为 Gouy 是常数）
    theory_full = -k * r_sq_new / (2 * R)
    
    # 比较（减去中心）
    reconstructed_rel = reconstructed - reconstructed[center, center]
    theory_full_rel = theory_full - theory_full[center, center]
    
    diff_final = np.angle(np.exp(1j * (reconstructed_rel - theory_full_rel)))
    error_final = np.sqrt(np.mean(diff_final[mask]**2))
    
    print(f"\n重建完整相位（减去中心后）与理论比较:")
    print(f"  误差 RMS = {error_final:.6f} rad")
    
    if error_final < 0.01:
        print(f"\n✓ 验证成功！")
        print(f"  PROPER 在 SPHERI 参考面时存储的是相对于参考球面的残差")
        print(f"  参考球面曲率半径 = z - z_w0")
        print(f"  重建完整相位 = PROPER_phase + (-k*r²/(2*R_ref))")
    else:
        print(f"\n✗ 验证失败，误差 = {error_final:.6f} rad")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x_mm = x_new * 1e3
    
    ax = axes[0, 0]
    ax.plot(x_mm, E_proper_phase[center, :], 'b-', label='PROPER')
    ax.plot(x_mm, residual_phase[center, :], 'r--', label='Expected residual')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.set_title('PROPER phase vs Expected residual')
    
    ax = axes[0, 1]
    ax.plot(x_mm, reconstructed_rel[center, :], 'b-', label='Reconstructed')
    ax.plot(x_mm, theory_full_rel[center, :], 'r--', label='Theory')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.set_title('Reconstructed vs Theory (rel to center)')
    
    ax = axes[0, 2]
    ax.plot(x_mm, diff_final[center, :], 'b-')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase diff (rad)')
    ax.set_xlim([-10, 10])
    ax.set_title(f'Difference: RMS = {error_final:.4f} rad')
    
    ax = axes[1, 0]
    im = ax.imshow(E_proper_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(reconstructed_rel, cmap='twilight')
    ax.set_title('Reconstructed (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(diff_final * mask, cmap='RdBu', vmin=-0.1, vmax=0.1)
    ax.set_title('Difference')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Final Analysis: error = {error_final:.4f} rad')
    plt.tight_layout()
    plt.savefig('final_proper_analysis.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: final_proper_analysis.png")
    
    return error_final < 0.01


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    result = final_analysis()
    print(f"\n最终结果: {'PASS' if result else 'FAIL'}")
