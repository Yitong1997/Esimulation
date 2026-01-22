"""
调试 PROPER 的 Gouy 相位处理

问题：重建后仍有 ~1 rad 的误差，这接近 Gouy 相位的值 (1.37 rad)

可能的原因：
1. Gouy 相位没有被正确包含
2. 参考球面的定义不同
3. 其他数值因素
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def debug_gouy_phase():
    """调试 Gouy 相位"""
    
    print("=" * 70)
    print("调试 PROPER Gouy 相位")
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
    
    print(f"\n参数:")
    print(f"  Gouy 相位 = {gouy:.6f} rad")
    print(f"  理论曲率 R = {R * 1e3:.1f} mm")
    print(f"  参考曲率 R_ref = {R_ref * 1e3:.1f} mm")
    
    center = n // 2
    mask = E_proper_amp > 0.01 * np.max(E_proper_amp)
    
    # ========== 测试不同的理论公式 ==========
    print("\n" + "=" * 70)
    print("测试不同的理论公式")
    print("=" * 70)
    
    # 参考球面相位
    ref_phase = -k * r_sq_new / (2 * R_ref)
    ref_phase_rel = ref_phase - ref_phase[center, center]
    
    # PROPER 相位
    proper_phase_rel = E_proper_phase - E_proper_phase[center, center]
    
    # 重建完整相位
    reconstructed = proper_phase_rel + ref_phase_rel
    
    # 测试 1：理论相位包含 Gouy
    theory1 = -k * r_sq_new / (2 * R) - gouy
    theory1_rel = theory1 - theory1[center, center]
    diff1 = np.angle(np.exp(1j * (reconstructed - theory1_rel)))
    error1 = np.sqrt(np.mean(diff1[mask]**2))
    print(f"\n理论 1 (包含 Gouy): 误差 = {error1:.6f} rad")
    
    # 测试 2：理论相位不包含 Gouy
    theory2 = -k * r_sq_new / (2 * R)
    theory2_rel = theory2 - theory2[center, center]
    diff2 = np.angle(np.exp(1j * (reconstructed - theory2_rel)))
    error2 = np.sqrt(np.mean(diff2[mask]**2))
    print(f"理论 2 (不含 Gouy): 误差 = {error2:.6f} rad")
    
    # 测试 3：使用 R_ref 作为曲率
    theory3 = -k * r_sq_new / (2 * R_ref)
    theory3_rel = theory3 - theory3[center, center]
    diff3 = np.angle(np.exp(1j * (reconstructed - theory3_rel)))
    error3 = np.sqrt(np.mean(diff3[mask]**2))
    print(f"理论 3 (使用 R_ref): 误差 = {error3:.6f} rad")
    
    # 测试 4：直接比较 PROPER 相位和残差理论
    # 残差 = 完整相位 - 参考相位 = -k*r²/(2R) - gouy - (-k*r²/(2*R_ref))
    residual_theory = -k * r_sq_new / (2 * R) - gouy + k * r_sq_new / (2 * R_ref)
    residual_theory_rel = residual_theory - residual_theory[center, center]
    diff4 = np.angle(np.exp(1j * (proper_phase_rel - residual_theory_rel)))
    error4 = np.sqrt(np.mean(diff4[mask]**2))
    print(f"理论 4 (残差含 Gouy): 误差 = {error4:.6f} rad")
    
    # 测试 5：残差不含 Gouy
    residual_theory5 = -k * r_sq_new / (2 * R) + k * r_sq_new / (2 * R_ref)
    residual_theory5_rel = residual_theory5 - residual_theory5[center, center]
    diff5 = np.angle(np.exp(1j * (proper_phase_rel - residual_theory5_rel)))
    error5 = np.sqrt(np.mean(diff5[mask]**2))
    print(f"理论 5 (残差不含 Gouy): 误差 = {error5:.6f} rad")
    
    # ========== 检查中心相位 ==========
    print("\n" + "=" * 70)
    print("检查中心相位")
    print("=" * 70)
    
    print(f"\nPROPER 中心相位: {E_proper_phase[center, center]:.6f} rad")
    print(f"Gouy 相位: {gouy:.6f} rad")
    print(f"差值: {E_proper_phase[center, center] + gouy:.6f} rad")
    
    # 如果 PROPER 中心相位 ≈ -gouy，说明 PROPER 包含了 Gouy 相位
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x_mm = x_new * 1e3
    
    ax = axes[0, 0]
    ax.plot(x_mm, proper_phase_rel[center, :], 'b-', label='PROPER')
    ax.plot(x_mm, residual_theory_rel[center, :], 'r--', label='Residual (with Gouy)')
    ax.plot(x_mm, residual_theory5_rel[center, :], 'g:', label='Residual (no Gouy)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.set_title('PROPER vs Residual theories')
    
    ax = axes[0, 1]
    ax.plot(x_mm, diff4[center, :], 'r-', label='With Gouy')
    ax.plot(x_mm, diff5[center, :], 'g-', label='No Gouy')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase diff (rad)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.set_title('Phase differences')
    
    ax = axes[0, 2]
    ax.plot(x_mm, reconstructed[center, :], 'b-', label='Reconstructed')
    ax.plot(x_mm, theory1_rel[center, :], 'r--', label='Theory (with Gouy)')
    ax.plot(x_mm, theory2_rel[center, :], 'g:', label='Theory (no Gouy)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.set_title('Reconstructed vs Theory')
    
    ax = axes[1, 0]
    im = ax.imshow(diff4 * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'PROPER - Residual(Gouy): err={error4:.4f}')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(diff5 * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'PROPER - Residual(no Gouy): err={error5:.4f}')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(diff1 * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'Reconstructed - Theory(Gouy): err={error1:.4f}')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Gouy phase analysis: gouy={gouy:.4f} rad')
    plt.tight_layout()
    plt.savefig('debug_proper_gouy_phase.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: debug_proper_gouy_phase.png")
    
    # 找出最小误差
    errors = [error1, error2, error3, error4, error5]
    labels = ['Theory(Gouy)', 'Theory(no Gouy)', 'Theory(R_ref)', 'Residual(Gouy)', 'Residual(no Gouy)']
    min_idx = np.argmin(errors)
    print(f"\n最小误差: {labels[min_idx]} = {errors[min_idx]:.6f} rad")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    debug_gouy_phase()
