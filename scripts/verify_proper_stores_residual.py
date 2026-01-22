"""
验证 PROPER 在 SPHERI 参考面时存储的是残差

关键发现：
- PROPER 相位拟合的曲率半径 R_fit ≈ 161000 mm
- 理论曲率半径 R = 6454 mm
- 参考球面曲率半径 R_ref = 6206 mm

如果 PROPER 存储的是残差：
  残差相位 = 完整相位 - 参考球面相位
           = -k*r²/(2R) - (-k*r²/(2*R_ref))
           = -k*r²/(2R) + k*r²/(2*R_ref)
           = k*r²/2 * (1/R_ref - 1/R)

残差的等效曲率半径：
  1/R_residual = 1/R_ref - 1/R
  R_residual = R * R_ref / (R - R_ref)
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def verify_residual_storage():
    """验证 PROPER 存储残差"""
    
    print("=" * 70)
    print("验证 PROPER 存储残差")
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
    print(f"  理论曲率 R = {R * 1e3:.1f} mm")
    print(f"  参考曲率 R_ref = {R_ref * 1e3:.1f} mm")
    
    # 计算预期的残差曲率
    R_residual = R * R_ref / (R - R_ref)
    print(f"  预期残差曲率 R_residual = {R_residual * 1e3:.1f} mm")
    
    # ========== 验证 ==========
    center = n // 2
    
    # 理论完整相位（相对于中心）
    theory_phase_full = -k * r_sq_new / (2 * R) - gouy
    theory_phase_rel = theory_phase_full - theory_phase_full[center, center]
    
    # 参考球面相位（相对于中心）
    ref_phase = -k * r_sq_new / (2 * R_ref)
    ref_phase_rel = ref_phase - ref_phase[center, center]
    
    # 预期残差相位（相对于中心）
    expected_residual = theory_phase_rel - ref_phase_rel
    
    # PROPER 相位（相对于中心）
    proper_phase_rel = E_proper_phase - E_proper_phase[center, center]
    
    # 比较
    mask = E_proper_amp > 0.01 * np.max(E_proper_amp)
    
    # 方案 1：PROPER 存储完整相位
    diff1 = np.angle(np.exp(1j * (proper_phase_rel - theory_phase_rel)))
    error1 = np.sqrt(np.mean(diff1[mask]**2))
    
    # 方案 2：PROPER 存储残差
    diff2 = np.angle(np.exp(1j * (proper_phase_rel - expected_residual)))
    error2 = np.sqrt(np.mean(diff2[mask]**2))
    
    print(f"\n验证结果:")
    print(f"  假设存储完整相位，误差 RMS: {error1:.6f} rad")
    print(f"  假设存储残差，误差 RMS: {error2:.6f} rad")
    
    if error2 < error1:
        print(f"\n结论：PROPER 在 SPHERI 参考面时存储的是残差！")
    else:
        print(f"\n结论：PROPER 存储的是完整相位")
    
    # ========== 重建完整相位 ==========
    print("\n" + "=" * 70)
    print("重建完整相位")
    print("=" * 70)
    
    # 完整相位 = PROPER 相位 + 参考球面相位
    reconstructed_phase = proper_phase_rel + ref_phase_rel
    
    diff3 = np.angle(np.exp(1j * (reconstructed_phase - theory_phase_rel)))
    error3 = np.sqrt(np.mean(diff3[mask]**2))
    
    print(f"  重建后与理论的误差 RMS: {error3:.6f} rad")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    ax = axes[0, 0]
    im = ax.imshow(proper_phase_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(expected_residual, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Expected residual')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(ref_phase_rel, cmap='twilight')
    ax.set_title('Reference sphere phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    im = ax.imshow(diff2 * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('PROPER - Expected residual')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    im = ax.imshow(reconstructed_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Reconstructed full phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(theory_phase_rel, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Theory full phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(diff3 * mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Reconstructed - Theory')
    plt.colorbar(im, ax=ax)
    
    # 剖面图
    ax = axes[1, 3]
    x_mm = x_new * 1e3
    ax.plot(x_mm, proper_phase_rel[center, :], 'b-', label='PROPER')
    ax.plot(x_mm, expected_residual[center, :], 'r--', label='Expected residual')
    ax.plot(x_mm, reconstructed_phase[center, :], 'g:', label='Reconstructed')
    ax.plot(x_mm, theory_phase_rel[center, :], 'k-.', label='Theory')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.legend(fontsize=8)
    ax.set_xlim([-10, 10])
    ax.set_title('Phase profiles')
    
    plt.suptitle(f'PROPER stores residual: error={error2:.4f} rad')
    plt.tight_layout()
    plt.savefig('verify_proper_stores_residual.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: verify_proper_stores_residual.png")
    
    return error2 < 0.1


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    result = verify_residual_storage()
    print(f"\n最终结果: {'PASS' if result else 'FAIL'}")
