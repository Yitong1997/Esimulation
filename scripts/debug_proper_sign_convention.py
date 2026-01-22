"""
调试 PROPER 的符号约定

问题：拟合曲率与预期残差曲率匹配，但重建后仍有误差。
可能原因：符号约定不一致
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def debug_sign():
    """调试符号约定"""
    
    print("=" * 70)
    print("调试 PROPER 符号约定")
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
    
    n = grid_size
    x_old = (np.arange(n) - n // 2) * dx_old
    X_old, Y_old = np.meshgrid(x_old, x_old)
    r_sq_old = X_old**2 + Y_old**2
    
    E_initial = np.exp(-r_sq_old / w0**2)
    
    # PROPER 传播
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    proper.prop_propagate(wfo, dz)
    
    E_proper_amp = proper.prop_get_amplitude(wfo)
    E_proper_phase = proper.prop_get_phase(wfo)
    
    dx_new = wfo.dx
    x_new = (np.arange(n) - n // 2) * dx_new
    X_new, Y_new = np.meshgrid(x_new, x_new)
    r_sq_new = X_new**2 + Y_new**2
    
    w = w0 * np.sqrt(1 + (dz / z_R)**2)
    R = dz * (1 + (z_R / dz)**2)
    gouy = np.arctan(dz / z_R)
    R_ref = wfo.z - wfo.z_w0
    
    center = n // 2
    mask = E_proper_amp > 0.01 * np.max(E_proper_amp)
    
    print(f"\n参数:")
    print(f"  R_ref = {R_ref * 1e3:.1f} mm")
    print(f"  R_theory = {R * 1e3:.1f} mm")
    
    # ========== 测试不同的符号组合 ==========
    print("\n" + "=" * 70)
    print("测试不同的符号组合")
    print("=" * 70)
    
    # PROPER 相位（相对于中心）
    proper_rel = E_proper_phase - E_proper_phase[center, center]
    
    # 测试所有可能的符号组合
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    for s1, s2 in signs:
        # 参考球面相位
        ref_phase = s1 * k * r_sq_new / (2 * R_ref)
        
        # 重建完整相位
        reconstructed = E_proper_phase + ref_phase
        reconstructed_rel = reconstructed - reconstructed[center, center]
        
        # 理论完整相位
        theory = s2 * k * r_sq_new / (2 * R)
        theory_rel = theory - theory[center, center]
        
        diff = np.angle(np.exp(1j * (reconstructed_rel - theory_rel)))
        error = np.sqrt(np.mean(diff[mask]**2))
        
        print(f"  ref_sign={s1:+d}, theory_sign={s2:+d}: error = {error:.6f} rad")
    
    # ========== 检查 prop_qphase 的符号 ==========
    print("\n" + "=" * 70)
    print("检查 prop_qphase 的符号")
    print("=" * 70)
    
    # prop_qphase 添加：exp(i*π/(λ*c) * r²)
    # 这等于：exp(i * k * r² / (2*c))
    # 其中 c = dz（传播距离）
    
    # 对于发散光束（从束腰向外传播），波前是凸的
    # 凸波前的相位应该是正的（边缘相位延迟）
    
    print(f"\nprop_qphase 添加的相位：exp(i * k * r² / (2*dz))")
    print(f"这是正的二次相位（边缘相位超前）")
    
    # ========== 检查高斯光束的相位约定 ==========
    print("\n" + "=" * 70)
    print("检查高斯光束相位约定")
    print("=" * 70)
    
    # 标准高斯光束公式（Siegman 约定）：
    # E(r,z) = (w0/w) * exp(-r²/w²) * exp(-i*k*z) * exp(-i*k*r²/(2*R)) * exp(i*ψ)
    #
    # 其中：
    # - exp(-i*k*z) 是平面波传播相位
    # - exp(-i*k*r²/(2*R)) 是球面波前相位（R > 0 时边缘相位延迟）
    # - exp(i*ψ) 是 Gouy 相位
    
    print(f"\n标准高斯光束相位（Siegman 约定）：")
    print(f"  φ = -k*z - k*r²/(2*R) + ψ")
    print(f"  其中 R > 0 表示发散光束")
    
    # PROPER 可能使用不同的约定
    # 让我检查 PROPER 的实际行为
    
    # ========== 直接比较相位形状 ==========
    print("\n" + "=" * 70)
    print("直接比较相位形状")
    print("=" * 70)
    
    # 在边缘处，PROPER 相位是正还是负？
    edge_idx = center + 3  # 距离中心 3 个像素
    proper_edge = E_proper_phase[center, edge_idx]
    proper_center = E_proper_phase[center, center]
    
    print(f"\nPROPER 相位:")
    print(f"  中心: {proper_center:.6f} rad")
    print(f"  边缘 (x={x_new[edge_idx]*1e3:.2f}mm): {proper_edge:.6f} rad")
    print(f"  差值 (边缘-中心): {proper_edge - proper_center:.6f} rad")
    
    # 理论预期（残差）
    r_edge = x_new[edge_idx]
    residual_expected = k * r_edge**2 / 2 * (1/R_ref - 1/R)
    print(f"\n预期残差相位差: {residual_expected:.6f} rad")
    
    # 如果 PROPER 存储的是残差，且 R > R_ref，则 1/R_ref - 1/R > 0
    # 所以残差相位应该是正的（边缘 > 中心）
    
    print(f"\n1/R_ref - 1/R = {1/R_ref - 1/R:.6e}")
    print(f"预期：边缘相位 > 中心相位" if 1/R_ref - 1/R > 0 else "预期：边缘相位 < 中心相位")
    print(f"实际：边缘相位 {'>' if proper_edge > proper_center else '<'} 中心相位")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    debug_sign()
