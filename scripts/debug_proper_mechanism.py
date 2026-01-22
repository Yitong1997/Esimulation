"""
深入调试 PROPER 的内部机制

目标：理解 PROPER 如何存储和处理相位
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def debug_proper_internal():
    """调试 PROPER 内部机制"""
    
    print("=" * 70)
    print("调试 PROPER 内部机制")
    print("=" * 70)
    
    # 参数设置
    wavelength = 632.8e-9  # m
    w0 = 1e-3  # 束腰半径 1 mm
    beam_diameter = 2 * w0
    grid_size = 256
    beam_diam_fraction = 0.3
    
    # 计算瑞利距离
    z_R = np.pi * w0**2 / wavelength
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  束腰半径 w0: {w0 * 1e3:.3f} mm")
    print(f"  瑞利距离 z_R: {z_R * 1e3:.3f} mm")
    
    # ========== 测试 1：标准 prop_begin 初始化 ==========
    print("\n" + "=" * 70)
    print("测试 1：标准 prop_begin 初始化")
    print("=" * 70)
    
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    print(f"\nprop_begin 后的 wfo 属性:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  wfo.w0: {wfo.w0 * 1e3:.6f} mm")
    print(f"  wfo.z_Rayleigh: {wfo.z_Rayleigh * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  wfo.reference_surface: {wfo.reference_surface}")
    print(f"  wfo.beam_type_old: {wfo.beam_type_old}")
    print(f"  wfo.R_beam: {wfo.R_beam}")
    print(f"  wfo.R_beam_inf: {wfo.R_beam_inf}")
    
    # 检查初始 wfarr
    amp = proper.prop_get_amplitude(wfo)
    phase = proper.prop_get_phase(wfo)
    
    print(f"\n初始 wfarr:")
    print(f"  振幅范围: [{np.min(amp):.6f}, {np.max(amp):.6f}]")
    print(f"  相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  wfarr 是否全为 1: {np.allclose(wfo.wfarr, 1.0)}")
    
    # ========== 测试 2：传播后的状态 ==========
    print("\n" + "=" * 70)
    print("测试 2：传播 0.5 m 后的状态")
    print("=" * 70)
    
    propagation_distance = 0.5  # m
    
    # 保存传播前的状态
    wfarr_before = wfo.wfarr.copy()
    
    proper.prop_propagate(wfo, propagation_distance)
    
    print(f"\n传播后的 wfo 属性:")
    print(f"  wfo.z: {wfo.z * 1e3:.3f} mm")
    print(f"  wfo.z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  wfo.dx: {wfo.dx * 1e6:.3f} μm")
    print(f"  wfo.reference_surface: {wfo.reference_surface}")
    print(f"  wfo.beam_type_old: {wfo.beam_type_old}")
    
    amp_after = proper.prop_get_amplitude(wfo)
    phase_after = proper.prop_get_phase(wfo)
    
    print(f"\n传播后 wfarr:")
    print(f"  振幅范围: [{np.min(amp_after):.6f}, {np.max(amp_after):.6f}]")
    print(f"  相位范围: [{np.min(phase_after):.6f}, {np.max(phase_after):.6f}] rad")
    
    # ========== 测试 3：理论高斯光束比较 ==========
    print("\n" + "=" * 70)
    print("测试 3：与理论高斯光束比较")
    print("=" * 70)
    
    z = propagation_distance
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)  # 严格公式
    gouy = np.arctan(z / z_R)
    
    print(f"\n理论高斯光束参数 (z = {z*1e3:.1f} mm):")
    print(f"  光斑半径 w: {w * 1e3:.6f} mm")
    print(f"  曲率半径 R: {R * 1e3:.3f} mm")
    print(f"  Gouy 相位: {gouy:.6f} rad")
    
    # 创建坐标网格
    n = grid_size
    sampling = wfo.dx
    x = (np.arange(n) - n // 2) * sampling
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    r = np.sqrt(r_sq)
    
    k = 2 * np.pi / wavelength
    
    # 理论振幅
    theory_amp = (w0 / w) * np.exp(-r_sq / w**2)
    
    # 理论相位（相对于主光线）
    theory_phase = k * r_sq / (2 * R)
    
    # 比较中心区域
    center = n // 2
    half_width = 50
    
    print(f"\n中心区域比较 (±{half_width} 像素):")
    
    # 振幅比较
    amp_center = amp_after[center-half_width:center+half_width, center-half_width:center+half_width]
    theory_amp_center = theory_amp[center-half_width:center+half_width, center-half_width:center+half_width]
    
    amp_ratio = amp_center / theory_amp_center
    print(f"  振幅比值范围: [{np.min(amp_ratio):.6f}, {np.max(amp_ratio):.6f}]")
    print(f"  振幅比值均值: {np.mean(amp_ratio):.6f}")
    
    # 相位比较
    phase_center = phase_after[center-half_width:center+half_width, center-half_width:center+half_width]
    theory_phase_center = theory_phase[center-half_width:center+half_width, center-half_width:center+half_width]
    
    # 减去中心相位（主光线处）
    phase_center_rel = phase_center - phase_center[half_width, half_width]
    theory_phase_center_rel = theory_phase_center - theory_phase_center[half_width, half_width]
    
    phase_diff = phase_center_rel - theory_phase_center_rel
    print(f"  相位差范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad")
    print(f"  相位差 RMS: {np.std(phase_diff):.6f} rad")
    
    # ========== 测试 4：检查 PROPER 是否存储了参考相位 ==========
    print("\n" + "=" * 70)
    print("测试 4：检查 PROPER 的参考相位机制")
    print("=" * 70)
    
    # 重新初始化
    wfo2 = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    # 手动添加一个已知的球面相位
    test_R = 10.0  # 10 m 曲率半径
    test_phase = k * r_sq / (2 * test_R)  # 球面相位
    
    print(f"\n添加测试球面相位 (R = {test_R} m):")
    print(f"  相位范围: [{np.min(test_phase):.6f}, {np.max(test_phase):.6f}] rad")
    
    # 使用 prop_add_phase 添加（单位是米）
    test_opd = r_sq / (2 * test_R)  # OPD in meters
    proper.prop_add_phase(wfo2, test_opd)
    
    # 读取相位
    phase_read = proper.prop_get_phase(wfo2)
    
    print(f"\n读取的相位:")
    print(f"  相位范围: [{np.min(phase_read):.6f}, {np.max(phase_read):.6f}] rad")
    
    # 比较
    phase_diff2 = phase_read - test_phase
    print(f"  与输入相位差: [{np.min(phase_diff2):.6f}, {np.max(phase_diff2):.6f}] rad")
    print(f"  差值 RMS: {np.std(phase_diff2):.6f} rad")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：传播后的结果
    ax = axes[0, 0]
    im = ax.imshow(amp_after, cmap='hot')
    ax.set_title('PROPER Amplitude after propagation')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(phase_after, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('PROPER Phase after propagation')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(theory_phase, cmap='twilight')
    ax.set_title('Theoretical Phase (spherical)')
    plt.colorbar(im, ax=ax)
    
    # 第二行：比较
    ax = axes[1, 0]
    im = ax.imshow(theory_amp, cmap='hot')
    ax.set_title('Theoretical Amplitude')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    mask = theory_amp > 0.01 * np.max(theory_amp)
    diff = (amp_after - theory_amp) * mask
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title('Amplitude Difference')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    # 相位差（考虑包裹）
    phase_diff_full = np.angle(np.exp(1j * (phase_after - theory_phase)))
    im = ax.imshow(phase_diff_full * mask, cmap='RdBu', vmin=-0.2, vmax=0.2)
    ax.set_title('Phase Difference (rad)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('PROPER Internal Mechanism Debug')
    plt.tight_layout()
    plt.savefig('debug_proper_mechanism.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved to: debug_proper_mechanism.png")


def debug_proper_reference_surface():
    """调试 PROPER 的参考面机制"""
    
    print("\n" + "=" * 70)
    print("调试 PROPER 参考面机制")
    print("=" * 70)
    
    # 参数设置
    wavelength = 632.8e-9  # m
    w0 = 1e-3  # 束腰半径 1 mm
    beam_diameter = 2 * w0
    grid_size = 256
    beam_diam_fraction = 0.3
    
    z_R = np.pi * w0**2 / wavelength
    
    # 初始化
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    print(f"\n初始状态:")
    print(f"  reference_surface: {wfo.reference_surface}")
    print(f"  z: {wfo.z}")
    print(f"  z_w0: {wfo.z_w0}")
    
    # 传播到远场
    far_field_distance = 10 * z_R
    proper.prop_propagate(wfo, far_field_distance)
    
    print(f"\n传播到远场 (z = {far_field_distance/z_R:.1f} z_R) 后:")
    print(f"  reference_surface: {wfo.reference_surface}")
    print(f"  z: {wfo.z * 1e3:.3f} mm")
    print(f"  z_w0: {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  z - z_w0: {(wfo.z - wfo.z_w0) * 1e3:.3f} mm")
    
    # 如果是 SPHERI 参考面，PROPER 内部存储的是相对于参考球面的残差
    # 参考球面的曲率半径是 z - z_w0
    
    if wfo.reference_surface == "SPHERI":
        R_ref = wfo.z - wfo.z_w0
        print(f"\n参考球面曲率半径 R_ref = z - z_w0 = {R_ref * 1e3:.3f} mm")
        
        # 理论高斯光束曲率半径（严格公式）
        z = wfo.z
        R_theory = z * (1 + (z_R / z)**2)
        print(f"理论曲率半径 R_theory = {R_theory * 1e3:.3f} mm")
        print(f"差异: {(R_ref - R_theory) * 1e3:.3f} mm")
        print(f"相对差异: {(R_ref - R_theory) / R_theory * 100:.2f}%")
    
    # 读取相位
    phase = proper.prop_get_phase(wfo)
    amp = proper.prop_get_amplitude(wfo)
    
    print(f"\n读取的相位:")
    print(f"  范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    
    # 如果 PROPER 存储的是残差，那么读取的相位应该接近零（对于理想高斯光束）
    mask = amp > 0.01 * np.max(amp)
    print(f"  有效区域 RMS: {np.std(phase[mask]):.6f} rad")


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    debug_proper_internal()
    debug_proper_reference_surface()
