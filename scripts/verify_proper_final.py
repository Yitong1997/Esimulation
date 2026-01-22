"""
最终验证 PROPER 的正确使用方式

本脚本验证：
1. StateConverter 的 PROPER 相位转换逻辑是否正确
2. 近场（PLANAR）和远场（SPHERI）的处理是否一致
3. 边界情况（z = z_R）的处理是否正确

关键发现：
1. PROPER 在 SPHERI 参考面时存储的是残差
2. 参考球面相位使用正号：+k*r²/(2*R_ref)
3. 理论完整相位也使用正号：+k*r²/(2*R)

正确的重建公式：
  完整相位 = PROPER相位 + k*r²/(2*R_ref)
"""

import numpy as np
import matplotlib.pyplot as plt
import proper
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.data_models import GridSampling, PilotBeamParams


def gaussian_beam_params(w0, wavelength, z):
    """计算高斯光束参数（理论值）
    
    参数:
        w0: 束腰半径 (m)
        wavelength: 波长 (m)
        z: 传播距离 (m)
    
    返回:
        (w, R, gouy, z_R) - 光斑大小、曲率半径、Gouy相位、瑞利长度
    """
    z_R = np.pi * w0**2 / wavelength
    
    if abs(z) < 1e-12:
        return w0, np.inf, 0.0, z_R
    
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)
    gouy = np.arctan(z / z_R)
    
    return w, R, gouy, z_R


def create_initial_gaussian(w0, grid_size, dx):
    """创建初始高斯光束（在束腰处）
    
    参数:
        w0: 束腰半径 (m)
        grid_size: 网格大小
        dx: 采样间隔 (m)
    
    返回:
        复振幅数组
    """
    n = grid_size
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    return np.exp(-r_sq / w0**2)


def verify_near_field(wavelength, w0, grid_size, beam_diam_fraction, converter):
    """验证近场传播（PLANAR 参考面）
    
    参数:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        grid_size: 网格大小
        beam_diam_fraction: 光束直径占网格比例
        converter: StateConverter 实例
    
    返回:
        (pass, amp_error, phase_error, results_dict)
    """
    print("\n" + "=" * 70)
    print("测试 1: 近场传播 (PLANAR 参考面)")
    print("=" * 70)
    
    z_R = np.pi * w0**2 / wavelength
    dz = 0.1 * z_R  # 传播 0.1 倍瑞利距离
    
    # 创建初始波前
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    dx_initial = wfo.dx
    E_initial = create_initial_gaussian(w0, grid_size, dx_initial)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    
    # 传播
    proper.prop_propagate(wfo, dz)
    
    print(f"\n传播距离: {dz * 1e3:.1f} mm ({dz/z_R:.2f} z_R)")
    print(f"参考面类型: {wfo.reference_surface}")
    
    # 使用 StateConverter 提取振幅和相位
    grid_sampling = GridSampling.from_proper(wfo)
    amplitude, phase = converter.proper_to_amplitude_phase(wfo, grid_sampling)
    
    # 理论值
    n = grid_size
    dx = wfo.dx
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    w_theory, R_theory, gouy_theory, _ = gaussian_beam_params(w0, wavelength, dz)
    
    theory_amp = (w0 / w_theory) * np.exp(-r_sq / w_theory**2)
    theory_phase = 2 * np.pi / wavelength * r_sq / (2 * R_theory) - gouy_theory
    
    # 比较（减去中心值）
    center = n // 2
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    phase_rel = phase - phase[center, center]
    theory_phase_rel = theory_phase - theory_phase[center, center]
    
    amp_error = np.sqrt(np.mean((amplitude[mask] - theory_amp[mask])**2))
    phase_diff = np.angle(np.exp(1j * (phase_rel - theory_phase_rel)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n近场结果:")
    print(f"  振幅误差 RMS: {amp_error:.6f}")
    print(f"  相位误差 RMS: {phase_error:.6f} rad")
    
    passed = amp_error < 0.001 and phase_error < 0.01
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    return passed, amp_error, phase_error, {
        'amplitude': amplitude,
        'phase_rel': phase_rel,
        'theory_amp': theory_amp,
        'theory_phase_rel': theory_phase_rel,
        'phase_diff': phase_diff,
        'mask': mask,
    }


def verify_far_field(wavelength, w0, grid_size, beam_diam_fraction, converter):
    """验证远场传播（SPHERI 参考面）
    
    参数:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        grid_size: 网格大小
        beam_diam_fraction: 光束直径占网格比例
        converter: StateConverter 实例
    
    返回:
        (pass, amp_error, phase_error, results_dict)
    """
    print("\n" + "=" * 70)
    print("测试 2: 远场传播 (SPHERI 参考面)")
    print("=" * 70)
    
    z_R = np.pi * w0**2 / wavelength
    dz = 5 * z_R  # 传播 5 倍瑞利距离
    
    # 创建初始波前
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    dx_initial = wfo.dx
    E_initial = create_initial_gaussian(w0, grid_size, dx_initial)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    
    # 传播
    proper.prop_propagate(wfo, dz)
    
    print(f"\n传播距离: {dz * 1e3:.1f} mm ({dz/z_R:.1f} z_R)")
    print(f"参考面类型: {wfo.reference_surface}")
    print(f"R_ref = z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.1f} mm")
    
    # 使用 StateConverter 提取振幅和相位
    grid_sampling = GridSampling.from_proper(wfo)
    amplitude, phase = converter.proper_to_amplitude_phase(wfo, grid_sampling)
    
    # 理论值
    n = grid_size
    dx = wfo.dx
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    w_theory, R_theory, gouy_theory, _ = gaussian_beam_params(w0, wavelength, dz)
    
    print(f"理论 R = {R_theory * 1e3:.1f} mm")
    
    theory_amp = (w0 / w_theory) * np.exp(-r_sq / w_theory**2)
    theory_phase = 2 * np.pi / wavelength * r_sq / (2 * R_theory) - gouy_theory
    
    # 比较（减去中心值）
    center = n // 2
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    phase_rel = phase - phase[center, center]
    theory_phase_rel = theory_phase - theory_phase[center, center]
    
    # 振幅需要归一化（FFT 缩放）
    scale = np.max(amplitude) / np.max(theory_amp)
    amp_normalized = amplitude / scale
    
    amp_error = np.sqrt(np.mean((amp_normalized[mask] - theory_amp[mask])**2))
    phase_diff = np.angle(np.exp(1j * (phase_rel - theory_phase_rel)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n远场结果:")
    print(f"  振幅缩放因子: {scale:.6f}")
    print(f"  振幅误差 RMS: {amp_error:.6f}")
    print(f"  相位误差 RMS: {phase_error:.6f} rad")
    
    passed = amp_error < 0.001 and phase_error < 0.01
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    return passed, amp_error, phase_error, {
        'amplitude': amp_normalized,
        'phase_rel': phase_rel,
        'theory_amp': theory_amp,
        'theory_phase_rel': theory_phase_rel,
        'phase_diff': phase_diff,
        'mask': mask,
    }



def verify_boundary_case(wavelength, w0, grid_size, beam_diam_fraction, converter):
    """验证边界情况（z = z_R，正好在瑞利距离处）
    
    参数:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        grid_size: 网格大小
        beam_diam_fraction: 光束直径占网格比例
        converter: StateConverter 实例
    
    返回:
        (pass, amp_error, phase_error, results_dict)
    """
    print("\n" + "=" * 70)
    print("测试 3: 边界情况 (z = z_R)")
    print("=" * 70)
    
    z_R = np.pi * w0**2 / wavelength
    dz = z_R  # 正好传播到瑞利距离
    
    # 创建初始波前
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    dx_initial = wfo.dx
    E_initial = create_initial_gaussian(w0, grid_size, dx_initial)
    wfo.wfarr = proper.prop_shift_center(E_initial.astype(np.complex128))
    
    # 传播
    proper.prop_propagate(wfo, dz)
    
    print(f"\n传播距离: {dz * 1e3:.3f} mm (1.0 z_R)")
    print(f"参考面类型: {wfo.reference_surface}")
    
    # 验证 PROPER 的判断逻辑
    # 条件: abs(z_w0 - z) < rayleigh_factor * z_Rayleigh
    # 当 z = z_R 时: z_R < 1.0 * z_R 是 False，所以应该是 SPHERI
    expected_surface = "SPHERI"
    actual_surface = wfo.reference_surface
    
    print(f"\n边界判断验证:")
    print(f"  PROPER rayleigh_factor: {proper.rayleigh_factor}")
    print(f"  条件: |z_w0 - z| < rayleigh_factor * z_Rayleigh")
    print(f"  实际: |{wfo.z_w0:.6f} - {wfo.z:.6f}| = {abs(wfo.z_w0 - wfo.z):.6f}")
    print(f"  阈值: {proper.rayleigh_factor} * {wfo.z_Rayleigh:.6f} = {proper.rayleigh_factor * wfo.z_Rayleigh:.6f}")
    print(f"  期望参考面: {expected_surface}")
    print(f"  实际参考面: {actual_surface}")
    
    surface_correct = actual_surface == expected_surface
    print(f"  参考面判断: {'PASS' if surface_correct else 'FAIL'}")
    
    # 使用 StateConverter 提取振幅和相位
    grid_sampling = GridSampling.from_proper(wfo)
    amplitude, phase = converter.proper_to_amplitude_phase(wfo, grid_sampling)
    
    # 理论值
    n = grid_size
    dx = wfo.dx
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    w_theory, R_theory, gouy_theory, _ = gaussian_beam_params(w0, wavelength, dz)
    
    print(f"\n理论参数:")
    print(f"  w(z_R) = {w_theory * 1e3:.4f} mm (应为 √2 * w0 = {np.sqrt(2) * w0 * 1e3:.4f} mm)")
    print(f"  R(z_R) = {R_theory * 1e3:.4f} mm (应为 2 * z_R = {2 * z_R * 1e3:.4f} mm)")
    
    theory_amp = (w0 / w_theory) * np.exp(-r_sq / w_theory**2)
    theory_phase = 2 * np.pi / wavelength * r_sq / (2 * R_theory) - gouy_theory
    
    # 比较（减去中心值）
    center = n // 2
    mask = theory_amp > 0.01 * np.max(theory_amp)
    
    phase_rel = phase - phase[center, center]
    theory_phase_rel = theory_phase - theory_phase[center, center]
    
    # 振幅归一化
    scale = np.max(amplitude) / np.max(theory_amp)
    amp_normalized = amplitude / scale
    
    amp_error = np.sqrt(np.mean((amp_normalized[mask] - theory_amp[mask])**2))
    phase_diff = np.angle(np.exp(1j * (phase_rel - theory_phase_rel)))
    phase_error = np.sqrt(np.mean(phase_diff[mask]**2))
    
    print(f"\n边界情况结果:")
    print(f"  振幅缩放因子: {scale:.6f}")
    print(f"  振幅误差 RMS: {amp_error:.6f}")
    print(f"  相位误差 RMS: {phase_error:.6f} rad")
    
    passed = surface_correct and amp_error < 0.001 and phase_error < 0.01
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    return passed, amp_error, phase_error, {
        'amplitude': amp_normalized,
        'phase_rel': phase_rel,
        'theory_amp': theory_amp,
        'theory_phase_rel': theory_phase_rel,
        'phase_diff': phase_diff,
        'mask': mask,
        'surface_correct': surface_correct,
    }


def verify_source_definition_consistency():
    """验证 SourceDefinition.create_initial_wavefront() 的一致性
    
    测试不同 z0 值时的 PLANAR/SPHERI 判断是否与 PROPER 一致
    """
    print("\n" + "=" * 70)
    print("测试 4: SourceDefinition 初始化一致性")
    print("=" * 70)
    
    from hybrid_optical_propagation.data_models import SourceDefinition
    
    wavelength_um = 0.6328
    w0_mm = 0.5
    
    # 计算瑞利长度
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  w0: {w0_mm} mm")
    print(f"  z_R: {z_R_mm:.3f} mm")
    
    # 测试不同的 z0 值
    test_cases = [
        (0.0, "PLANAR", "在束腰处"),
        (-0.5 * z_R_mm, "PLANAR", "在 0.5 z_R 处"),
        (-0.99 * z_R_mm, "PLANAR", "在 0.99 z_R 处（边界内）"),
        (-z_R_mm, "SPHERI", "在 z_R 处（边界）"),
        (-1.01 * z_R_mm, "SPHERI", "在 1.01 z_R 处（边界外）"),
        (-5 * z_R_mm, "SPHERI", "在 5 z_R 处（远场）"),
    ]
    
    all_passed = True
    
    print(f"\n测试结果:")
    print(f"  {'z0 (mm)':<12} {'期望':<8} {'实际':<8} {'结果':<6} {'说明'}")
    print(f"  {'-'*60}")
    
    for z0_mm, expected_surface, description in test_cases:
        source = SourceDefinition(
            wavelength_um=wavelength_um,
            w0_mm=w0_mm,
            z0_mm=z0_mm,
            grid_size=128,
            physical_size_mm=10.0,
        )
        
        _, _, _, wfo = source.create_initial_wavefront()
        actual_surface = wfo.reference_surface
        
        passed = actual_surface == expected_surface
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  {z0_mm:<12.3f} {expected_surface:<8} {actual_surface:<8} {status:<6} {description}")
    
    print(f"\n总体结果: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


def plot_results(near_results, far_results, boundary_results):
    """绘制验证结果"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 近场
    ax = axes[0, 0]
    im = ax.imshow(near_results['amplitude'], cmap='hot')
    ax.set_title('Near-field Amp (PROPER)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(near_results['phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Near-field Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(near_results['theory_phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Near-field Theory Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    phase_error = np.sqrt(np.mean(near_results['phase_diff'][near_results['mask']]**2))
    im = ax.imshow(near_results['phase_diff'] * near_results['mask'], cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title(f'Near-field Diff: {phase_error:.4f} rad')
    plt.colorbar(im, ax=ax)
    
    # 远场
    ax = axes[1, 0]
    im = ax.imshow(far_results['amplitude'], cmap='hot')
    ax.set_title('Far-field Amp (normalized)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(far_results['phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Far-field Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    im = ax.imshow(far_results['theory_phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Far-field Theory Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 3]
    phase_error = np.sqrt(np.mean(far_results['phase_diff'][far_results['mask']]**2))
    im = ax.imshow(far_results['phase_diff'] * far_results['mask'], cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title(f'Far-field Diff: {phase_error:.4f} rad')
    plt.colorbar(im, ax=ax)
    
    # 边界情况
    ax = axes[2, 0]
    im = ax.imshow(boundary_results['amplitude'], cmap='hot')
    ax.set_title('Boundary Amp (z=z_R)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 1]
    im = ax.imshow(boundary_results['phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Boundary Phase (rel)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 2]
    im = ax.imshow(boundary_results['theory_phase_rel'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Boundary Theory Phase')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2, 3]
    phase_error = np.sqrt(np.mean(boundary_results['phase_diff'][boundary_results['mask']]**2))
    im = ax.imshow(boundary_results['phase_diff'] * boundary_results['mask'], cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_title(f'Boundary Diff: {phase_error:.4f} rad')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('PROPER Usage Verification (Using StateConverter)')
    plt.tight_layout()
    plt.savefig('verify_proper_final.png', dpi=150)
    plt.close()
    
    print(f"\nFigure saved: verify_proper_final.png")


def main():
    """主函数"""
    print("=" * 70)
    print("最终验证 PROPER 使用方式（使用 StateConverter）")
    print("=" * 70)
    
    # 参数
    wavelength = 632.8e-9  # m
    w0 = 0.5e-3  # m
    grid_size = 256
    beam_diam_fraction = 0.3
    
    wavelength_um = wavelength * 1e6
    
    z_R = np.pi * w0**2 / wavelength
    
    print(f"\n参数:")
    print(f"  波长: {wavelength * 1e9:.1f} nm")
    print(f"  w0: {w0 * 1e3:.3f} mm")
    print(f"  z_R: {z_R * 1e3:.3f} mm")
    print(f"  PROPER rayleigh_factor: {proper.rayleigh_factor}")
    
    # 创建 StateConverter
    converter = StateConverter(wavelength_um)
    
    # 运行测试
    near_pass, _, _, near_results = verify_near_field(
        wavelength, w0, grid_size, beam_diam_fraction, converter
    )
    
    far_pass, _, _, far_results = verify_far_field(
        wavelength, w0, grid_size, beam_diam_fraction, converter
    )
    
    boundary_pass, _, _, boundary_results = verify_boundary_case(
        wavelength, w0, grid_size, beam_diam_fraction, converter
    )
    
    source_pass = verify_source_definition_consistency()
    
    # 绘制结果
    plot_results(near_results, far_results, boundary_results)
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    print(f"\n测试 1 - 近场 (PLANAR): {'PASS' if near_pass else 'FAIL'}")
    print(f"测试 2 - 远场 (SPHERI): {'PASS' if far_pass else 'FAIL'}")
    print(f"测试 3 - 边界 (z=z_R): {'PASS' if boundary_pass else 'FAIL'}")
    print(f"测试 4 - SourceDefinition: {'PASS' if source_pass else 'FAIL'}")
    
    all_pass = near_pass and far_pass and boundary_pass and source_pass
    
    if all_pass:
        print("\n✓ 所有验证通过！")
        print("\n关键结论:")
        print("  1. StateConverter.proper_to_amplitude_phase() 正确处理 PLANAR 和 SPHERI")
        print("  2. PROPER 参考球面相位使用正号: +k*r²/(2*R_ref)")
        print("  3. 重建公式: 完整相位 = PROPER相位 + 参考球面相位")
        print("  4. R_ref = z - z_w0 (PROPER 远场近似)")
        print("  5. 边界判断: |z - z_w0| < rayleigh_factor * z_R → PLANAR")
        print("  6. SourceDefinition.create_initial_wavefront() 与 PROPER 逻辑一致")
    else:
        print("\n✗ 部分验证失败，请检查代码！")
    
    return all_pass


if __name__ == "__main__":
    proper.print_it = False
    proper.verbose = False
    
    result = main()
    print(f"\n最终结果: {'PASS' if result else 'FAIL'}")
