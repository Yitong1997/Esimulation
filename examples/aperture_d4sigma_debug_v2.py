# -*- coding: utf-8 -*-
"""
深入调试 PROPER 高斯光束问题

发现：PROPER 创建的高斯光束 D4sigma = 4.6 mm，而理论值应该是 2 mm
这说明 PROPER 的高斯光束本身就有问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from bts.beam_measurement import D4sigmaCalculator


def investigate_proper_gaussian():
    """深入调查 PROPER 高斯光束"""
    print("=" * 70)
    print("深入调查 PROPER 高斯光束")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    # 创建 PROPER 高斯光束
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    # 获取振幅
    amp_proper = proper.prop_get_amplitude(wfo)
    sampling = proper.prop_get_sampling(wfo)
    
    print(f"\n1. PROPER 参数:")
    print(f"   beam_diameter = {beam_diameter * 1e3:.4f} mm")
    print(f"   wavelength = {wavelength * 1e9:.1f} nm")
    print(f"   grid_size = {grid_size}")
    print(f"   sampling = {sampling * 1e6:.4f} μm")
    print(f"   physical_size = {sampling * grid_size * 1e3:.4f} mm")
    
    # 检查 PROPER 振幅的形状
    print(f"\n2. PROPER 振幅数组分析:")
    print(f"   形状: {amp_proper.shape}")
    print(f"   最小值: {amp_proper.min():.6e}")
    print(f"   最大值: {amp_proper.max():.6e}")
    print(f"   中心值: {amp_proper[grid_size//2, grid_size//2]:.6e}")
    
    # 创建坐标网格
    x = (np.arange(grid_size) - grid_size / 2) * sampling
    y = (np.arange(grid_size) - grid_size / 2) * sampling
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 理论高斯振幅（归一化到 1）
    amp_theory = np.exp(-(R / w0)**2)
    
    print(f"\n3. 理论高斯振幅:")
    print(f"   中心值: {amp_theory[grid_size//2, grid_size//2]:.6f}")
    print(f"   r=w0 处值: {np.exp(-1):.6f}")
    
    # 检查 PROPER 振幅的径向分布
    center = grid_size // 2
    r_pixels = np.arange(grid_size // 2)
    r_mm = r_pixels * sampling * 1e3
    
    proper_profile = amp_proper[center, center:]
    theory_profile = amp_theory[center, center:]
    
    print(f"\n4. 径向剖面对比:")
    print(f"   r(mm)  | PROPER    | 理论      | 比值")
    print("-" * 50)
    for i in [0, 32, 64, 96, 128, 160, 192]:
        if i < len(proper_profile):
            r = i * sampling * 1e3
            ratio = proper_profile[i] / theory_profile[i] if theory_profile[i] > 1e-10 else 0
            print(f"   {r:6.3f} | {proper_profile[i]:.6e} | {theory_profile[i]:.6f} | {ratio:.6f}")
    
    # 检查 PROPER 是否在边缘有问题
    print(f"\n5. 边缘分析（r > 1.5 mm）:")
    edge_mask = R > 1.5e-3
    print(f"   PROPER 边缘平均值: {amp_proper[edge_mask].mean():.6e}")
    print(f"   理论边缘平均值: {amp_theory[edge_mask].mean():.6e}")
    
    # 检查 PROPER 的 wfarr 是否使用了 FFT 坐标系
    print(f"\n6. 检查 wfarr 坐标系:")
    wfarr = wfo.wfarr
    print(f"   wfarr 形状: {wfarr.shape}")
    print(f"   wfarr[0,0] 振幅: {np.abs(wfarr[0, 0]):.6e}")
    print(f"   wfarr[center,center] 振幅: {np.abs(wfarr[center, center]):.6e}")
    
    # 检查 prop_get_amplitude 是否正确移动了中心
    amp_direct = np.abs(wfarr)
    amp_shifted = np.fft.fftshift(amp_direct)
    
    print(f"\n7. 坐标系移动检查:")
    print(f"   直接取 wfarr 的中心值: {amp_direct[center, center]:.6e}")
    print(f"   fftshift 后的中心值: {amp_shifted[center, center]:.6e}")
    print(f"   prop_get_amplitude 的中心值: {amp_proper[center, center]:.6e}")
    
    return amp_proper, amp_theory, sampling, w0



def investigate_proper_normalization():
    """调查 PROPER 的归一化方式"""
    print("\n" + "=" * 70)
    print("调查 PROPER 的归一化方式")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    amp_proper = proper.prop_get_amplitude(wfo)
    sampling = proper.prop_get_sampling(wfo)
    
    # 计算 PROPER 的总功率
    power_proper = np.sum(amp_proper**2) * sampling**2
    
    # 创建理论高斯
    x = (np.arange(grid_size) - grid_size / 2) * sampling
    y = (np.arange(grid_size) - grid_size / 2) * sampling
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 理论高斯（归一化到总功率 = 1）
    amp_theory_raw = np.exp(-(R / w0)**2)
    power_theory_raw = np.sum(amp_theory_raw**2) * sampling**2
    amp_theory_norm = amp_theory_raw / np.sqrt(power_theory_raw)
    power_theory_norm = np.sum(amp_theory_norm**2) * sampling**2
    
    print(f"\n功率分析:")
    print(f"  PROPER 总功率: {power_proper:.6e}")
    print(f"  理论（原始）总功率: {power_theory_raw:.6e}")
    print(f"  理论（归一化）总功率: {power_theory_norm:.6e}")
    
    # 检查 PROPER 的归一化因子
    print(f"\n归一化因子:")
    print(f"  PROPER 峰值: {amp_proper.max():.6e}")
    print(f"  理论（原始）峰值: {amp_theory_raw.max():.6f}")
    print(f"  理论（归一化）峰值: {amp_theory_norm.max():.6e}")
    
    # 尝试用相同的归一化方式比较
    # 将 PROPER 归一化到峰值 = 1
    amp_proper_norm = amp_proper / amp_proper.max()
    
    # 计算 D4sigma
    calc = D4sigmaCalculator()
    
    result_proper = calc.calculate(amp_proper, sampling=sampling)
    result_proper_norm = calc.calculate(amp_proper_norm, sampling=sampling)
    result_theory = calc.calculate(amp_theory_raw, sampling=sampling)
    
    print(f"\nD4sigma 测量:")
    print(f"  PROPER（原始）: {result_proper.d_mean * 1e3:.4f} mm")
    print(f"  PROPER（归一化）: {result_proper_norm.d_mean * 1e3:.4f} mm")
    print(f"  理论: {result_theory.d_mean * 1e3:.4f} mm")
    print(f"  理想值 2×w0: {2 * w0 * 1e3:.4f} mm")
    
    # 关键：D4sigma 不应该受归一化影响！
    print(f"\n关键发现：D4sigma 不应该受归一化影响")
    print(f"  但 PROPER 的 D4sigma = {result_proper.d_mean * 1e3:.4f} mm")
    print(f"  这说明 PROPER 的振幅分布形状与理论高斯不同！")
    
    return amp_proper, amp_theory_raw, sampling, w0


def investigate_proper_beam_shape():
    """调查 PROPER 光束的实际形状"""
    print("\n" + "=" * 70)
    print("调查 PROPER 光束的实际形状")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    amp_proper = proper.prop_get_amplitude(wfo)
    sampling = proper.prop_get_sampling(wfo)
    
    # 创建坐标网格
    x = (np.arange(grid_size) - grid_size / 2) * sampling
    y = (np.arange(grid_size) - grid_size / 2) * sampling
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 理论高斯
    amp_theory = np.exp(-(R / w0)**2)
    
    # 归一化到相同峰值
    amp_proper_norm = amp_proper / amp_proper.max()
    
    # 计算差异
    diff = amp_proper_norm - amp_theory
    
    print(f"\n形状差异分析（归一化到峰值=1）:")
    print(f"  差异 RMS: {np.sqrt(np.mean(diff**2)):.6e}")
    print(f"  差异最大值: {np.abs(diff).max():.6e}")
    
    # 检查 PROPER 是否在边缘有额外的能量
    # 这可能是 D4sigma 偏大的原因
    
    # 计算不同半径范围内的能量占比
    print(f"\n能量分布分析:")
    print(f"  半径范围 | PROPER 能量占比 | 理论能量占比")
    print("-" * 55)
    
    intensity_proper = amp_proper_norm**2
    intensity_theory = amp_theory**2
    
    total_proper = np.sum(intensity_proper)
    total_theory = np.sum(intensity_theory)
    
    for r_max in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r_max_m = r_max * w0
        mask = R <= r_max_m
        
        frac_proper = np.sum(intensity_proper[mask]) / total_proper * 100
        frac_theory = np.sum(intensity_theory[mask]) / total_theory * 100
        
        print(f"  r < {r_max:.1f}×w0 | {frac_proper:6.2f}%         | {frac_theory:6.2f}%")
    
    # 检查边缘能量
    edge_mask = R > 2 * w0
    edge_proper = np.sum(intensity_proper[edge_mask]) / total_proper * 100
    edge_theory = np.sum(intensity_theory[edge_mask]) / total_theory * 100
    
    print(f"\n边缘能量（r > 2×w0）:")
    print(f"  PROPER: {edge_proper:.4f}%")
    print(f"  理论: {edge_theory:.4f}%")
    
    if edge_proper > edge_theory * 10:
        print(f"\n⚠️ PROPER 在边缘有异常高的能量！")
        print(f"   这是 D4sigma 偏大的原因")
    
    return amp_proper_norm, amp_theory, diff, sampling, w0



def investigate_proper_wfarr():
    """直接检查 PROPER 的 wfarr 数组"""
    print("\n" + "=" * 70)
    print("直接检查 PROPER 的 wfarr 数组")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    sampling = proper.prop_get_sampling(wfo)
    center = grid_size // 2
    
    # 直接访问 wfarr
    wfarr = wfo.wfarr
    
    print(f"\n1. wfarr 基本信息:")
    print(f"   形状: {wfarr.shape}")
    print(f"   dtype: {wfarr.dtype}")
    
    # 检查 wfarr 的振幅分布
    amp_wfarr = np.abs(wfarr)
    
    print(f"\n2. wfarr 振幅分布:")
    print(f"   [0,0]: {amp_wfarr[0, 0]:.6e}")
    print(f"   [0,center]: {amp_wfarr[0, center]:.6e}")
    print(f"   [center,0]: {amp_wfarr[center, 0]:.6e}")
    print(f"   [center,center]: {amp_wfarr[center, center]:.6e}")
    
    # 检查最大值位置
    max_idx = np.unravel_index(np.argmax(amp_wfarr), amp_wfarr.shape)
    print(f"\n3. 最大值位置:")
    print(f"   索引: {max_idx}")
    print(f"   值: {amp_wfarr[max_idx]:.6e}")
    
    # 检查 prop_get_amplitude 的实现
    amp_proper = proper.prop_get_amplitude(wfo)
    
    print(f"\n4. prop_get_amplitude 结果:")
    print(f"   [0,0]: {amp_proper[0, 0]:.6e}")
    print(f"   [center,center]: {amp_proper[center, center]:.6e}")
    
    max_idx_proper = np.unravel_index(np.argmax(amp_proper), amp_proper.shape)
    print(f"   最大值位置: {max_idx_proper}")
    print(f"   最大值: {amp_proper[max_idx_proper]:.6e}")
    
    # 手动进行 fftshift
    amp_shifted = np.fft.fftshift(amp_wfarr)
    
    print(f"\n5. 手动 fftshift 结果:")
    print(f"   [center,center]: {amp_shifted[center, center]:.6e}")
    
    max_idx_shifted = np.unravel_index(np.argmax(amp_shifted), amp_shifted.shape)
    print(f"   最大值位置: {max_idx_shifted}")
    print(f"   最大值: {amp_shifted[max_idx_shifted]:.6e}")
    
    # 比较 prop_get_amplitude 和手动 fftshift
    diff = amp_proper - amp_shifted
    print(f"\n6. prop_get_amplitude vs 手动 fftshift:")
    print(f"   差异 RMS: {np.sqrt(np.mean(diff**2)):.6e}")
    print(f"   差异最大值: {np.abs(diff).max():.6e}")
    
    return wfarr, amp_proper, amp_shifted, sampling


def investigate_proper_beam_radius():
    """检查 PROPER 的光束半径定义"""
    print("\n" + "=" * 70)
    print("检查 PROPER 的光束半径定义")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    # 获取 PROPER 报告的光束半径
    beam_radius_proper = proper.prop_get_beamradius(wfo)
    
    print(f"\n1. PROPER 报告的光束半径:")
    print(f"   prop_get_beamradius: {beam_radius_proper * 1e3:.4f} mm")
    print(f"   输入的 w0: {w0 * 1e3:.4f} mm")
    print(f"   比值: {beam_radius_proper / w0:.4f}")
    
    # 从振幅分布拟合光束半径
    amp_proper = proper.prop_get_amplitude(wfo)
    sampling = proper.prop_get_sampling(wfo)
    
    # 创建坐标网格
    x = (np.arange(grid_size) - grid_size / 2) * sampling
    y = (np.arange(grid_size) - grid_size / 2) * sampling
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 归一化振幅
    amp_norm = amp_proper / amp_proper.max()
    
    # 找到振幅下降到 1/e 的位置（对应 r = w）
    # 对于高斯光束，A(r=w) = A(0) × exp(-1) ≈ 0.368
    target_amp = np.exp(-1)
    
    # 沿 x 轴找到这个位置
    center = grid_size // 2
    profile = amp_norm[center, center:]
    
    # 找到最接近 target_amp 的位置
    idx = np.argmin(np.abs(profile - target_amp))
    fitted_w = idx * sampling
    
    print(f"\n2. 从振幅分布拟合的光束半径:")
    print(f"   1/e 振幅位置: {fitted_w * 1e3:.4f} mm")
    print(f"   与输入 w0 的比值: {fitted_w / w0:.4f}")
    
    # 检查 PROPER 的 beam_diameter 定义
    print(f"\n3. PROPER beam_diameter 定义检查:")
    print(f"   输入 beam_diameter = {beam_diameter * 1e3:.4f} mm")
    print(f"   这应该是 2×w0 = {2 * w0 * 1e3:.4f} mm")
    print(f"   PROPER 可能将其解释为 1/e² 直径（强度）")
    print(f"   或 1/e 直径（振幅）")
    
    # 检查 1/e² 强度位置
    intensity_norm = amp_norm**2
    target_intensity = np.exp(-2)  # 1/e² ≈ 0.135
    
    profile_intensity = intensity_norm[center, center:]
    idx_intensity = np.argmin(np.abs(profile_intensity - target_intensity))
    fitted_w_intensity = idx_intensity * sampling
    
    print(f"\n4. 从强度分布拟合的光束半径:")
    print(f"   1/e² 强度位置: {fitted_w_intensity * 1e3:.4f} mm")
    print(f"   与输入 w0 的比值: {fitted_w_intensity / w0:.4f}")
    
    return beam_radius_proper, fitted_w, fitted_w_intensity, w0



def plot_comparison():
    """绘制对比图"""
    print("\n" + "=" * 70)
    print("绘制对比图")
    print("=" * 70)
    
    wavelength = 633e-9
    w0 = 1e-3
    grid_size = 512
    
    beam_diameter = 2 * w0
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)
    proper.prop_define_entrance(wfo)
    
    amp_proper = proper.prop_get_amplitude(wfo)
    sampling = proper.prop_get_sampling(wfo)
    
    # 创建坐标网格
    x = (np.arange(grid_size) - grid_size / 2) * sampling
    y = (np.arange(grid_size) - grid_size / 2) * sampling
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 理论高斯
    amp_theory = np.exp(-(R / w0)**2)
    
    # 归一化
    amp_proper_norm = amp_proper / amp_proper.max()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PROPER 高斯光束 vs 理论高斯光束', fontsize=14)
    
    extent = np.array([-grid_size/2, grid_size/2, -grid_size/2, grid_size/2]) * sampling * 1e3
    
    # 1. PROPER 振幅
    im1 = axes[0, 0].imshow(amp_proper_norm, extent=extent, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('PROPER 振幅（归一化）')
    axes[0, 0].set_xlabel('x (mm)')
    axes[0, 0].set_ylabel('y (mm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 理论振幅
    im2 = axes[0, 1].imshow(amp_theory, extent=extent, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('理论高斯振幅')
    axes[0, 1].set_xlabel('x (mm)')
    axes[0, 1].set_ylabel('y (mm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 差异
    diff = amp_proper_norm - amp_theory
    im3 = axes[1, 0].imshow(diff, extent=extent, cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('差异（PROPER - 理论）')
    axes[1, 0].set_xlabel('x (mm)')
    axes[1, 0].set_ylabel('y (mm)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. 径向剖面
    center = grid_size // 2
    r_mm = np.arange(grid_size // 2) * sampling * 1e3
    
    axes[1, 1].plot(r_mm, amp_proper_norm[center, center:], 'b-', label='PROPER', linewidth=2)
    axes[1, 1].plot(r_mm, amp_theory[center, center:], 'r--', label='理论', linewidth=2)
    axes[1, 1].axvline(x=w0 * 1e3, color='g', linestyle=':', label=f'w0 = {w0*1e3:.1f} mm')
    axes[1, 1].axhline(y=np.exp(-1), color='gray', linestyle=':', alpha=0.5, label='1/e')
    axes[1, 1].set_xlabel('r (mm)')
    axes[1, 1].set_ylabel('归一化振幅')
    axes[1, 1].set_title('径向剖面对比')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 3 * w0 * 1e3)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/proper_gaussian_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  图形已保存: output/proper_gaussian_debug.png")


def main():
    """主函数"""
    print("=" * 70)
    print("PROPER 高斯光束深入调试")
    print("=" * 70)
    
    investigate_proper_gaussian()
    investigate_proper_normalization()
    investigate_proper_beam_shape()
    investigate_proper_wfarr()
    investigate_proper_beam_radius()
    plot_comparison()
    
    print("\n" + "=" * 70)
    print("调试总结")
    print("=" * 70)
    print("""
关键发现：

1. PROPER 创建的高斯光束 D4sigma ≈ 4.6 mm，而理论值应该是 2 mm
   这是一个 130% 的误差！

2. 这说明 PROPER 的振幅分布形状与理论高斯光束完全不同

3. 可能的原因：
   - PROPER 的 beam_diameter 定义与我们的理解不同
   - PROPER 在边缘有额外的能量（可能是数值问题）
   - prop_get_amplitude 的坐标系处理有问题

4. 下一步：
   - 检查 PROPER 的文档，确认 beam_diameter 的定义
   - 检查 PROPER 的源代码，了解高斯光束的创建方式
   - 考虑直接使用 numpy 创建高斯光束，而不是依赖 PROPER
""")
    
    print("\n" + "=" * 70)
    print("调试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
