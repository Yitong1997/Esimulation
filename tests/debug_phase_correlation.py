"""调试相位相关性问题

问题：相位值在中心和边缘看起来正确，但相关性是 -0.5
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
import matplotlib.pyplot as plt


def debug_phase_correlation():
    """调试相位相关性"""
    print("=" * 70)
    print("调试相位相关性")
    print("=" * 70)
    
    wavelength_m = 0.633e-6
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 128
    beam_ratio = 0.25
    
    # 初始化波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    print(f"\nPROPER 参数:")
    print(f"  网格大小: {n}x{n}")
    print(f"  采样间隔: {sampling_mm:.4f} mm/pixel")
    
    # 创建坐标网格
    proper_half_size_mm = sampling_mm * n / 2
    coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
    X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
    R_mm = np.sqrt(X_mm**2 + Y_mm**2)
    
    # 创建一个简单的离焦像差
    beam_radius_mm = 5.0
    r_norm = R_mm / beam_radius_mm
    defocus_waves = r_norm**2
    mask = R_mm <= beam_radius_mm
    defocus_waves = np.where(mask, defocus_waves, 0.0)
    
    # 应用相位
    wfo_test = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    phase_initial = proper.prop_get_phase(wfo_test)
    
    aberration_phase = -2 * np.pi * defocus_waves
    phase_field = np.exp(1j * aberration_phase)
    phase_field_fft = proper.prop_shift_center(phase_field)
    wfo_test.wfarr = wfo_test.wfarr * phase_field_fft
    
    phase_final = proper.prop_get_phase(wfo_test)
    phase_change = phase_final - phase_initial
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 应用的相位
    ax = axes[0, 0]
    im = ax.imshow(aberration_phase, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm,
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('Applied Phase')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='rad')
    
    # 2. 测量的相位变化
    ax = axes[0, 1]
    im = ax.imshow(phase_change, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm,
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('Measured Phase Change')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='rad')
    
    # 3. 差异
    ax = axes[0, 2]
    diff = phase_change - aberration_phase
    im = ax.imshow(diff, cmap='RdBu_r', origin='lower',
                   extent=[-proper_half_size_mm, proper_half_size_mm,
                          -proper_half_size_mm, proper_half_size_mm])
    ax.set_title('Difference')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.colorbar(im, ax=ax, label='rad')
    
    # 4. 沿 X 轴的剖面
    ax = axes[1, 0]
    center_idx = n // 2
    ax.plot(coords_mm, aberration_phase[center_idx, :], 'b-', label='Applied')
    ax.plot(coords_mm, phase_change[center_idx, :], 'r--', label='Measured')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Profile along X (Y=0)')
    ax.set_xlim(-10, 10)
    ax.legend()
    ax.grid(True)
    
    # 5. 沿 Y 轴的剖面
    ax = axes[1, 1]
    ax.plot(coords_mm, aberration_phase[:, center_idx], 'b-', label='Applied')
    ax.plot(coords_mm, phase_change[:, center_idx], 'r--', label='Measured')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Profile along Y (X=0)')
    ax.set_xlim(-10, 10)
    ax.legend()
    ax.grid(True)
    
    # 6. 散点图
    ax = axes[1, 2]
    valid_applied = aberration_phase[mask]
    valid_measured = phase_change[mask]
    ax.scatter(valid_applied, valid_measured, alpha=0.1, s=1)
    ax.plot([-10, 10], [-10, 10], 'k--', label='y=x')
    ax.set_xlabel('Applied Phase (rad)')
    ax.set_ylabel('Measured Phase (rad)')
    ax.set_title('Scatter Plot')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('tests/output/debug_phase_correlation.png', dpi=150)
    plt.close()
    print("\nFigure saved to tests/output/debug_phase_correlation.png")
    
    # 打印一些统计信息
    print("\n统计信息:")
    print(f"  应用相位范围: [{np.min(aberration_phase[mask]):.4f}, {np.max(aberration_phase[mask]):.4f}] rad")
    print(f"  测量相位范围: [{np.min(phase_change[mask]):.4f}, {np.max(phase_change[mask]):.4f}] rad")
    
    # 检查是否有相位包裹
    print(f"\n  应用相位 > π 的点数: {np.sum(np.abs(aberration_phase[mask]) > np.pi)}")
    print(f"  测量相位 > π 的点数: {np.sum(np.abs(phase_change[mask]) > np.pi)}")


if __name__ == "__main__":
    debug_phase_correlation()
