"""
调试振幅剖面问题

问题：出射面振幅剖面应该是高斯型，但实际显示为近似恒定强度。
目标：定位问题的精确原因，不修改代码。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import matplotlib.pyplot as plt

from hybrid_simulation import HybridSimulator


def main():
    """逐步追踪振幅变化"""
    print("=" * 70)
    print("调试振幅剖面问题")
    print("=" * 70)
    
    # 仿真参数
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 256
    mirror_z_mm = 50.0
    tilt_angle_deg = 45.0
    
    # 执行仿真
    sim = HybridSimulator(verbose=False)
    sim.add_flat_mirror(z=mirror_z_mm, tilt_x=tilt_angle_deg, aperture=30.0)
    sim.set_source(wavelength_um=wavelength_um, w0_mm=w0_mm, grid_size=grid_size,
                   physical_size_mm=8 * w0_mm)
    result = sim.run()
    
    if not result.success:
        print(f"仿真失败: {result.error_message}")
        return
    
    # 分析各阶段振幅
    print("\n" + "=" * 70)
    print("各阶段振幅分析")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for surface in result.surfaces:
        print(f"\n--- 表面 {surface.index}: {surface.name} ---")
        
        # 入射面
        if surface.entrance is not None:
            wf = surface.entrance
            amp = wf.amplitude
            grid = wf.grid
            
            print(f"\n  入射面振幅:")
            print(f"    形状: {amp.shape}")
            print(f"    最大值: {np.max(amp):.6f}")
            print(f"    最小值: {np.min(amp):.6f}")
            print(f"    中心值: {amp[grid.grid_size//2, grid.grid_size//2]:.6f}")
            print(f"    边缘值 (角落): {amp[0, 0]:.6f}")
            
            # 计算理论高斯振幅
            half_size = grid.physical_size_mm / 2
            coords = np.linspace(-half_size, half_size, grid.grid_size)
            center = grid.grid_size // 2
            
            # 理论高斯
            r = coords[center:]
            theory_amp = np.exp(-r**2 / w0_mm**2)
            actual_amp = amp[center, center:] / np.max(amp)
            
            # 检查是否为高斯分布
            if np.max(amp) > 0:
                ratio = amp[0, 0] / np.max(amp)
                print(f"    边缘/中心比: {ratio:.6f}")
                print(f"    理论边缘/中心比 (高斯): {np.exp(-(half_size**2) / w0_mm**2):.6f}")
            
            if surface.index == -1:
                ax = axes[0, 0]
                ax.plot(r, actual_amp, 'b-', label='实际', linewidth=2)
                ax.plot(r, theory_amp, 'r--', label='理论高斯', linewidth=2)
                ax.set_title(f'初始光源振幅剖面')
                ax.set_xlabel('半径 (mm)')
                ax.set_ylabel('归一化振幅')
                ax.legend()
                ax.grid(True, alpha=0.3)
            elif surface.index == 0:
                ax = axes[0, 1]
                ax.plot(r, actual_amp, 'b-', label='实际', linewidth=2)
                ax.plot(r, theory_amp, 'r--', label='理论高斯', linewidth=2)
                ax.set_title(f'Surface 0 入射面振幅剖面')
                ax.set_xlabel('半径 (mm)')
                ax.set_ylabel('归一化振幅')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 出射面
        if surface.exit is not None:
            wf = surface.exit
            amp = wf.amplitude
            grid = wf.grid
            
            print(f"\n  出射面振幅:")
            print(f"    形状: {amp.shape}")
            print(f"    最大值: {np.max(amp):.6f}")
            print(f"    最小值: {np.min(amp):.6f}")
            print(f"    中心值: {amp[grid.grid_size//2, grid.grid_size//2]:.6f}")
            print(f"    边缘值 (角落): {amp[0, 0]:.6f}")
            
            half_size = grid.physical_size_mm / 2
            coords = np.linspace(-half_size, half_size, grid.grid_size)
            center = grid.grid_size // 2
            
            r = coords[center:]
            theory_amp = np.exp(-r**2 / w0_mm**2)
            actual_amp = amp[center, center:] / np.max(amp) if np.max(amp) > 0 else amp[center, center:]
            
            if np.max(amp) > 0:
                ratio = amp[0, 0] / np.max(amp)
                print(f"    边缘/中心比: {ratio:.6f}")
                print(f"    理论边缘/中心比 (高斯): {np.exp(-(half_size**2) / w0_mm**2):.6f}")
            
            ax = axes[0, 2]
            ax.plot(r, actual_amp, 'b-', label='实际', linewidth=2)
            ax.plot(r, theory_amp, 'r--', label='理论高斯', linewidth=2)
            ax.set_title(f'Surface 0 出射面振幅剖面')
            ax.set_xlabel('半径 (mm)')
            ax.set_ylabel('归一化振幅')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2D 振幅图
            ax = axes[1, 0]
            extent = [-half_size, half_size, -half_size, half_size]
            im = ax.imshow(amp, extent=extent, cmap='hot', origin='lower')
            ax.set_title('出射面振幅 2D')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax)
            
            # X 和 Y 剖面对比
            ax = axes[1, 1]
            ax.plot(coords, amp[center, :], 'b-', label='X 剖面', linewidth=2)
            ax.plot(coords, amp[:, center], 'r--', label='Y 剖面', linewidth=2)
            ax.set_title('出射面 X/Y 剖面对比')
            ax.set_xlabel('位置 (mm)')
            ax.set_ylabel('振幅')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 振幅直方图
            ax = axes[1, 2]
            valid_amp = amp[amp > 0.01 * np.max(amp)]
            ax.hist(valid_amp.flatten(), bins=50, edgecolor='black')
            ax.set_title('出射面振幅分布直方图')
            ax.set_xlabel('振幅值')
            ax.set_ylabel('像素数')
            ax.axvline(np.max(amp), color='r', linestyle='--', label=f'最大值: {np.max(amp):.4f}')
            ax.axvline(np.mean(valid_amp), color='g', linestyle='--', label=f'均值: {np.mean(valid_amp):.4f}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('output/debug_amplitude_profile.png', dpi=150)
    print(f"\n调试图已保存: output/debug_amplitude_profile.png")
    plt.close()
    
    # 详细数值分析
    print("\n" + "=" * 70)
    print("详细数值分析")
    print("=" * 70)
    
    for surface in result.surfaces:
        if surface.exit is not None:
            wf = surface.exit
            amp = wf.amplitude
            grid = wf.grid
            center = grid.grid_size // 2
            
            print(f"\n出射面振幅详细分析:")
            print(f"  中心 5x5 区域振幅:")
            print(amp[center-2:center+3, center-2:center+3])
            
            print(f"\n  边缘区域振幅 (左上角 5x5):")
            print(amp[0:5, 0:5])
            
            print(f"\n  边缘区域振幅 (右下角 5x5):")
            print(amp[-5:, -5:])
            
            # 检查振幅变化率
            half_size = grid.physical_size_mm / 2
            coords = np.linspace(-half_size, half_size, grid.grid_size)
            
            x_profile = amp[center, :]
            y_profile = amp[:, center]
            
            # 计算梯度
            dx = coords[1] - coords[0]
            x_gradient = np.gradient(x_profile, dx)
            y_gradient = np.gradient(y_profile, dx)
            
            print(f"\n  X 剖面梯度:")
            print(f"    最大梯度: {np.max(np.abs(x_gradient)):.6f}")
            print(f"    中心梯度: {x_gradient[center]:.6f}")
            
            print(f"\n  Y 剖面梯度:")
            print(f"    最大梯度: {np.max(np.abs(y_gradient)):.6f}")
            print(f"    中心梯度: {y_gradient[center]:.6f}")


if __name__ == '__main__':
    main()
