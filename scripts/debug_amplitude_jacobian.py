"""
调试振幅雅可比矩阵计算

问题：出射面振幅应该是高斯型，但实际是恒定值。
原因分析：RayToWavefrontReconstructor 使用雅可比矩阵方法计算振幅，
基于能量守恒原理：A_out = A_in / sqrt(|J|)

对于平面镜，雅可比行列式 |J| ≈ 1（无聚焦/发散），
所以振幅变化很小，导致输出振幅近似恒定。

但问题是：输入振幅是高斯分布，输出振幅也应该保持高斯分布！
雅可比矩阵方法只计算振幅的**变化**，不保留输入振幅的分布。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np

from hybrid_simulation import HybridSimulator


def main():
    print("=" * 70)
    print("调试振幅雅可比矩阵计算")
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
    
    # 分析入射面和出射面振幅
    print("\n" + "=" * 70)
    print("振幅分析")
    print("=" * 70)
    
    for surface in result.surfaces:
        if surface.index == 0:  # Surface_0
            entrance = surface.entrance
            exit_wf = surface.exit
            
            if entrance is not None and exit_wf is not None:
                ent_amp = entrance.amplitude
                exit_amp = exit_wf.amplitude
                
                grid = entrance.grid
                center = grid.grid_size // 2
                
                print(f"\n入射面振幅:")
                print(f"  中心值: {ent_amp[center, center]:.6f}")
                print(f"  边缘值 (r=10mm): {ent_amp[center, center + int(10/grid.physical_size_mm*grid.grid_size)]:.6f}")
                print(f"  最大值: {np.max(ent_amp):.6f}")
                print(f"  最小值 (非零): {np.min(ent_amp[ent_amp > 0.01]):.6f}")
                
                print(f"\n出射面振幅:")
                print(f"  中心值: {exit_amp[center, center]:.6f}")
                print(f"  边缘值 (r=10mm): {exit_amp[center, center + int(10/grid.physical_size_mm*grid.grid_size)]:.6f}")
                print(f"  最大值: {np.max(exit_amp):.6f}")
                print(f"  最小值 (非零): {np.min(exit_amp[exit_amp > 0.01]):.6f}")
                
                # 计算振幅比
                valid_mask = (ent_amp > 0.01) & (exit_amp > 0.01)
                if np.any(valid_mask):
                    ratio = exit_amp[valid_mask] / ent_amp[valid_mask]
                    print(f"\n振幅比 (出射/入射):")
                    print(f"  平均值: {np.mean(ratio):.6f}")
                    print(f"  标准差: {np.std(ratio):.6f}")
                    print(f"  最大值: {np.max(ratio):.6f}")
                    print(f"  最小值: {np.min(ratio):.6f}")
                
                # 关键发现
                print("\n" + "=" * 70)
                print("关键发现")
                print("=" * 70)
                print("""
问题根源：RayToWavefrontReconstructor 的雅可比矩阵方法

1. 雅可比矩阵方法计算的是振幅的**变化率**，不是绝对振幅
2. 对于平面镜，|J| ≈ 1，所以计算出的振幅变化很小
3. 代码中振幅被归一化：amplitude_valid = amplitude_valid / mean_amplitude
4. 这导致输出振幅在有效区域内近似恒定为 1

正确的做法应该是：
- 输出振幅 = 输入振幅 × (1 / sqrt(|J|))
- 需要保留输入振幅的高斯分布

当前代码 (reconstructor.py 第 260-263 行):
```python
# 归一化振幅（需求 2.4）
mean_amplitude = np.mean(amplitude_valid)
if mean_amplitude > 0:
    amplitude_valid = amplitude_valid / mean_amplitude
```

这个归一化操作丢失了输入振幅的分布信息！
""")


if __name__ == '__main__':
    main()
