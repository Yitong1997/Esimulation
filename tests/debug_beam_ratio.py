"""调试 beam_ratio 的含义

验证 PROPER 中 beam_ratio 的实际效果
"""

import numpy as np
import proper


def analyze_beam_ratio():
    """分析 beam_ratio 的效果"""
    print("=" * 70)
    print("分析 beam_ratio 的效果")
    print("=" * 70)
    
    wavelength_m = 0.633e-6
    beam_diameter_m = 0.01  # 10 mm
    grid_size = 512
    
    for beam_ratio in [0.1, 0.25, 0.5]:
        print(f"\nbeam_ratio = {beam_ratio}")
        print("-" * 50)
        
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        
        print(f"  网格大小: {n}x{n}")
        print(f"  采样间隔: {sampling_mm:.4f} mm/pixel")
        print(f"  网格范围: ±{sampling_mm * n / 2:.2f} mm")
        
        # 获取振幅
        amp = proper.prop_get_amplitude(wfo)
        amp_threshold = 0.01 * np.max(amp)
        beam_mask = amp > amp_threshold
        
        # 计算光束范围
        proper_half_size_mm = sampling_mm * n / 2
        coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
        X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
        
        beam_x = X_mm[beam_mask]
        beam_y = Y_mm[beam_mask]
        
        print(f"  光束 X 范围: [{np.min(beam_x):.2f}, {np.max(beam_x):.2f}] mm")
        print(f"  光束 Y 范围: [{np.min(beam_y):.2f}, {np.max(beam_y):.2f}] mm")
        
        # 计算光束半径（1/e² 半径）
        R = np.sqrt(X_mm**2 + Y_mm**2)
        beam_radius_mm = np.max(R[beam_mask])
        print(f"  光束半径（1%阈值）: {beam_radius_mm:.2f} mm")
        
        # 计算理论光束半径
        # beam_ratio = beam_diameter / grid_size_physical
        # grid_size_physical = beam_diameter / beam_ratio
        grid_physical_mm = beam_diameter_m * 1e3 / beam_ratio
        print(f"  网格物理尺寸: {grid_physical_mm:.2f} mm")
        print(f"  光束直径: {beam_diameter_m * 1e3:.2f} mm")


if __name__ == "__main__":
    analyze_beam_ratio()
