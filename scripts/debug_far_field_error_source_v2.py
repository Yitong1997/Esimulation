"""
定位远场传播误差的来源 - 修正版

关键问题：误差是来自 PROPER 传播本身，还是来自我们使用 Pilot Beam 严格曲率
与 PROPER 远场近似曲率之间的差异？

测试方法：
1. PROPER 传播后，使用 PROPER 自己的参考面曲率计算相位，与理论高斯光束比较
2. PROPER 传播后，使用 Pilot Beam 严格曲率计算相位，与理论高斯光束比较
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import proper


def test_proper_propagation_with_proper_reference():
    """测试: PROPER 传播后，使用 PROPER 自己的参考面重建相位
    
    如果 PROPER 传播是准确的，那么使用 PROPER 自己的参考面重建的相位
    应该与理论高斯光束相位一致（使用相同的远场近似曲率）
    """
    print("=" * 80)
    print("测试: PROPER 传播精度（使用 PROPER 自己的参考面）")
    print("=" * 80)
    
    wavelength_um = 0.633
    w0_mm = 2.0
    grid_size = 128
    physical_size_mm = 20.0
    
    wavelength_mm = wavelength_um * 1e-3
    wavelength_m = wavelength_um * 1e-6
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    z_R_m = z_R_mm * 1e-3
    
    z_factors = [1.0, 3.0, 10.0, 100.0]
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R_mm:.1f} mm")
    
    print(f"\n{'z/z_R':<10} {'参考面':<10} {'PROPER参考面误差':<20} {'Pilot Beam误差':<20}")
    print("-" * 65)
    
    for z_factor in z_factors:
        propagation_distance_mm = z_factor * z_R_mm
        propagation_distance_m = propagation_distance_mm * 1e-3
        
        # 使用 PROPER 创建初始高斯光束
        beam_diameter_m = 2 * w0_mm * 1e-3
        grid_width_m = physical_size_mm * 1e-3
        beam_diam_fraction = beam_diameter_m / grid_width_m
        
        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            grid_size,
            beam_diam_fraction,
        )
        
        # 传播
        proper.prop_propagate(wfo, propagation_distance_m)
        
        # 从 PROPER 提取
        amplitude = proper.prop_get_amplitude(wfo)
        proper_residual_phase = proper.prop_get_phase(wfo)
        
        # 获取采样
        sampling_m = proper.prop_get_sampling(wfo)
        n = grid_size
        coords_m = (np.arange(n) - n // 2) * sampling_m
        X_m, Y_m = np.meshgrid(coords_m, coords_m)
        r_sq_m = X_m**2 + Y_m**2
        
        k = 2 * np.pi / wavelength_m
        
        # ========== 方法 1: 使用 PROPER 参考面曲率 ==========
        if wfo.reference_surface == "SPHERI":
            R_proper_m = wfo.z - wfo.z_w0
            proper_ref_phase = k * r_sq_m / (2 * R_proper_m)
        else:
            proper_ref_phase = np.zeros((n, n))
            R_proper_m = np.inf
        
        # 重建完整相位
        full_phase_proper = proper_ref_phase + proper_residual_phase
        
        # 理论高斯光束相位（使用 PROPER 的远场近似曲率）
        # 这样比较的是 PROPER 传播本身的精度
        if np.isinf(R_proper_m):
            theory_phase_proper = np.zeros((n, n))
        else:
            theory_phase_proper = k * r_sq_m / (2 * R_proper_m)
        
        # 计算误差
        valid_mask = amplitude > 0.01 * np.max(amplitude)
        phase_diff_proper = np.angle(np.exp(1j * (full_phase_proper - theory_phase_proper)))
        rms_error_proper = np.sqrt(np.mean(phase_diff_proper[valid_mask]**2)) / (2 * np.pi)
        
        # ========== 方法 2: 使用 Pilot Beam 严格曲率 ==========
        z_m = propagation_distance_m
        R_pilot_m = z_m * (1 + (z_R_m / z_m)**2)
        
        # 理论高斯光束相位（使用 Pilot Beam 严格曲率）
        theory_phase_pilot = k * r_sq_m / (2 * R_pilot_m)
        
        # 计算误差
        phase_diff_pilot = np.angle(np.exp(1j * (full_phase_proper - theory_phase_pilot)))
        rms_error_pilot = np.sqrt(np.mean(phase_diff_pilot[valid_mask]**2)) / (2 * np.pi)
        
        print(f"{z_factor:<10.1f} {wfo.reference_surface:<10} {rms_error_proper:<20.6f} {rms_error_pilot:<20.6f}")
    
    print("\n结论:")
    print("  - 'PROPER参考面误差' 反映 PROPER 传播本身的精度")
    print("  - 'Pilot Beam误差' 反映使用不同曲率公式导致的差异")


if __name__ == "__main__":
    test_proper_propagation_with_proper_reference()
