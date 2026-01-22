"""
调试 PROPER 相位提取

检查从 PROPER 提取的相位是否正确
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import proper
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation.data_models import PilotBeamParams, GridSampling


def main():
    """主函数"""
    
    print("=" * 60)
    print("测试 PROPER 相位提取")
    print("=" * 60)
    
    # 参数
    wavelength_um = 0.55
    wavelength_m = wavelength_um * 1e-6
    w0_mm = 5.0
    w0_m = w0_mm * 1e-3
    grid_size = 256
    physical_size_mm = 40.0
    physical_size_m = physical_size_mm * 1e-3
    
    # 计算瑞利长度
    wavelength_mm = wavelength_um * 1e-3
    z_R_mm = np.pi * w0_mm**2 / wavelength_mm
    z_R_m = z_R_mm * 1e-3
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R_mm:.2f} mm")
    print(f"  网格大小: {grid_size}")
    print(f"  物理尺寸: {physical_size_mm} mm")
    
    # 创建 PROPER 波前
    beam_diameter_m = 2 * w0_m
    beam_diam_fraction = beam_diameter_m / physical_size_m
    
    wfo = proper.prop_begin(
        beam_diameter_m,
        wavelength_m,
        grid_size,
        beam_diam_fraction,
    )
    
    print(f"\n初始 PROPER 状态:")
    print(f"  z: {wfo.z} m")
    print(f"  z_w0: {wfo.z_w0} m")
    print(f"  w0: {wfo.w0} m")
    print(f"  z_Rayleigh: {wfo.z_Rayleigh} m")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 传播 40mm
    distance_mm = 40.0
    distance_m = distance_mm * 1e-3
    
    proper.prop_propagate(wfo, distance_m)
    
    print(f"\n传播 {distance_mm} mm 后:")
    print(f"  z: {wfo.z} m = {wfo.z * 1e3} mm")
    print(f"  z_w0: {wfo.z_w0} m")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 提取相位
    amplitude = proper.prop_get_amplitude(wfo)
    phase = proper.prop_get_phase(wfo)
    
    print(f"\n从 PROPER 提取的相位:")
    print(f"  范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    
    # 计算 PROPER 参考面相位
    if wfo.reference_surface == "SPHERI":
        R_ref_m = wfo.z - wfo.z_w0
        print(f"\nPROPER 参考面:")
        print(f"  类型: SPHERI")
        print(f"  R_ref: {R_ref_m} m = {R_ref_m * 1e3} mm")
        
        # 计算参考面相位
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        n = grid_size
        x = np.arange(n) - n // 2
        y = np.arange(n) - n // 2
        xx, yy = np.meshgrid(x * sampling_m, y * sampling_m)
        r_sq_m = xx**2 + yy**2
        
        k = 2 * np.pi / wavelength_m
        ref_phase = k * r_sq_m / (2 * R_ref_m)
        
        print(f"  参考面相位范围: [{np.min(ref_phase):.6f}, {np.max(ref_phase):.6f}] rad")
        
        # 重建绝对相位
        absolute_phase = ref_phase + phase
        print(f"\n重建的绝对相位:")
        print(f"  范围: [{np.min(absolute_phase):.6f}, {np.max(absolute_phase):.6f}] rad")
    else:
        print(f"\nPROPER 参考面: PLANAR")
        absolute_phase = phase
    
    # 计算 Pilot Beam 相位
    pilot = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, 0.0)
    pilot_after = pilot.propagate(distance_mm)
    
    print(f"\nPilot Beam 参数（传播后）:")
    print(f"  束腰位置: {pilot_after.waist_position_mm} mm")
    print(f"  曲率半径: {pilot_after.curvature_radius_mm:.2f} mm")
    
    # 计算 Pilot Beam 相位
    R_pilot_mm = pilot_after.curvature_radius_mm
    k_mm = 2 * np.pi / wavelength_mm
    
    sampling_mm = proper.prop_get_sampling(wfo) * 1e3
    x_mm = np.arange(grid_size) - grid_size // 2
    y_mm = np.arange(grid_size) - grid_size // 2
    xx_mm, yy_mm = np.meshgrid(x_mm * sampling_mm, y_mm * sampling_mm)
    r_sq_mm = xx_mm**2 + yy_mm**2
    
    if np.isinf(R_pilot_mm):
        pilot_phase = np.zeros_like(r_sq_mm)
    else:
        pilot_phase = k_mm * r_sq_mm / (2 * R_pilot_mm)
    
    print(f"\nPilot Beam 相位:")
    print(f"  范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    
    # 比较
    mask = amplitude > 0.01
    phase_diff = absolute_phase - pilot_phase
    
    print(f"\n绝对相位与 Pilot Beam 相位的差异:")
    print(f"  范围: [{np.min(phase_diff[mask]):.6f}, {np.max(phase_diff[mask]):.6f}] rad")
    print(f"  RMS: {np.std(phase_diff[mask]):.6f} rad")
    print(f"  RMS (waves): {np.std(phase_diff[mask]) / (2 * np.pi):.6f}")
    
    # 在 r = 20mm 处比较
    r_edge = 20.0  # mm
    r_sq_edge = r_edge**2
    
    if np.isinf(R_pilot_mm):
        pilot_phase_edge = 0.0
    else:
        pilot_phase_edge = k_mm * r_sq_edge / (2 * R_pilot_mm)
    
    # PROPER 参考面相位在 r = 20mm 处
    if wfo.reference_surface == "SPHERI":
        r_sq_edge_m = (r_edge * 1e-3)**2
        ref_phase_edge = k * r_sq_edge_m / (2 * R_ref_m)
    else:
        ref_phase_edge = 0.0
    
    print(f"\n在 r = {r_edge} mm 处:")
    print(f"  PROPER 参考面相位: {ref_phase_edge:.6f} rad = {ref_phase_edge / (2 * np.pi):.6f} waves")
    print(f"  Pilot Beam 相位: {pilot_phase_edge:.6f} rad = {pilot_phase_edge / (2 * np.pi):.6f} waves")
    print(f"  差异: {ref_phase_edge - pilot_phase_edge:.6f} rad")
    
    # 比较曲率半径
    print(f"\n曲率半径比较:")
    print(f"  PROPER R_ref: {R_ref_m * 1e3:.2f} mm")
    print(f"  Pilot Beam R: {R_pilot_mm:.2f} mm")
    print(f"  比值: {(R_ref_m * 1e3) / R_pilot_mm:.6f}")
    
    print("\n完成")


if __name__ == '__main__':
    main()
