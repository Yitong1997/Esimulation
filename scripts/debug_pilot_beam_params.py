"""
分析 Pilot Beam 参数计算是否正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

# 创建源
source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=256,
    physical_size_mm=40.0,
)

# 创建传播器
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=256,
    num_rays=150,
)

# 初始化并传播
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]

for i in range(4):
    propagator._propagate_to_surface(i)

# 分析每个状态的 Pilot Beam 参数
print("=" * 80)
print("Pilot Beam 参数分析")
print("=" * 80)

wavelength_um = 0.55
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

# 计算理论 Rayleigh 距离
w0_mm = 5.0
z_R_mm = np.pi * w0_mm**2 / wavelength_mm
print(f"\n理论参数:")
print(f"  w0 = {w0_mm} mm")
print(f"  λ = {wavelength_um} μm = {wavelength_mm} mm")
print(f"  z_R = π × w0² / λ = {z_R_mm:.2f} mm")

print("\n" + "-" * 80)

for state in propagator._surface_states:
    print(f"\n表面 {state.surface_index}, 位置: {state.position}")
    print("-" * 40)
    
    pb = state.pilot_beam_params
    print(f"  w0 (束腰半径): {pb.waist_radius_mm:.4f} mm")
    print(f"  waist_position (束腰位置): {pb.waist_position_mm:.4f} mm")
    print(f"  spot_size (光斑大小): {pb.spot_size_mm:.4f} mm")
    print(f"  曲率半径 R: {pb.curvature_radius_mm:.4f} mm")
    print(f"  q 参数: {pb.q_parameter}")
    print(f"  瑞利长度: {pb.rayleigh_length_mm:.4f} mm")
    
    # 从 q 参数提取 z
    z = np.real(pb.q_parameter)
    z_R = np.imag(pb.q_parameter)
    print(f"  从 q 提取: z = {z:.4f} mm, z_R = {z_R:.4f} mm")
    
    # 使用严格公式计算曲率半径
    if abs(z) > 1e-10:
        R_strict = z * (1 + (z_R / z)**2)
        print(f"  严格公式 R = z × (1 + (z_R/z)²) = {R_strict:.4f} mm")
    else:
        print(f"  严格公式 R = ∞ (在束腰处)")
    
    # 检查相位范围
    if state.phase is not None:
        grid_size = state.grid_sampling.grid_size
        physical_size_mm = state.grid_sampling.physical_size_mm
        
        # 计算 Pilot Beam 相位
        pilot_phase = pb.compute_phase_grid(grid_size, physical_size_mm)
        
        # 有效区域
        mask = state.amplitude > 0.01 * np.max(state.amplitude)
        
        # 计算最大半径处的相位
        half_size = physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
        r_sq = X**2 + Y**2
        
        max_r_sq = np.max(r_sq[mask])
        max_r = np.sqrt(max_r_sq)
        
        # 理论相位（使用 Pilot Beam 曲率半径）
        if not np.isinf(pb.curvature_radius_mm):
            theoretical_phase_max = k * max_r_sq / (2 * pb.curvature_radius_mm)
        else:
            theoretical_phase_max = 0.0
        
        # 实际仿真相位
        sim_phase_max = np.max(state.phase[mask])
        sim_phase_min = np.min(state.phase[mask])
        
        print(f"\n  相位分析:")
        print(f"    有效区域最大半径: {max_r:.4f} mm")
        print(f"    仿真相位范围: [{sim_phase_min:.6f}, {sim_phase_max:.6f}] rad")
        print(f"    Pilot Beam 相位最大值: {theoretical_phase_max:.6f} rad")
        print(f"    Pilot Beam 相位范围: [{np.min(pilot_phase[mask]):.6f}, {np.max(pilot_phase[mask]):.6f}] rad")
        
        # 如果仿真相位和 Pilot Beam 相位差异很大，计算实际曲率半径
        if sim_phase_max > 0.001:  # 相位足够大才有意义
            # 从仿真相位反推曲率半径
            # phase = k * r² / (2 * R) => R = k * r² / (2 * phase)
            R_from_sim = k * max_r_sq / (2 * sim_phase_max)
            print(f"    从仿真相位反推的曲率半径: {R_from_sim:.4f} mm")

print("\n" + "=" * 80)
print("分析完成")
