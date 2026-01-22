"""
验证出现误差的入射面的参考波前类型

检查 Surface 4 入射面（第一个出现大误差的位置）的：
1. PROPER 参考面类型（PLANAR 还是 SPHERI）
2. proper_to_amplitude_phase 方法的处理是否正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import proper

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.state_converter import StateConverter

print("=" * 70)
print("验证参考波前类型和 proper_to_amplitude_phase 处理")
print("=" * 70)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=512,
    physical_size_mm=40.0,
)

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=512,
    num_rays=150,
)

# 传播到各个表面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(6):
    propagator._propagate_to_surface(i)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("\n各表面的 PROPER 参考面类型和参数:")
print("-" * 70)

for state in propagator._surface_states:
    wfo = state.proper_wfo
    pb = state.pilot_beam_params
    
    # PROPER 参数
    z = wfo.z
    z_w0 = wfo.z_w0
    z_R = wfo.z_Rayleigh
    ref_surface = wfo.reference_surface
    
    # 判断条件
    rayleigh_factor = proper.rayleigh_factor
    distance_from_waist = abs(z - z_w0)
    threshold = rayleigh_factor * z_R
    
    # Pilot Beam 曲率半径
    R_pilot = pb.curvature_radius_mm
    
    # PROPER 参考面曲率半径（如果是 SPHERI）
    R_ref = (z - z_w0) * 1e3 if ref_surface == "SPHERI" else np.inf  # 转换为 mm
    
    surface_name = f"Surface {state.surface_index}" if state.surface_index >= 0 else "Initial"
    print(f"\n{surface_name} ({state.position}):")
    print(f"  reference_surface: {ref_surface}")
    print(f"  z = {z*1e3:.2f} mm, z_w0 = {z_w0*1e3:.2f} mm, z_R = {z_R*1e3:.2f} mm")
    print(f"  |z - z_w0| = {distance_from_waist*1e3:.2f} mm")
    print(f"  rayleigh_factor * z_R = {threshold*1e3:.2f} mm")
    print(f"  判断: |z - z_w0| {'<' if distance_from_waist < threshold else '>='} threshold → {ref_surface}")
    print(f"  R_ref (PROPER) = {R_ref:.2f} mm")
    print(f"  R_pilot (Pilot Beam) = {R_pilot:.2f} mm")
    if not np.isinf(R_ref) and not np.isinf(R_pilot):
        print(f"  R 差异: {(R_ref - R_pilot)/R_pilot * 100:.4f}%")

print("\n" + "=" * 70)
print("验证 proper_to_amplitude_phase 方法")
print("=" * 70)

# 找到 Surface 4 入射面（第一个出现大误差的位置）
state_s4_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 4 and state.position == 'entrance':
        state_s4_entrance = state
        break

if state_s4_entrance is None:
    print("未找到 Surface 4 入射面状态")
else:
    wfo = state_s4_entrance.proper_wfo
    grid_sampling = state_s4_entrance.grid_sampling
    pb = state_s4_entrance.pilot_beam_params
    
    print(f"\nSurface 4 入射面:")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    # 手动执行 proper_to_amplitude_phase 的步骤
    state_converter = StateConverter(0.55)
    
    # 1. 从 PROPER 提取残差相位
    amplitude = proper.prop_get_amplitude(wfo)
    residual_phase = proper.prop_get_phase(wfo)
    
    print(f"\n  1. PROPER 残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")
    
    # 2. 计算 PROPER 参考面相位
    proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    print(f"  2. PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")
    
    # 3. 重建绝对相位
    absolute_phase = proper_ref_phase + residual_phase
    
    print(f"  3. 重建绝对相位范围: [{np.min(absolute_phase):.6f}, {np.max(absolute_phase):.6f}] rad")
    
    # 4. 与 state.phase 比较
    state_phase = state_s4_entrance.phase
    
    print(f"  4. state.phase 范围: [{np.min(state_phase):.6f}, {np.max(state_phase):.6f}] rad")
    
    diff = absolute_phase - state_phase
    print(f"  5. 差异范围: [{np.min(diff):.9f}, {np.max(diff):.9f}] rad")
    print(f"     差异 RMS: {np.std(diff):.9f} rad = {np.std(diff)/(2*np.pi)*1000:.6f} milli-waves")
    
    # 5. 与 Pilot Beam 比较
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    print(f"\n  6. Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    
    # 在有效区域比较
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    
    diff_pilot = state_phase - pilot_phase
    diff_pilot_valid = diff_pilot[valid_mask]
    mean_diff = np.mean(diff_pilot_valid)
    diff_centered = diff_pilot_valid - mean_diff
    
    print(f"  7. state.phase vs Pilot Beam (有效区域):")
    print(f"     平均偏移: {mean_diff:.6f} rad = {mean_diff/(2*np.pi)*1000:.4f} milli-waves")
    print(f"     去偏移后 RMS: {np.std(diff_centered)/(2*np.pi)*1000:.4f} milli-waves")

print("\n" + "=" * 70)
print("检查 Surface 3 出射面（误差引入点）")
print("=" * 70)

# 找到 Surface 3 出射面
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
        break

if state_s3_exit is not None:
    wfo = state_s3_exit.proper_wfo
    grid_sampling = state_s3_exit.grid_sampling
    pb = state_s3_exit.pilot_beam_params
    
    print(f"\nSurface 3 出射面:")
    print(f"  reference_surface: {wfo.reference_surface}")
    
    state_converter = StateConverter(0.55)
    
    amplitude = proper.prop_get_amplitude(wfo)
    residual_phase = proper.prop_get_phase(wfo)
    proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
    absolute_phase = proper_ref_phase + residual_phase
    state_phase = state_s3_exit.phase
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    print(f"  PROPER 残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")
    print(f"  PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")
    print(f"  重建绝对相位范围: [{np.min(absolute_phase):.6f}, {np.max(absolute_phase):.6f}] rad")
    print(f"  state.phase 范围: [{np.min(state_phase):.6f}, {np.max(state_phase):.6f}] rad")
    
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    diff_pilot = state_phase - pilot_phase
    diff_pilot_valid = diff_pilot[valid_mask]
    mean_diff = np.mean(diff_pilot_valid)
    diff_centered = diff_pilot_valid - mean_diff
    
    print(f"\n  state.phase vs Pilot Beam (有效区域):")
    print(f"     平均偏移: {mean_diff:.6f} rad = {mean_diff/(2*np.pi)*1000:.4f} milli-waves")
    print(f"     去偏移后 RMS: {np.std(diff_centered)/(2*np.pi)*1000:.4f} milli-waves")
