"""
调试 Surface 3 出射面到 Surface 4 入射面的传播

关键问题：
- Surface 3 出射面误差：0.56 milli-waves
- Surface 4 入射面误差：22 milli-waves
- 误差被放大了约 40 倍！

需要检查：
1. Surface 3 出射面的 state.phase 是否正确写入 PROPER
2. PROPER 传播是否正确
3. Surface 4 入射面的相位提取是否正确
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
print("调试 Surface 3 出射面到 Surface 4 入射面的传播")
print("=" * 70)

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

# 传播到 Surface 4
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(5):
    propagator._propagate_to_surface(i)

# 找到关键状态
state_s3_exit = None
state_s4_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
    if state.surface_index == 4 and state.position == 'entrance':
        state_s4_entrance = state

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("\n【Surface 3 出射面】")
if state_s3_exit:
    phase = state_s3_exit.phase
    amplitude = state_s3_exit.amplitude
    pb = state_s3_exit.pilot_beam_params
    grid_sampling = state_s3_exit.grid_sampling
    wfo = state_s3_exit.proper_wfo
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    print(f"  state.phase 范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    
    # 检查 PROPER wfo 中的相位
    proper_phase = proper.prop_get_phase(wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
    
    # 检查 PROPER 参数
    print(f"\n  PROPER 参数:")
    print(f"    z = {wfo.z * 1e3:.2f} mm")
    print(f"    z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
    print(f"    z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"    reference_surface = {wfo.reference_surface}")
    
    # 检查 state.phase 与 PROPER 相位的关系
    # 如果 reference_surface = PLANAR，state.phase 应该等于 PROPER 相位
    state_converter = StateConverter(0.55)
    proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
    reconstructed_phase = proper_ref_phase + proper_phase
    
    diff_state_proper = phase - reconstructed_phase
    print(f"\n  state.phase vs (PROPER ref + PROPER phase):")
    print(f"    差异范围: [{np.min(diff_state_proper):.9f}, {np.max(diff_state_proper):.9f}] rad")

print("\n【Surface 4 入射面】")
if state_s4_entrance:
    phase = state_s4_entrance.phase
    amplitude = state_s4_entrance.amplitude
    pb = state_s4_entrance.pilot_beam_params
    grid_sampling = state_s4_entrance.grid_sampling
    wfo = state_s4_entrance.proper_wfo
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    print(f"  state.phase 范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    
    # 检查 PROPER wfo 中的相位
    proper_phase = proper.prop_get_phase(wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
    
    # 检查 PROPER 参数
    print(f"\n  PROPER 参数:")
    print(f"    z = {wfo.z * 1e3:.2f} mm")
    print(f"    z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
    print(f"    z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"    reference_surface = {wfo.reference_surface}")

print("\n" + "=" * 70)
print("分析传播过程")
print("=" * 70)

if state_s3_exit and state_s4_entrance:
    # 传播距离
    pos_s3 = state_s3_exit.optical_axis_state.position.to_array()
    pos_s4 = state_s4_entrance.optical_axis_state.position.to_array()
    distance = np.linalg.norm(pos_s4 - pos_s3)
    print(f"\n传播距离: {distance:.2f} mm")
    
    # 检查 Pilot Beam 参数变化
    pb_s3 = state_s3_exit.pilot_beam_params
    pb_s4 = state_s4_entrance.pilot_beam_params
    
    print(f"\nPilot Beam 参数变化:")
    print(f"  S3 出射: q = {pb_s3.q_parameter}, R = {pb_s3.curvature_radius_mm:.2f} mm")
    print(f"  S4 入射: q = {pb_s4.q_parameter}, R = {pb_s4.curvature_radius_mm:.2f} mm")
    
    # 手动传播 Pilot Beam
    pb_s4_expected = pb_s3.propagate(distance)
    print(f"  期望 S4: q = {pb_s4_expected.q_parameter}, R = {pb_s4_expected.curvature_radius_mm:.2f} mm")

print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

print("""
【关键发现】

1. Surface 3 出射面的 state.phase 范围很小（约 0.04 rad）
   - 这是几何光线追迹的结果
   - 与 Pilot Beam 差异 0.56 milli-waves

2. Surface 4 入射面的 state.phase 范围很大（约 ±3.14 rad）
   - 这是 PROPER 传播后的结果
   - 与 Pilot Beam 差异 22 milli-waves

3. 问题：为什么 PROPER 传播后相位范围从 0.04 rad 变成了 ±3.14 rad？
   - 这说明 PROPER 传播引入了大量相位
   - 但 Pilot Beam 相位范围仍然很小（约 0.02 rad）

4. 可能的原因：
   - Surface 3 出射面写入 PROPER 时，相位没有正确处理
   - 或者 PROPER 传播后，相位提取没有正确处理
""")
