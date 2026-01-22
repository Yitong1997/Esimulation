"""
调试实际传播流程

追踪 Surface 3 出射面到 Surface 4 入射面的实际传播
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
from hybrid_optical_propagation.data_models import GridSampling

print("=" * 70)
print("调试实际传播流程")
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

# 传播到 Surface 3 出射面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):  # 传播到 Surface 3
    propagator._propagate_to_surface(i)

# 找到 Surface 3 出射面状态
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
        break

print("\n【Surface 3 出射面状态】")
print(f"  state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")

wfo_s3 = state_s3_exit.proper_wfo
print(f"\n  PROPER wfo 参数:")
print(f"    z = {wfo_s3.z * 1e3:.2f} mm")
print(f"    z_w0 = {wfo_s3.z_w0 * 1e3:.2f} mm")
print(f"    reference_surface = {wfo_s3.reference_surface}")

proper_phase_s3 = proper.prop_get_phase(wfo_s3)
print(f"    prop_get_phase 范围: [{np.min(proper_phase_s3):.6f}, {np.max(proper_phase_s3):.6f}] rad")

# 保存 wfo 的副本用于比较
wfarr_s3_copy = wfo_s3.wfarr.copy()
z_s3 = wfo_s3.z
z_w0_s3 = wfo_s3.z_w0

print("\n" + "=" * 70)
print("执行传播到 Surface 4")
print("=" * 70)

# 继续传播到 Surface 4
propagator._propagate_to_surface(4)

# 找到 Surface 4 入射面状态
state_s4_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 4 and state.position == 'entrance':
        state_s4_entrance = state
        break

print("\n【Surface 4 入射面状态】")
print(f"  state.phase 范围: [{np.min(state_s4_entrance.phase):.6f}, {np.max(state_s4_entrance.phase):.6f}] rad")

wfo_s4 = state_s4_entrance.proper_wfo
print(f"\n  PROPER wfo 参数:")
print(f"    z = {wfo_s4.z * 1e3:.2f} mm")
print(f"    z_w0 = {wfo_s4.z_w0 * 1e3:.2f} mm")
print(f"    reference_surface = {wfo_s4.reference_surface}")

proper_phase_s4 = proper.prop_get_phase(wfo_s4)
print(f"    prop_get_phase 范围: [{np.min(proper_phase_s4):.6f}, {np.max(proper_phase_s4):.6f}] rad")

# 检查 wfo 是否被修改
print("\n" + "=" * 70)
print("检查 wfo 是否被共享")
print("=" * 70)

print(f"\nSurface 3 出射面 wfo 是否与 Surface 4 入射面 wfo 相同: {wfo_s3 is wfo_s4}")

# 检查 Surface 3 出射面的 wfo 是否被修改
print(f"\nSurface 3 出射面 wfo 当前状态:")
print(f"  z = {wfo_s3.z * 1e3:.2f} mm (原始: {z_s3 * 1e3:.2f} mm)")
print(f"  z_w0 = {wfo_s3.z_w0 * 1e3:.2f} mm (原始: {z_w0_s3 * 1e3:.2f} mm)")

# 检查 wfarr 是否被修改
wfarr_diff = np.max(np.abs(wfo_s3.wfarr - wfarr_s3_copy))
print(f"  wfarr 差异: {wfarr_diff:.9e}")

print("\n" + "=" * 70)
print("分析 Pilot Beam 参数")
print("=" * 70)

pb_s3 = state_s3_exit.pilot_beam_params
pb_s4 = state_s4_entrance.pilot_beam_params

print(f"\nSurface 3 出射面 Pilot Beam:")
print(f"  waist_position_mm = {pb_s3.waist_position_mm:.2f} mm")
print(f"  R = {pb_s3.curvature_radius_mm:.2f} mm")

print(f"\nSurface 4 入射面 Pilot Beam:")
print(f"  waist_position_mm = {pb_s4.waist_position_mm:.2f} mm")
print(f"  R = {pb_s4.curvature_radius_mm:.2f} mm")

# 计算 Pilot Beam 相位
grid_sampling = state_s4_entrance.grid_sampling
pilot_phase_s4 = pb_s4.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase_s4):.6f}, {np.max(pilot_phase_s4):.6f}] rad")

print("\n" + "=" * 70)
print("分析解包裹过程")
print("=" * 70)

state_converter = StateConverter(0.55)

# 计算 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo_s4, grid_sampling)
print(f"\nPROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建绝对相位
wrapped_phase = proper_ref_phase + proper_phase_s4
print(f"重建的绝对相位（折叠）范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")

# 解包裹
phase_diff = wrapped_phase - pilot_phase_s4
print(f"相位差（wrapped - pilot）范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad")

unwrapped_phase = pilot_phase_s4 + np.angle(np.exp(1j * phase_diff))
print(f"解包裹后相位范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

# 检查相位差是否超过 π
max_phase_diff = np.max(np.abs(phase_diff))
print(f"\n相位差最大绝对值: {max_phase_diff:.6f} rad")
print(f"是否超过 π: {max_phase_diff > np.pi}")

if max_phase_diff > np.pi:
    print("\n【问题】相位差超过 π，解包裹失败！")
