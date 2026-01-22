"""
逐步调试数据流

详细追踪从 Surface 3 出射面到 Surface 4 入射面的每一步数据流
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
from hybrid_optical_propagation.data_models import GridSampling, PilotBeamParams

print("=" * 70)
print("逐步调试数据流")
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

print("\n" + "=" * 70)
print("【步骤 1】Surface 3 出射面状态")
print("=" * 70)

print(f"\n1.1 state.amplitude 范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")
print(f"1.2 state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")

pb_s3 = state_s3_exit.pilot_beam_params
print(f"\n1.3 Pilot Beam 参数:")
print(f"    waist_radius_mm = {pb_s3.waist_radius_mm:.4f} mm")
print(f"    waist_position_mm = {pb_s3.waist_position_mm:.4f} mm")
print(f"    curvature_radius_mm = {pb_s3.curvature_radius_mm:.4f} mm")
print(f"    rayleigh_length_mm = {pb_s3.rayleigh_length_mm:.4f} mm")

wfo_s3 = state_s3_exit.proper_wfo
print(f"\n1.4 PROPER wfo 参数:")
print(f"    z = {wfo_s3.z * 1e3:.4f} mm")
print(f"    z_w0 = {wfo_s3.z_w0 * 1e3:.4f} mm")
print(f"    z - z_w0 = {(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm")
print(f"    w0 = {wfo_s3.w0 * 1e3:.4f} mm")
print(f"    z_Rayleigh = {wfo_s3.z_Rayleigh * 1e3:.4f} mm")
print(f"    reference_surface = {wfo_s3.reference_surface}")

# 检查 wfarr 内容
proper_amp_s3 = proper.prop_get_amplitude(wfo_s3)
proper_phase_s3 = proper.prop_get_phase(wfo_s3)
print(f"\n1.5 PROPER wfarr 内容:")
print(f"    prop_get_amplitude 范围: [{np.min(proper_amp_s3):.6f}, {np.max(proper_amp_s3):.6f}]")
print(f"    prop_get_phase 范围: [{np.min(proper_phase_s3):.6f}, {np.max(proper_phase_s3):.6f}] rad")

print("\n" + "=" * 70)
print("【步骤 2】计算传播距离")
print("=" * 70)

# 使用固定的传播距离（从 Surface 3 到 Surface 4 的距离）
# 根据 ZMX 文件，Surface 3 的厚度是 100mm
distance_mm = 100.0

print(f"\n2.1 传播距离: {distance_mm:.4f} mm")

print("\n" + "=" * 70)
print("【步骤 3】执行 PROPER 传播")
print("=" * 70)

# 保存传播前的状态
wfarr_before = wfo_s3.wfarr.copy()
z_before = wfo_s3.z
z_w0_before = wfo_s3.z_w0
ref_surface_before = wfo_s3.reference_surface

print(f"\n3.1 传播前:")
print(f"    z = {z_before * 1e3:.4f} mm")
print(f"    z_w0 = {z_w0_before * 1e3:.4f} mm")
print(f"    reference_surface = {ref_surface_before}")

# 执行传播
distance_m = distance_mm * 1e-3
proper.prop_propagate(wfo_s3, distance_m)

print(f"\n3.2 传播后:")
print(f"    z = {wfo_s3.z * 1e3:.4f} mm")
print(f"    z_w0 = {wfo_s3.z_w0 * 1e3:.4f} mm")
print(f"    z - z_w0 = {(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm")
print(f"    reference_surface = {wfo_s3.reference_surface}")

# 检查传播后的 wfarr
proper_amp_after = proper.prop_get_amplitude(wfo_s3)
proper_phase_after = proper.prop_get_phase(wfo_s3)
print(f"\n3.3 传播后 PROPER wfarr 内容:")
print(f"    prop_get_amplitude 范围: [{np.min(proper_amp_after):.6f}, {np.max(proper_amp_after):.6f}]")
print(f"    prop_get_phase 范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

print("\n" + "=" * 70)
print("【步骤 4】更新 Pilot Beam 参数")
print("=" * 70)

pb_s4 = pb_s3.propagate(distance_mm)
print(f"\n4.1 传播后 Pilot Beam 参数:")
print(f"    waist_radius_mm = {pb_s4.waist_radius_mm:.4f} mm")
print(f"    waist_position_mm = {pb_s4.waist_position_mm:.4f} mm")
print(f"    curvature_radius_mm = {pb_s4.curvature_radius_mm:.4f} mm")
print(f"    rayleigh_length_mm = {pb_s4.rayleigh_length_mm:.4f} mm")

# 计算 Pilot Beam 相位
grid_sampling = state_s3_exit.grid_sampling
pilot_phase_s4 = pb_s4.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
print(f"\n4.2 Pilot Beam 相位范围: [{np.min(pilot_phase_s4):.6f}, {np.max(pilot_phase_s4):.6f}] rad")

print("\n" + "=" * 70)
print("【步骤 5】从 PROPER 提取相位（proper_to_amplitude_phase）")
print("=" * 70)

state_converter = StateConverter(0.55)

# 5.1 计算 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo_s3, grid_sampling)
print(f"\n5.1 PROPER 参考面相位:")
print(f"    reference_surface = {wfo_s3.reference_surface}")
print(f"    R_ref = z - z_w0 = {(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm")
print(f"    参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 5.2 重建绝对相位
wrapped_phase = proper_ref_phase + proper_phase_after
print(f"\n5.2 重建的绝对相位（折叠）:")
print(f"    wrapped_phase = proper_ref_phase + proper_phase")
print(f"    范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")

# 5.3 计算相位差
phase_diff = wrapped_phase - pilot_phase_s4
print(f"\n5.3 相位差（wrapped - pilot）:")
print(f"    范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad")
print(f"    最大绝对值: {np.max(np.abs(phase_diff)):.6f} rad")
print(f"    是否超过 π: {np.max(np.abs(phase_diff)) > np.pi}")

# 5.4 解包裹
unwrapped_phase = pilot_phase_s4 + np.angle(np.exp(1j * phase_diff))
print(f"\n5.4 解包裹后相位:")
print(f"    范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

print("\n" + "=" * 70)
print("【分析】问题根源")
print("=" * 70)

print(f"""
问题分析：

1. Surface 3 出射面的 state.phase 范围很小: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad
   这是因为 state.phase 存储的是相对于 Pilot Beam 的残差相位

2. PROPER 传播后，prop_get_phase 返回的相位范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad
   这是 PROPER 内部存储的相位（相对于 PROPER 参考面的残差）

3. PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad
   R_ref = z - z_w0 = {(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm

4. Pilot Beam 相位范围: [{np.min(pilot_phase_s4):.6f}, {np.max(pilot_phase_s4):.6f}] rad
   R_pilot = {pb_s4.curvature_radius_mm:.4f} mm

5. 关键问题：PROPER 参考面曲率半径 ({(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm) 
   与 Pilot Beam 曲率半径 ({pb_s4.curvature_radius_mm:.4f} mm) 差异很大！
   
   这导致 wrapped_phase 和 pilot_phase 差异超过 π，解包裹失败。
""")

print("\n" + "=" * 70)
print("【验证】检查曲率半径计算")
print("=" * 70)

# 检查 Pilot Beam 曲率半径计算
z_mm = -pb_s4.waist_position_mm  # 当前位置相对于束腰的距离
z_R_mm = pb_s4.rayleigh_length_mm
R_strict = z_mm * (1 + (z_R_mm / z_mm)**2) if abs(z_mm) > 1e-10 else np.inf
R_approx = z_mm  # 远场近似

print(f"\nPilot Beam 曲率半径计算:")
print(f"  z (当前位置相对于束腰) = {z_mm:.4f} mm")
print(f"  z_R (瑞利长度) = {z_R_mm:.4f} mm")
print(f"  R_strict (严格公式) = {R_strict:.4f} mm")
print(f"  R_approx (远场近似) = {R_approx:.4f} mm")
print(f"  pb_s4.curvature_radius_mm = {pb_s4.curvature_radius_mm:.4f} mm")

print(f"\nPROPER 参考面曲率半径:")
print(f"  z = {wfo_s3.z * 1e3:.4f} mm")
print(f"  z_w0 = {wfo_s3.z_w0 * 1e3:.4f} mm")
print(f"  R_ref = z - z_w0 = {(wfo_s3.z - wfo_s3.z_w0) * 1e3:.4f} mm")

print("\n" + "=" * 70)
print("【检查】wfo.z_w0 是否正确设置")
print("=" * 70)

# 检查 _sync_gaussian_params 的逻辑
# wfo.z_w0 = wfo.z + pilot_beam_params.waist_position_mm * 1e-3
# 
# 对于 Surface 3 出射面：
# - pb_s3.waist_position_mm = -40 mm（束腰在当前位置之前 40mm）
# - wfo.z 应该是传播后的位置
# - wfo.z_w0 应该是 wfo.z + (-40) * 1e-3 = wfo.z - 0.04

print(f"\n检查 Surface 3 出射面的 wfo 参数设置:")
print(f"  pb_s3.waist_position_mm = {pb_s3.waist_position_mm:.4f} mm")
print(f"  期望的 z_w0 = z + waist_position_mm = {z_before * 1e3:.4f} + {pb_s3.waist_position_mm:.4f} = {z_before * 1e3 + pb_s3.waist_position_mm:.4f} mm")
print(f"  实际的 z_w0 = {z_w0_before * 1e3:.4f} mm")

# 传播后
print(f"\n传播后:")
print(f"  pb_s4.waist_position_mm = {pb_s4.waist_position_mm:.4f} mm")
print(f"  wfo.z = {wfo_s3.z * 1e3:.4f} mm")
print(f"  期望的 z_w0 = z + waist_position_mm = {wfo_s3.z * 1e3:.4f} + {pb_s4.waist_position_mm:.4f} = {wfo_s3.z * 1e3 + pb_s4.waist_position_mm:.4f} mm")
print(f"  实际的 z_w0 = {wfo_s3.z_w0 * 1e3:.4f} mm")
print(f"  注意：prop_propagate 不会更新 z_w0，它保持不变")
