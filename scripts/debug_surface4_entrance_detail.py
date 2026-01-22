"""
详细调试 Surface 4 入射面的相位问题

Surface 4 入射面是第一个出现大误差的位置（22 milli-waves RMS）
需要理解：
1. Surface 3 出射面的相位是否正确
2. 从 Surface 3 出射面到 Surface 4 入射面的传播是否正确
3. Surface 4 入射面的 state.phase 与 Pilot Beam 为什么有差异
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
print("详细调试 Surface 4 入射面的相位问题")
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

# 传播到 Surface 4
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(5):  # 0, 1, 2, 3, 4
    propagator._propagate_to_surface(i)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

# 找到关键状态
state_s3_exit = None
state_s4_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
    if state.surface_index == 4 and state.position == 'entrance':
        state_s4_entrance = state

print("\n【Surface 3 出射面】")
if state_s3_exit:
    phase = state_s3_exit.phase
    amplitude = state_s3_exit.amplitude
    pb = state_s3_exit.pilot_beam_params
    grid_sampling = state_s3_exit.grid_sampling
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    diff = phase - pilot_phase
    diff_valid = diff[valid_mask]
    mean_diff = np.mean(diff_valid)
    diff_centered = diff_valid - mean_diff
    
    print(f"  state.phase 范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    print(f"  差异（去偏移后）RMS: {np.std(diff_centered)/(2*np.pi)*1000:.4f} milli-waves")
    print(f"  Pilot Beam R: {pb.curvature_radius_mm:.2f} mm")

print("\n【Surface 4 入射面】")
if state_s4_entrance:
    phase = state_s4_entrance.phase
    amplitude = state_s4_entrance.amplitude
    pb = state_s4_entrance.pilot_beam_params
    grid_sampling = state_s4_entrance.grid_sampling
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    diff = phase - pilot_phase
    diff_valid = diff[valid_mask]
    mean_diff = np.mean(diff_valid)
    diff_centered = diff_valid - mean_diff
    
    print(f"  state.phase 范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")
    print(f"  差异（去偏移后）RMS: {np.std(diff_centered)/(2*np.pi)*1000:.4f} milli-waves")
    print(f"  Pilot Beam R: {pb.curvature_radius_mm:.2f} mm")

print("\n" + "=" * 70)
print("分析传播过程")
print("=" * 70)

# Surface 3 出射面到 Surface 4 入射面是自由空间传播
# 检查传播距离
if state_s3_exit and state_s4_entrance:
    pos_s3 = state_s3_exit.optical_axis_state.position.to_array()
    pos_s4 = state_s4_entrance.optical_axis_state.position.to_array()
    dir_s3 = state_s3_exit.optical_axis_state.direction.to_array()
    
    displacement = pos_s4 - pos_s3
    distance = np.linalg.norm(displacement)
    
    print(f"\n从 Surface 3 出射面到 Surface 4 入射面:")
    print(f"  位置 S3: {pos_s3}")
    print(f"  位置 S4: {pos_s4}")
    print(f"  方向 S3: {dir_s3}")
    print(f"  传播距离: {distance:.2f} mm")

print("\n" + "=" * 70)
print("检查 PROPER 传播是否正确")
print("=" * 70)

# 从 Surface 3 出射面的 PROPER wfo 手动传播到 Surface 4 入射面
# 然后比较结果

if state_s3_exit and state_s4_entrance:
    # 复制 Surface 3 出射面的 wfo
    wfo_s3 = state_s3_exit.proper_wfo
    
    # 获取传播距离
    distance_mm = 40.0  # 从上面的计算
    distance_m = distance_mm * 1e-3
    
    # 创建一个新的 wfo 用于测试
    import copy
    # 不能直接复制 wfo，需要重新创建
    
    # 检查 Surface 3 出射面的 PROPER 状态
    print(f"\nSurface 3 出射面 PROPER 状态:")
    print(f"  z = {wfo_s3.z * 1e3:.2f} mm")
    print(f"  z_w0 = {wfo_s3.z_w0 * 1e3:.2f} mm")
    print(f"  reference_surface = {wfo_s3.reference_surface}")
    
    # 检查 Surface 4 入射面的 PROPER 状态
    wfo_s4 = state_s4_entrance.proper_wfo
    print(f"\nSurface 4 入射面 PROPER 状态:")
    print(f"  z = {wfo_s4.z * 1e3:.2f} mm")
    print(f"  z_w0 = {wfo_s4.z_w0 * 1e3:.2f} mm")
    print(f"  reference_surface = {wfo_s4.reference_surface}")

print("\n" + "=" * 70)
print("检查相位折叠问题")
print("=" * 70)

if state_s4_entrance:
    phase = state_s4_entrance.phase
    
    # 检查相位是否有折叠
    print(f"\nSurface 4 入射面相位分析:")
    print(f"  相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
    print(f"  相位范围（波长数）: [{np.min(phase)/(2*np.pi):.4f}, {np.max(phase)/(2*np.pi):.4f}] waves")
    
    # 检查是否有 2π 跳变
    phase_diff_x = np.abs(np.diff(phase, axis=1))
    phase_diff_y = np.abs(np.diff(phase, axis=0))
    max_diff = max(np.max(phase_diff_x), np.max(phase_diff_y))
    
    print(f"  相邻像素最大相位差: {max_diff:.4f} rad = {max_diff/(2*np.pi):.4f} waves")
    print(f"  是否有 2π 跳变: {'是' if max_diff > np.pi else '否'}")

print("\n" + "=" * 70)
print("问题定位")
print("=" * 70)

print("""
【关键发现】

1. Surface 3 出射面的 state.phase 与 Pilot Beam 差异很小（< 1 milli-wave）
   - 这说明几何光线追迹的结果是正确的

2. Surface 4 入射面的 state.phase 与 Pilot Beam 差异很大（22 milli-waves）
   - 这说明从 Surface 3 出射面到 Surface 4 入射面的传播出了问题

3. 传播过程是自由空间传播（FreeSpacePropagator）
   - 使用 PROPER 的 prop_propagate
   - 然后使用 proper_to_amplitude_phase 提取相位

4. 可能的问题：
   - PROPER 传播本身是正确的（之前已验证）
   - 问题可能在 proper_to_amplitude_phase 的相位提取
   - 或者在 Pilot Beam 参数的更新
""")

# 检查 Pilot Beam 参数是否正确更新
print("\n【检查 Pilot Beam 参数更新】")
if state_s3_exit and state_s4_entrance:
    pb_s3 = state_s3_exit.pilot_beam_params
    pb_s4 = state_s4_entrance.pilot_beam_params
    
    print(f"\nSurface 3 出射面 Pilot Beam:")
    print(f"  q = {pb_s3.q_parameter}")
    print(f"  R = {pb_s3.curvature_radius_mm:.2f} mm")
    print(f"  w = {pb_s3.spot_size_mm:.4f} mm")
    print(f"  z0 = {pb_s3.waist_position_mm:.2f} mm")
    
    print(f"\nSurface 4 入射面 Pilot Beam:")
    print(f"  q = {pb_s4.q_parameter}")
    print(f"  R = {pb_s4.curvature_radius_mm:.2f} mm")
    print(f"  w = {pb_s4.spot_size_mm:.4f} mm")
    print(f"  z0 = {pb_s4.waist_position_mm:.2f} mm")
    
    # 手动计算传播后的 Pilot Beam 参数
    pb_s4_expected = pb_s3.propagate(40.0)  # 传播 40 mm
    print(f"\n期望的 Surface 4 入射面 Pilot Beam（从 S3 传播 40mm）:")
    print(f"  q = {pb_s4_expected.q_parameter}")
    print(f"  R = {pb_s4_expected.curvature_radius_mm:.2f} mm")
    print(f"  w = {pb_s4_expected.spot_size_mm:.4f} mm")
    print(f"  z0 = {pb_s4_expected.waist_position_mm:.2f} mm")
