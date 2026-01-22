"""
调试 Surface 3 出射面到 Surface 4 入射面的数据流

关键检查点：
1. Surface 3 出射面的 state.phase 范围
2. 写入 PROPER 后的 wfarr 相位范围
3. 传播距离计算
4. 传播后的 wfo.z 和 wfo.z_w0
5. 从 PROPER 读取的相位范围
6. Pilot Beam 解包裹后的相位范围
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

from hybrid_optical_propagation.data_models import (
    PilotBeamParams, GridSampling, PropagationState, SourceDefinition
)
from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.free_space_propagator import (
    FreeSpacePropagator, compute_propagation_distance
)
from sequential_system.coordinate_tracking import (
    OpticalAxisState, Position3D, RayDirection
)

print("=" * 70)
print("调试 Surface 3 出射面到 Surface 4 入射面的数据流")
print("=" * 70)

# 参数设置（模拟实际系统）
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 50.0
w0_mm = 5.0

# 创建 StateConverter
converter = StateConverter(wavelength_um)

# 创建 GridSampling
grid_sampling = GridSampling.create(grid_size, physical_size_mm, beam_ratio=0.25)

print(f"\n参数:")
print(f"  波长: {wavelength_um} μm")
print(f"  网格大小: {grid_size}")
print(f"  物理尺寸: {physical_size_mm} mm")
print(f"  采样间隔: {grid_sampling.sampling_mm:.6f} mm")

# ============================================================================
# 模拟 Surface 3 出射面的状态
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 1】模拟 Surface 3 出射面的状态")
print("=" * 70)

# 创建坐标网格
n = grid_size
sampling_mm = grid_sampling.sampling_mm
coords_mm = (np.arange(n) - n // 2) * sampling_mm
X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
R_sq_mm = X_mm**2 + Y_mm**2

# 模拟高斯振幅
amplitude = np.exp(-R_sq_mm / (2 * w0_mm)**2)

# 模拟小的相位（类似实际系统中 Surface 3 出射面的相位）
# 假设相位范围是 [-0.038, 0] rad
phase = -0.038 * (R_sq_mm / np.max(R_sq_mm))

print(f"\nSurface 3 出射面状态:")
print(f"  振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
print(f"  相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")

# 创建 Pilot Beam 参数（模拟 Surface 3 出射面的参数）
# 假设束腰在当前位置之前 40mm
pilot_beam_s3_exit = PilotBeamParams.from_gaussian_source(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=-40.0,  # 束腰在当前位置之前 40mm
)

print(f"\nPilot Beam 参数 (Surface 3 出射面):")
print(f"  束腰半径: {pilot_beam_s3_exit.waist_radius_mm:.4f} mm")
print(f"  束腰位置: {pilot_beam_s3_exit.waist_position_mm:.4f} mm")
print(f"  曲率半径: {pilot_beam_s3_exit.curvature_radius_mm:.4f} mm")
print(f"  瑞利长度: {pilot_beam_s3_exit.rayleigh_length_mm:.4f} mm")

# ============================================================================
# 写入 PROPER
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 2】写入 PROPER")
print("=" * 70)

wfo = converter.amplitude_phase_to_proper(
    amplitude, phase, grid_sampling, pilot_beam_s3_exit
)

print(f"\nPROPER wfo 参数:")
print(f"  z = {wfo.z * 1e3:.4f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.4f} mm")
print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.4f} mm")
print(f"  reference_surface = {wfo.reference_surface}")
print(f"  beam_type_old = {wfo.beam_type_old}")

# 检查写入后的相位
wfarr_phase = np.angle(proper.prop_shift_center(wfo.wfarr))
print(f"\n写入后 wfarr 相位范围: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")

# 使用 prop_get_phase 读取
proper_phase = proper.prop_get_phase(wfo)
print(f"prop_get_phase 范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# ============================================================================
# 模拟传播距离计算
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 3】计算传播距离")
print("=" * 70)

# 模拟 Surface 3 出射面和 Surface 4 入射面的位置
# 假设 Surface 3 在 z=40mm，Surface 4 在 z=140mm（距离 100mm）
s3_exit_pos = np.array([0.0, 0.0, 40.0])
s4_entrance_pos = np.array([0.0, 0.0, 140.0])
s3_exit_dir = np.array([0.0, 0.0, 1.0])

distance_mm = compute_propagation_distance(s3_exit_pos, s4_entrance_pos, s3_exit_dir)
print(f"\n传播距离: {distance_mm:.4f} mm")

# ============================================================================
# 执行 PROPER 传播
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 4】执行 PROPER 传播")
print("=" * 70)

print(f"\n传播前:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

distance_m = distance_mm * 1e-3
proper.prop_propagate(wfo, distance_m)

print(f"\n传播后:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.4f} mm")
print(f"  reference_surface = {wfo.reference_surface}")
print(f"  beam_type_old = {wfo.beam_type_old}")

# 检查传播后的相位
proper_phase_after = proper.prop_get_phase(wfo)
print(f"\n传播后 prop_get_phase 范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

# ============================================================================
# 更新 Pilot Beam 参数
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 5】更新 Pilot Beam 参数")
print("=" * 70)

pilot_beam_s4_entrance = pilot_beam_s3_exit.propagate(distance_mm)

print(f"\nPilot Beam 参数 (Surface 4 入射面):")
print(f"  束腰半径: {pilot_beam_s4_entrance.waist_radius_mm:.4f} mm")
print(f"  束腰位置: {pilot_beam_s4_entrance.waist_position_mm:.4f} mm")
print(f"  曲率半径: {pilot_beam_s4_entrance.curvature_radius_mm:.4f} mm")
print(f"  瑞利长度: {pilot_beam_s4_entrance.rayleigh_length_mm:.4f} mm")

# 计算 Pilot Beam 参考相位
pilot_phase = pilot_beam_s4_entrance.compute_phase_grid(grid_size, physical_size_mm)
print(f"\nPilot Beam 参考相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")

# ============================================================================
# 从 PROPER 读取并解包裹
# ============================================================================
print("\n" + "=" * 70)
print("【步骤 6】从 PROPER 读取并解包裹")
print("=" * 70)

# 计算 PROPER 参考面相位
proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"\nPROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建绝对相位
wrapped_phase = proper_ref_phase + proper_phase_after
print(f"重建的折叠相位范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")

# 使用 Pilot Beam 解包裹
phase_diff = wrapped_phase - pilot_phase
unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
print(f"解包裹后相位范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

# ============================================================================
# 分析问题
# ============================================================================
print("\n" + "=" * 70)
print("【分析】")
print("=" * 70)

# 检查相位差是否超过 π
max_phase_diff = np.max(np.abs(phase_diff))
print(f"\n相位差最大值: {max_phase_diff:.4f} rad = {max_phase_diff / np.pi:.4f} π")

if max_phase_diff > np.pi:
    print("⚠️ 相位差超过 π，解包裹可能失败！")
else:
    print("✓ 相位差在 π 范围内，解包裹应该正确")

# 检查 PROPER 参考面类型
print(f"\n传播后 PROPER 参考面类型: {wfo.reference_surface}")
if wfo.reference_surface == "SPHERI":
    R_ref = wfo.z - wfo.z_w0
    print(f"PROPER 参考球面曲率半径: {R_ref * 1e3:.4f} mm")
    print(f"Pilot Beam 曲率半径: {pilot_beam_s4_entrance.curvature_radius_mm:.4f} mm")
    
    # 两者的差异
    if not np.isinf(pilot_beam_s4_entrance.curvature_radius_mm):
        diff_ratio = abs(R_ref * 1e3 - pilot_beam_s4_entrance.curvature_radius_mm) / abs(pilot_beam_s4_entrance.curvature_radius_mm)
        print(f"曲率半径差异: {diff_ratio * 100:.2f}%")

# 检查是否在瑞利距离内
z_diff = abs(wfo.z - wfo.z_w0)
z_R = wfo.z_Rayleigh
rayleigh_factor = proper.rayleigh_factor
threshold = rayleigh_factor * z_R

print(f"\n|z - z_w0| = {z_diff * 1e3:.4f} mm")
print(f"rayleigh_factor * z_R = {threshold * 1e3:.4f} mm")
print(f"在瑞利距离内: {z_diff < threshold}")
