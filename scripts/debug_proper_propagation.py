"""
调试 PROPER 传播过程

检查 Surface 3 出射面到 Surface 4 入射面的 PROPER 传播
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

from hybrid_optical_propagation.data_models import GridSampling, PilotBeamParams
from hybrid_optical_propagation.state_converter import StateConverter

print("=" * 70)
print("调试 PROPER 传播过程")
print("=" * 70)

# 参数设置
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0
sampling_mm = physical_size_mm / grid_size
w0_mm = 5.0

grid_sampling = GridSampling(
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
    sampling_mm=sampling_mm,
    beam_ratio=0.25,
)

# Surface 3 出射面的 Pilot Beam 参数
# 当前在 z=40mm，束腰在 z=0
pb_s3_exit = PilotBeamParams.from_gaussian_source(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=-40.0,  # 束腰在当前位置之前 40mm
)

print(f"\nSurface 3 出射面 Pilot Beam:")
print(f"  waist_position_mm = {pb_s3_exit.waist_position_mm:.2f} mm")
print(f"  R = {pb_s3_exit.curvature_radius_mm:.2f} mm")

# 创建测试相位（模拟 Surface 3 出射面的相位）
X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
r_sq = X_mm**2 + Y_mm**2

# 小的像差相位
test_phase = -0.04 * r_sq / (physical_size_mm/2)**2

# 高斯振幅
test_amplitude = np.exp(-r_sq / (2 * (w0_mm * 2)**2))

print(f"\n测试相位范围: [{np.min(test_phase):.6f}, {np.max(test_phase):.6f}] rad")

# 写入 PROPER
state_converter = StateConverter(wavelength_um)
wfo = state_converter.amplitude_phase_to_proper(
    test_amplitude,
    test_phase,
    grid_sampling,
    pb_s3_exit,
)

print(f"\n写入后 PROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

# 检查写入后的相位
proper_phase_before = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_before):.6f}, {np.max(proper_phase_before):.6f}] rad")

print("\n" + "=" * 70)
print("执行 PROPER 传播 (40 mm)")
print("=" * 70)

# 传播 40 mm
distance_mm = 40.0
distance_m = distance_mm * 1e-3
proper.prop_propagate(wfo, distance_m)

print(f"\n传播后 PROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

# 检查传播后的相位
proper_phase_after = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

# Surface 4 入射面的 Pilot Beam 参数
pb_s4_entrance = pb_s3_exit.propagate(distance_mm)

print(f"\nSurface 4 入射面 Pilot Beam:")
print(f"  waist_position_mm = {pb_s4_entrance.waist_position_mm:.2f} mm")
print(f"  R = {pb_s4_entrance.curvature_radius_mm:.2f} mm")

# 计算 Pilot Beam 相位
pilot_phase_s4 = pb_s4_entrance.compute_phase_grid(grid_size, physical_size_mm)
print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase_s4):.6f}, {np.max(pilot_phase_s4):.6f}] rad")

# 计算 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"  PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建绝对相位
wrapped_phase = proper_ref_phase + proper_phase_after
print(f"\n重建的绝对相位（折叠）范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")

# 解包裹
phase_diff = wrapped_phase - pilot_phase_s4
print(f"相位差（wrapped - pilot）范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad")

unwrapped_phase = pilot_phase_s4 + np.angle(np.exp(1j * phase_diff))
print(f"解包裹后相位范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

print("\n" + "=" * 70)
print("问题分析")
print("=" * 70)

# 检查相位差是否超过 π
max_phase_diff = np.max(np.abs(phase_diff))
print(f"\n相位差最大绝对值: {max_phase_diff:.6f} rad")
print(f"是否超过 π: {max_phase_diff > np.pi}")

if max_phase_diff > np.pi:
    print("\n【问题】相位差超过 π，解包裹公式失效！")
    print("解包裹公式 T_unwrapped = T_pilot + angle(T - T_pilot) 只能处理 |T - T_pilot| < π 的情况")
    
    # 分析原因
    print("\n【原因分析】")
    print(f"  PROPER 传播后的相位范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")
    print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase_s4):.6f}, {np.max(pilot_phase_s4):.6f}] rad")
    
    # 检查 PROPER 传播是否引入了额外的相位
    print("\n【PROPER 传播引入的相位】")
    print(f"  传播前相位范围: [{np.min(proper_phase_before):.6f}, {np.max(proper_phase_before):.6f}] rad")
    print(f"  传播后相位范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")
    
    # 理论上，近场传播不应该引入大的相位变化
    # 但 PROPER 可能使用了不同的参考面
    
print("\n" + "=" * 70)
print("检查 PROPER 内部状态")
print("=" * 70)

print(f"\nPROPER wfo 属性:")
print(f"  lamda = {wfo.lamda * 1e6:.4f} μm")
print(f"  w0 = {wfo.w0 * 1e3:.4f} mm")
print(f"  z = {wfo.z * 1e3:.4f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.4f} mm")
print(f"  reference_surface = {wfo.reference_surface}")
print(f"  beam_type_old = {wfo.beam_type_old}")

# 检查 z - z_w0
z_minus_z_w0 = (wfo.z - wfo.z_w0) * 1e3
print(f"\n  z - z_w0 = {z_minus_z_w0:.4f} mm")
print(f"  这是 PROPER 用于计算参考球面曲率的距离")
