"""
调试 Surface 3 出射面的 PROPER 写入和读取

检查：
1. Surface 3 出射面写入 PROPER 时是否正确
2. 写入后立即读取是否能还原
3. 传播后读取是否正确
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
print("调试 Surface 3 出射面的 PROPER 写入和读取")
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
print("【检查 1】Surface 3 出射面的 state 数据")
print("=" * 70)

print(f"\nstate.amplitude 范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")
print(f"state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")

# 检查中心点
n = state_s3_exit.amplitude.shape[0]
center = n // 2
print(f"\n中心点 [{center}, {center}]:")
print(f"  amplitude = {state_s3_exit.amplitude[center, center]:.6f}")
print(f"  phase = {state_s3_exit.phase[center, center]:.6f} rad")

print("\n" + "=" * 70)
print("【检查 2】Surface 3 出射面的 wfo 数据")
print("=" * 70)

wfo = state_s3_exit.proper_wfo
print(f"\nwfo 参数:")
print(f"  z = {wfo.z * 1e3:.4f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  reference_surface = {wfo.reference_surface}")


# 直接检查 wfarr
print(f"\nwfarr 形状: {wfo.wfarr.shape}")
print(f"wfarr 类型: {wfo.wfarr.dtype}")

# wfarr 是 FFT 坐标系，需要 shift 回来
wfarr_centered = proper.prop_shift_center(wfo.wfarr)
wfarr_amp = np.abs(wfarr_centered)
wfarr_phase = np.angle(wfarr_centered)

print(f"\nwfarr（shift 后）:")
print(f"  振幅范围: [{np.min(wfarr_amp):.6f}, {np.max(wfarr_amp):.6f}]")
print(f"  相位范围: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")
print(f"  中心点振幅: {wfarr_amp[center, center]:.6f}")
print(f"  中心点相位: {wfarr_phase[center, center]:.6f} rad")

# 使用 prop_get_amplitude/phase
proper_amp = proper.prop_get_amplitude(wfo)
proper_phase = proper.prop_get_phase(wfo)

print(f"\nprop_get_amplitude/phase:")
print(f"  振幅范围: [{np.min(proper_amp):.6f}, {np.max(proper_amp):.6f}]")
print(f"  相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
print(f"  中心点振幅: {proper_amp[center, center]:.6f}")
print(f"  中心点相位: {proper_phase[center, center]:.6f} rad")

print("\n" + "=" * 70)
print("【检查 3】比较 state 和 wfo 的数据")
print("=" * 70)

# 对于 PLANAR 参考面，wfarr 应该直接等于 amplitude * exp(1j * phase)
# 检查是否一致

# 计算期望的 wfarr（如果写入正确）
expected_complex = state_s3_exit.amplitude * np.exp(1j * state_s3_exit.phase)
expected_amp = np.abs(expected_complex)
expected_phase = np.angle(expected_complex)

print(f"\n期望的复振幅（从 state 计算）:")
print(f"  振幅范围: [{np.min(expected_amp):.6f}, {np.max(expected_amp):.6f}]")
print(f"  相位范围: [{np.min(expected_phase):.6f}, {np.max(expected_phase):.6f}] rad")

# 比较
amp_diff = np.max(np.abs(proper_amp - expected_amp))
phase_diff_raw = proper_phase - expected_phase
# 处理相位差的 2π 周期性
phase_diff = np.angle(np.exp(1j * phase_diff_raw))
phase_diff_max = np.max(np.abs(phase_diff))

print(f"\n比较结果:")
print(f"  振幅最大差异: {amp_diff:.9e}")
print(f"  相位最大差异: {phase_diff_max:.9e} rad")

if amp_diff < 1e-6 and phase_diff_max < 1e-6:
    print("  ✓ wfo 数据与 state 数据一致")
else:
    print("  ✗ wfo 数据与 state 数据不一致！")

print("\n" + "=" * 70)
print("【检查 4】手动写入和读取测试")
print("=" * 70)

# 使用 StateConverter 手动写入和读取
state_converter = StateConverter(0.55)
grid_sampling = state_s3_exit.grid_sampling
pb = state_s3_exit.pilot_beam_params

print(f"\nPilot Beam 参数:")
print(f"  waist_position_mm = {pb.waist_position_mm:.4f} mm")
print(f"  curvature_radius_mm = {pb.curvature_radius_mm:.4f} mm")
print(f"  rayleigh_length_mm = {pb.rayleigh_length_mm:.4f} mm")

# 手动创建新的 wfo 并写入
print("\n手动写入测试:")
new_wfo = state_converter.amplitude_phase_to_proper(
    state_s3_exit.amplitude,
    state_s3_exit.phase,
    grid_sampling,
    pb,
)

print(f"  new_wfo.z = {new_wfo.z * 1e3:.4f} mm")
print(f"  new_wfo.z_w0 = {new_wfo.z_w0 * 1e3:.4f} mm")
print(f"  new_wfo.reference_surface = {new_wfo.reference_surface}")

# 读取
new_amp = proper.prop_get_amplitude(new_wfo)
new_phase = proper.prop_get_phase(new_wfo)

print(f"\n  读取的振幅范围: [{np.min(new_amp):.6f}, {np.max(new_amp):.6f}]")
print(f"  读取的相位范围: [{np.min(new_phase):.6f}, {np.max(new_phase):.6f}] rad")

# 使用 proper_to_amplitude_phase 读取
read_amp, read_phase = state_converter.proper_to_amplitude_phase(new_wfo, grid_sampling, pb)

print(f"\n  proper_to_amplitude_phase 读取:")
print(f"    振幅范围: [{np.min(read_amp):.6f}, {np.max(read_amp):.6f}]")
print(f"    相位范围: [{np.min(read_phase):.6f}, {np.max(read_phase):.6f}] rad")

# 比较
amp_diff2 = np.max(np.abs(read_amp - state_s3_exit.amplitude))
phase_diff2 = np.max(np.abs(read_phase - state_s3_exit.phase))

print(f"\n  与原始数据比较:")
print(f"    振幅最大差异: {amp_diff2:.9e}")
print(f"    相位最大差异: {phase_diff2:.9e} rad")

if amp_diff2 < 1e-6 and phase_diff2 < 1e-6:
    print("    ✓ 写入-读取循环正确")
else:
    print("    ✗ 写入-读取循环有误差！")


print("\n" + "=" * 70)
print("【检查 5】传播后的数据")
print("=" * 70)

# 保存传播前的 wfarr
wfarr_before = wfo.wfarr.copy()

# 执行传播
distance_mm = 100.0
distance_m = distance_mm * 1e-3
print(f"\n传播距离: {distance_mm} mm")
print(f"传播前 wfo.z = {wfo.z * 1e3:.4f} mm")

proper.prop_propagate(wfo, distance_m)

print(f"传播后 wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"传播后 wfo.reference_surface = {wfo.reference_surface}")

# 检查传播后的 wfarr
wfarr_after_centered = proper.prop_shift_center(wfo.wfarr)
wfarr_after_amp = np.abs(wfarr_after_centered)
wfarr_after_phase = np.angle(wfarr_after_centered)

print(f"\n传播后 wfarr（shift 后）:")
print(f"  振幅范围: [{np.min(wfarr_after_amp):.6f}, {np.max(wfarr_after_amp):.6f}]")
print(f"  相位范围: [{np.min(wfarr_after_phase):.6f}, {np.max(wfarr_after_phase):.6f}] rad")

# 使用 prop_get_amplitude/phase
proper_amp_after = proper.prop_get_amplitude(wfo)
proper_phase_after = proper.prop_get_phase(wfo)

print(f"\n传播后 prop_get_amplitude/phase:")
print(f"  振幅范围: [{np.min(proper_amp_after):.6f}, {np.max(proper_amp_after):.6f}]")
print(f"  相位范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

print("\n" + "=" * 70)
print("【检查 6】分析传播后相位为什么是 [-π, π]")
print("=" * 70)

# 计算传播后的 Pilot Beam 参数
pb_after = pb.propagate(distance_mm)
print(f"\n传播后 Pilot Beam 参数:")
print(f"  waist_position_mm = {pb_after.waist_position_mm:.4f} mm")
print(f"  curvature_radius_mm = {pb_after.curvature_radius_mm:.4f} mm")

# 计算 Pilot Beam 相位
pilot_phase_after = pb_after.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase_after):.6f}, {np.max(pilot_phase_after):.6f}] rad")

# 计算 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"\nPROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建绝对相位
wrapped_phase = proper_ref_phase + proper_phase_after
print(f"重建的绝对相位（折叠）范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")

print("\n" + "=" * 70)
print("【关键分析】为什么传播后相位是 [-π, π]？")
print("=" * 70)

# 计算传播前的相位（应该很小）
print(f"\n传播前相位范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")

# 传播后，PROPER 内部会发生什么？
# 1. 如果是近场传播（PLANAR），PROPER 使用 Fresnel 传播
# 2. 传播会引入衍射相位
# 3. 这个衍射相位可能很大，导致相位折叠

# 计算理论上的衍射相位
wavelength_m = 0.55e-6
k = 2 * np.pi / wavelength_m
X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
r_sq_m = (X_mm * 1e-3)**2 + (Y_mm * 1e-3)**2

# 菲涅尔传播的相位因子: exp(i * k * r² / (2 * z))
# 这里 z = 100 mm = 0.1 m
z_m = distance_m
fresnel_phase = k * r_sq_m / (2 * z_m)
print(f"\n菲涅尔传播相位因子范围: [{np.min(fresnel_phase):.2f}, {np.max(fresnel_phase):.2f}] rad")
print(f"菲涅尔传播相位因子最大值: {np.max(fresnel_phase):.2f} rad = {np.max(fresnel_phase) / np.pi:.2f} π")

# 这就是问题所在！
# 菲涅尔传播会引入很大的相位（几十个 π），导致相位折叠
print(f"\n【结论】")
print(f"传播距离 {distance_mm} mm 引入的菲涅尔相位最大约 {np.max(fresnel_phase):.0f} rad")
print(f"这远超过 2π，所以传播后的相位会折叠到 [-π, π]")
print(f"而 Pilot Beam 相位只有 {np.max(pilot_phase_after):.4f} rad（因为曲率半径很大）")
print(f"两者差异太大，无法正确解包裹")
