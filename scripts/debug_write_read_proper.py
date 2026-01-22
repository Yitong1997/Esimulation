"""
调试 PROPER 写入和读取的一致性

验证：state.phase → amplitude_phase_to_proper → prop_get_phase 是否一致
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
print("调试 PROPER 写入和读取的一致性")
print("=" * 70)

# 创建测试数据
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0
sampling_mm = physical_size_mm / grid_size

grid_sampling = GridSampling(
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
    sampling_mm=sampling_mm,
    beam_ratio=0.25,
)

# 创建 Pilot Beam 参数（与 Surface 3 出射面类似）
w0_mm = 5.0
z_R_mm = np.pi * w0_mm**2 / (wavelength_um * 1e-3)

# 使用 from_gaussian_source 创建，z0_mm 是束腰相对于当前位置的距离
# 当前在 z=40mm，束腰在 z=0，所以 z0_mm = -40（束腰在当前位置之前）
pilot_beam_params = PilotBeamParams.from_gaussian_source(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=-40.0,  # 束腰在当前位置之前 40mm
)

print(f"\nPilot Beam 参数:")
print(f"  w0 = {w0_mm} mm")
print(f"  z_R = {z_R_mm:.2f} mm")
print(f"  z = 40 mm")
print(f"  R = {pilot_beam_params.curvature_radius_mm:.2f} mm")

# 创建测试相位（模拟 Surface 3 出射面的相位）
# 范围约 [-0.04, 0.0] rad
X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
r_sq = X_mm**2 + Y_mm**2
test_phase = -0.04 * r_sq / (physical_size_mm/2)**2  # 简单的二次相位

print(f"\n测试相位范围: [{np.min(test_phase):.6f}, {np.max(test_phase):.6f}] rad")

# 创建测试振幅（高斯分布）
test_amplitude = np.exp(-r_sq / (2 * (w0_mm * 2)**2))

print(f"测试振幅范围: [{np.min(test_amplitude):.6f}, {np.max(test_amplitude):.6f}]")

# 使用 StateConverter 写入 PROPER
state_converter = StateConverter(wavelength_um)

print("\n" + "=" * 70)
print("写入 PROPER")
print("=" * 70)

wfo = state_converter.amplitude_phase_to_proper(
    test_amplitude,
    test_phase,
    grid_sampling,
    pilot_beam_params,
)

print(f"\nPROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

# 检查 wfarr 的相位
wfarr_phase = np.angle(proper.prop_shift_center(wfo.wfarr))  # 移回中心
print(f"\nwfarr 相位范围（移回中心后）: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")

# 使用 prop_get_phase 读取
proper_phase = proper.prop_get_phase(wfo)
print(f"prop_get_phase 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# 计算 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建相位
reconstructed_phase = proper_ref_phase + proper_phase
print(f"重建相位范围: [{np.min(reconstructed_phase):.6f}, {np.max(reconstructed_phase):.6f}] rad")

# 比较
diff = test_phase - reconstructed_phase
print(f"\n原始相位 vs 重建相位:")
print(f"  差异范围: [{np.min(diff):.9f}, {np.max(diff):.9f}] rad")
print(f"  差异 RMS: {np.std(diff):.9f} rad")

# 检查中心点
center = grid_size // 2
print(f"\n中心点检查:")
print(f"  原始相位[{center},{center}] = {test_phase[center, center]:.9f} rad")
print(f"  重建相位[{center},{center}] = {reconstructed_phase[center, center]:.9f} rad")
print(f"  wfarr 相位[{center},{center}] = {wfarr_phase[center, center]:.9f} rad")
print(f"  prop_get_phase[{center},{center}] = {proper_phase[center, center]:.9f} rad")

print("\n" + "=" * 70)
print("使用 proper_to_amplitude_phase 读取")
print("=" * 70)

amplitude_out, phase_out = state_converter.proper_to_amplitude_phase(
    wfo, grid_sampling, pilot_beam_params
)

print(f"\n读取的振幅范围: [{np.min(amplitude_out):.6f}, {np.max(amplitude_out):.6f}]")
print(f"读取的相位范围: [{np.min(phase_out):.6f}, {np.max(phase_out):.6f}] rad")

diff_phase = test_phase - phase_out
print(f"\n原始相位 vs 读取相位:")
print(f"  差异范围: [{np.min(diff_phase):.9f}, {np.max(diff_phase):.9f}] rad")
print(f"  差异 RMS: {np.std(diff_phase):.9f} rad")

# 检查 Pilot Beam 相位
pilot_phase = pilot_beam_params.compute_phase_grid(grid_size, physical_size_mm)
print(f"\nPilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")

# 检查解包裹过程
wrapped_phase = proper_ref_phase + proper_phase
phase_diff = wrapped_phase - pilot_phase
unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))

print(f"\n解包裹过程:")
print(f"  wrapped_phase 范围: [{np.min(wrapped_phase):.6f}, {np.max(wrapped_phase):.6f}] rad")
print(f"  phase_diff 范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad")
print(f"  unwrapped_phase 范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

print("\n" + "=" * 70)
print("诊断")
print("=" * 70)

# 检查 wfarr 是否正确
# 在 amplitude_phase_to_proper 中：
# residual_phase = phase - proper_ref_phase
# residual_field = amplitude * np.exp(1j * residual_phase)
# wfo.wfarr = proper.prop_shift_center(residual_field)

expected_residual_phase = test_phase - proper_ref_phase
expected_residual_field = test_amplitude * np.exp(1j * expected_residual_phase)
expected_wfarr = proper.prop_shift_center(expected_residual_field)

# 比较 wfarr
actual_wfarr = wfo.wfarr
diff_wfarr = np.abs(actual_wfarr - expected_wfarr)
print(f"\nwfarr 差异:")
print(f"  最大差异: {np.max(diff_wfarr):.9e}")

# 检查 prop_get_phase 的实现
# prop_get_phase 返回 np.angle(prop_shift_center(wfarr))
# 但是 wfarr 已经是 shift_center 后的结果，所以需要再 shift 一次？

# 让我们手动检查
manual_phase = np.angle(proper.prop_shift_center(wfo.wfarr))
print(f"\n手动计算相位（shift_center 后）:")
print(f"  范围: [{np.min(manual_phase):.6f}, {np.max(manual_phase):.6f}] rad")

# 不 shift 的情况
manual_phase_no_shift = np.angle(wfo.wfarr)
print(f"手动计算相位（不 shift）:")
print(f"  范围: [{np.min(manual_phase_no_shift):.6f}, {np.max(manual_phase_no_shift):.6f}] rad")
