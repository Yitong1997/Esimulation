"""
调试 PROPER 解包裹问题的解决方案

问题：在近场（PLANAR 参考面）时，PROPER 传播后的相位包含完整的菲涅尔相位，
但 Pilot Beam 严格公式给出的曲率半径很大，相位很小，无法用于解包裹。

解决方案：使用 PROPER 的远场近似曲率半径 R = z - z_w0 来计算参考相位进行解包裹
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
print("调试 PROPER 解包裹问题的解决方案")
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
for i in range(4):
    propagator._propagate_to_surface(i)

# 找到 Surface 3 出射面状态
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
        break

wfo = state_s3_exit.proper_wfo
grid_sampling = state_s3_exit.grid_sampling
pb = state_s3_exit.pilot_beam_params

# 保存传播前的相位
phase_before = state_s3_exit.phase.copy()

print(f"\n传播前:")
print(f"  state.phase 范围: [{np.min(phase_before):.6f}, {np.max(phase_before):.6f}] rad")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")

# 执行传播
distance_mm = 100.0
distance_m = distance_mm * 1e-3
proper.prop_propagate(wfo, distance_m)

print(f"\n传播后:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# 获取传播后的相位
proper_phase_after = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

print("\n" + "=" * 70)
print("【方案 1】使用 Pilot Beam 严格公式（当前实现，失败）")
print("=" * 70)

pb_after = pb.propagate(distance_mm)
pilot_phase_strict = pb_after.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)

print(f"\nPilot Beam 参数（严格公式）:")
print(f"  R = {pb_after.curvature_radius_mm:.4f} mm")
print(f"  相位范围: [{np.min(pilot_phase_strict):.6f}, {np.max(pilot_phase_strict):.6f}] rad")

# 尝试解包裹
phase_diff_strict = proper_phase_after - pilot_phase_strict
unwrapped_strict = pilot_phase_strict + np.angle(np.exp(1j * phase_diff_strict))

print(f"\n解包裹结果:")
print(f"  相位差范围: [{np.min(phase_diff_strict):.6f}, {np.max(phase_diff_strict):.6f}] rad")
print(f"  解包裹后范围: [{np.min(unwrapped_strict):.6f}, {np.max(unwrapped_strict):.6f}] rad")
print(f"  相位差最大绝对值: {np.max(np.abs(phase_diff_strict)):.4f} rad")
print(f"  是否超过 π: {np.max(np.abs(phase_diff_strict)) > np.pi}")

print("\n" + "=" * 70)
print("【方案 2】使用 PROPER 远场近似曲率半径")
print("=" * 70)

# 使用 PROPER 的远场近似曲率半径
R_proper = (wfo.z - wfo.z_w0) * 1e3  # 转换为 mm
print(f"\nPROPER 远场近似曲率半径: R = z - z_w0 = {R_proper:.4f} mm")

# 计算参考相位
wavelength_mm = 0.55e-3
k = 2 * np.pi / wavelength_mm
X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
r_sq = X_mm**2 + Y_mm**2

if abs(R_proper) > 1e-10:
    pilot_phase_proper = k * r_sq / (2 * R_proper)
else:
    pilot_phase_proper = np.zeros_like(r_sq)

print(f"参考相位范围: [{np.min(pilot_phase_proper):.6f}, {np.max(pilot_phase_proper):.6f}] rad")

# 尝试解包裹
phase_diff_proper = proper_phase_after - pilot_phase_proper
unwrapped_proper = pilot_phase_proper + np.angle(np.exp(1j * phase_diff_proper))

print(f"\n解包裹结果:")
print(f"  相位差范围: [{np.min(phase_diff_proper):.6f}, {np.max(phase_diff_proper):.6f}] rad")
print(f"  解包裹后范围: [{np.min(unwrapped_proper):.6f}, {np.max(unwrapped_proper):.6f}] rad")
print(f"  相位差最大绝对值: {np.max(np.abs(phase_diff_proper)):.4f} rad")
print(f"  是否超过 π: {np.max(np.abs(phase_diff_proper)) > np.pi}")


print("\n" + "=" * 70)
print("【方案 3】理解 PROPER 传播的物理意义")
print("=" * 70)

# PROPER 传播后的相位是什么？
# 对于 PLANAR 参考面，wfarr 存储的是完整的复振幅
# 传播后，相位 = 原始相位 + 菲涅尔传播相位
# 菲涅尔传播相位 = k * r² / (2 * z)

# 但是！PROPER 使用的是 FFT 传播，不是简单的菲涅尔近似
# FFT 传播会自动处理衍射效应

# 让我们验证：传播后的相位是否等于 原始相位 + k * r² / (2 * z)？
z_m = distance_m
r_sq_m = (X_mm * 1e-3)**2 + (Y_mm * 1e-3)**2
k_m = 2 * np.pi / (0.55e-6)

# 理论上的菲涅尔相位
fresnel_phase = k_m * r_sq_m / (2 * z_m)
print(f"\n理论菲涅尔相位范围: [{np.min(fresnel_phase):.2f}, {np.max(fresnel_phase):.2f}] rad")

# 期望的传播后相位 = 原始相位 + 菲涅尔相位
expected_phase = phase_before + fresnel_phase
expected_phase_wrapped = np.angle(np.exp(1j * expected_phase))
print(f"期望的传播后相位（折叠）范围: [{np.min(expected_phase_wrapped):.6f}, {np.max(expected_phase_wrapped):.6f}] rad")

# 比较
phase_error = np.angle(np.exp(1j * (proper_phase_after - expected_phase_wrapped)))
print(f"实际与期望的差异范围: [{np.min(phase_error):.6f}, {np.max(phase_error):.6f}] rad")

# 在有效区域内比较
amp_after = proper.prop_get_amplitude(wfo)
valid_mask = amp_after > 0.01 * np.max(amp_after)
phase_error_valid = phase_error[valid_mask]
print(f"有效区域内差异 RMS: {np.std(phase_error_valid):.6f} rad")

print("\n" + "=" * 70)
print("【结论】")
print("=" * 70)

print("""
问题分析：
1. PROPER 传播后的相位确实包含菲涅尔相位 k*r²/(2*z)
2. 这个相位非常大（约 45000 rad），会折叠到 [-π, π]
3. Pilot Beam 严格公式在近场给出的曲率半径很大，相位很小
4. 两者差异太大，无法正确解包裹

解决方案：
在近场（PLANAR 参考面）时，应该使用 PROPER 的远场近似曲率半径 R = z - z_w0
来计算参考相位进行解包裹，而不是 Pilot Beam 的严格公式。

但是！这会导致一个新问题：
- 使用 R = z - z_w0 = 140 mm 计算的参考相位也很大
- 这个参考相位与 PROPER 传播后的相位应该非常接近
- 解包裹后的残差应该很小

让我验证这一点...
""")

# 使用 R = z - z_w0 计算参考相位
R_ref_mm = (wfo.z - wfo.z_w0) * 1e3
ref_phase = k * r_sq / (2 * R_ref_mm)
print(f"使用 R = {R_ref_mm:.2f} mm 计算的参考相位范围: [{np.min(ref_phase):.2f}, {np.max(ref_phase):.2f}] rad")

# 计算残差
residual = proper_phase_after - np.angle(np.exp(1j * ref_phase))
residual = np.angle(np.exp(1j * residual))  # 折叠到 [-π, π]
print(f"残差范围: [{np.min(residual):.6f}, {np.max(residual):.6f}] rad")

# 在有效区域内
residual_valid = residual[valid_mask]
print(f"有效区域内残差 RMS: {np.std(residual_valid):.6f} rad")

# 与原始相位比较
original_phase_valid = phase_before[valid_mask]
print(f"原始相位有效区域 RMS: {np.std(original_phase_valid):.6f} rad")

# 残差应该接近原始相位（因为原始相位很小）
diff_from_original = residual - phase_before
diff_from_original = np.angle(np.exp(1j * diff_from_original))
diff_valid = diff_from_original[valid_mask]
print(f"残差与原始相位的差异 RMS: {np.std(diff_valid):.6f} rad")
