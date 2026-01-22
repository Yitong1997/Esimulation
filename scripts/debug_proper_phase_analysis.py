"""
深入分析 PROPER 传播后的相位

关键问题：PROPER 传播后的相位到底是什么？
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

print("=" * 70)
print("深入分析 PROPER 传播后的相位")
print("=" * 70)

# 创建一个简单的测试：平面波传播
wavelength_m = 0.55e-6
grid_size = 512
beam_diameter_m = 0.01  # 10 mm
physical_size_m = 0.04  # 40 mm

# 创建 PROPER 波前
wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.25)

print(f"\n初始状态:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  wfo.w0 = {wfo.w0 * 1e3:.4f} mm")
print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh * 1e3:.4f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# 获取初始相位
phase_before = proper.prop_get_phase(wfo)
print(f"  初始相位范围: [{np.min(phase_before):.6f}, {np.max(phase_before):.6f}] rad")

# 传播 100 mm
distance_m = 0.1
proper.prop_propagate(wfo, distance_m)

print(f"\n传播 {distance_m * 1e3:.0f} mm 后:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# 获取传播后的相位
phase_after = proper.prop_get_phase(wfo)
print(f"  传播后相位范围: [{np.min(phase_after):.6f}, {np.max(phase_after):.6f}] rad")

# 计算理论上的菲涅尔相位
sampling_m = proper.prop_get_sampling(wfo)
n = grid_size
coords_m = (np.arange(n) - n // 2) * sampling_m
X_m, Y_m = np.meshgrid(coords_m, coords_m)
r_sq_m = X_m**2 + Y_m**2

k = 2 * np.pi / wavelength_m

# 使用传播距离计算菲涅尔相位
fresnel_phase_z = k * r_sq_m / (2 * distance_m)
print(f"\n使用传播距离 z={distance_m*1e3:.0f}mm 计算的菲涅尔相位:")
print(f"  范围: [{np.min(fresnel_phase_z):.2f}, {np.max(fresnel_phase_z):.2f}] rad")

# 期望的传播后相位 = 原始相位 + 菲涅尔相位（折叠）
expected_phase = phase_before + fresnel_phase_z
expected_phase_wrapped = np.angle(np.exp(1j * expected_phase))

# 比较
amp_after = proper.prop_get_amplitude(wfo)
valid_mask = amp_after > 0.01 * np.max(amp_after)

phase_diff = np.angle(np.exp(1j * (phase_after - expected_phase_wrapped)))
print(f"  与实际相位的差异（有效区域）RMS: {np.std(phase_diff[valid_mask]):.6f} rad")

print("\n" + "=" * 70)
print("【关键测试】验证 PROPER 传播是否保持相位连续性")
print("=" * 70)

# 重新创建一个 wfo，这次设置一个已知的初始相位
wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.25)

# 设置一个小的初始相位（模拟像差）
initial_aberration = 0.1 * np.sin(2 * np.pi * X_m / 0.01)  # 小的正弦像差
initial_complex = np.exp(1j * initial_aberration)

# 写入 wfo
wfo2.wfarr = proper.prop_shift_center(initial_complex * proper.prop_shift_center(wfo2.wfarr))

print(f"\n初始像差范围: [{np.min(initial_aberration):.6f}, {np.max(initial_aberration):.6f}] rad")

# 获取写入后的相位
phase_written = proper.prop_get_phase(wfo2)
print(f"写入后读取的相位范围: [{np.min(phase_written):.6f}, {np.max(phase_written):.6f}] rad")

# 传播
proper.prop_propagate(wfo2, distance_m)

# 获取传播后的相位
phase_propagated = proper.prop_get_phase(wfo2)
print(f"传播后的相位范围: [{np.min(phase_propagated):.6f}, {np.max(phase_propagated):.6f}] rad")

# 计算期望的传播后相位
expected_propagated = initial_aberration + fresnel_phase_z
expected_propagated_wrapped = np.angle(np.exp(1j * expected_propagated))

# 比较
amp2 = proper.prop_get_amplitude(wfo2)
valid2 = amp2 > 0.01 * np.max(amp2)
diff2 = np.angle(np.exp(1j * (phase_propagated - expected_propagated_wrapped)))
print(f"与期望相位的差异（有效区域）RMS: {np.std(diff2[valid2]):.6f} rad")

print("\n" + "=" * 70)
print("【结论】")
print("=" * 70)

print("""
PROPER 传播的物理意义：
1. PROPER 使用 FFT 进行衍射传播
2. 传播后的相位 = 原始相位 + 菲涅尔相位 k*r²/(2*z)
3. 由于菲涅尔相位很大，相位会折叠到 [-π, π]

问题的根源：
- 在近场（PLANAR 参考面），PROPER 不减去任何参考相位
- 传播后的相位包含完整的菲涅尔相位
- Pilot Beam 严格公式在近场给出的曲率半径很大，相位很小
- 两者差异太大，无法正确解包裹

正确的解决方案：
- 在近场传播时，应该使用传播距离 z 来计算参考相位
- 参考相位 = k * r² / (2 * z)
- 这样残差相位就会很小，可以正确解包裹
""")

print("\n" + "=" * 70)
print("【验证解决方案】")
print("=" * 70)

# 使用传播距离计算参考相位
z_propagation = distance_m
ref_phase_correct = k * r_sq_m / (2 * z_propagation)
print(f"\n使用传播距离 z={z_propagation*1e3:.0f}mm 计算的参考相位:")
print(f"  范围: [{np.min(ref_phase_correct):.2f}, {np.max(ref_phase_correct):.2f}] rad")

# 计算残差
residual_correct = phase_after - np.angle(np.exp(1j * ref_phase_correct))
residual_correct = np.angle(np.exp(1j * residual_correct))
print(f"  残差范围: [{np.min(residual_correct):.6f}, {np.max(residual_correct):.6f}] rad")
print(f"  残差（有效区域）RMS: {np.std(residual_correct[valid_mask]):.6f} rad")

# 与原始相位比较
diff_from_original = residual_correct - phase_before
diff_from_original = np.angle(np.exp(1j * diff_from_original))
print(f"  残差与原始相位的差异（有效区域）RMS: {np.std(diff_from_original[valid_mask]):.6f} rad")
