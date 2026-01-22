"""
调试 wfo 共享问题

关键发现：
1. Surface 3 出射面和 Surface 4 入射面共享同一个 wfo 对象
2. 传播距离只有 40mm，而不是预期的 100mm
3. 传播后相位变成了 [-π, π]

需要检查：
1. 为什么 wfo 被共享？
2. 传播距离是如何计算的？
3. 为什么传播后相位会折叠？
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
print("调试 wfo 共享问题")
print("=" * 70)

# 简单测试：PROPER 传播 40mm 是否会导致相位折叠？
wavelength_m = 0.55e-6
grid_size = 512

wfo = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
wfo.z_w0 = -0.04  # -40 mm

# 设置初始相位（模拟实际系统）
sampling_m = proper.prop_get_sampling(wfo)
n = grid_size
coords_m = (np.arange(n) - n // 2) * sampling_m
X_m, Y_m = np.meshgrid(coords_m, coords_m)
r_sq_m = X_m**2 + Y_m**2

initial_phase = -0.038 * (r_sq_m / np.max(r_sq_m))
original_amp = proper.prop_get_amplitude(wfo)
new_complex = original_amp * np.exp(1j * initial_phase)
wfo.wfarr = proper.prop_shift_center(new_complex)

print(f"\n初始状态:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

phase_before = proper.prop_get_phase(wfo)
print(f"  相位范围: [{np.min(phase_before):.6f}, {np.max(phase_before):.6f}] rad")

# 传播 40mm
distance_m = 0.04  # 40 mm
print(f"\n传播 {distance_m * 1e3:.0f} mm...")
proper.prop_propagate(wfo, distance_m)

print(f"\n传播后:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

phase_after = proper.prop_get_phase(wfo)
print(f"  相位范围: [{np.min(phase_after):.6f}, {np.max(phase_after):.6f}] rad")

# 计算菲涅尔相位
k = 2 * np.pi / wavelength_m
fresnel_phase = k * r_sq_m / (2 * distance_m)
print(f"\n菲涅尔相位范围: [{np.min(fresnel_phase):.2f}, {np.max(fresnel_phase):.2f}] rad")
print(f"菲涅尔相位最大值: {np.max(fresnel_phase):.2f} rad = {np.max(fresnel_phase) / np.pi:.2f} π")


print("\n" + "=" * 70)
print("【关键测试】检查 PROPER 近场传播的行为")
print("=" * 70)

# 重新创建 wfo，这次不设置 z_w0
wfo2 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
# 不设置 z_w0，保持默认值 0

original_amp2 = proper.prop_get_amplitude(wfo2)
new_complex2 = original_amp2 * np.exp(1j * initial_phase)
wfo2.wfarr = proper.prop_shift_center(new_complex2)

print(f"\n初始状态（z_w0 = 0）:")
print(f"  z = {wfo2.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo2.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo2.reference_surface}")

phase_before2 = proper.prop_get_phase(wfo2)
print(f"  相位范围: [{np.min(phase_before2):.6f}, {np.max(phase_before2):.6f}] rad")

# 传播 40mm
proper.prop_propagate(wfo2, distance_m)

print(f"\n传播后:")
print(f"  z = {wfo2.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo2.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo2.reference_surface}")

phase_after2 = proper.prop_get_phase(wfo2)
print(f"  相位范围: [{np.min(phase_after2):.6f}, {np.max(phase_after2):.6f}] rad")

print("\n" + "=" * 70)
print("【分析】")
print("=" * 70)

print("""
关键发现：
1. 当 z_w0 = -40mm 时，传播 40mm 后相位变成 [-π, π]
2. 当 z_w0 = 0 时，传播 40mm 后相位保持很小

这说明 PROPER 的传播行为与 z_w0 的设置有关！

可能的原因：
- PROPER 在传播时会根据 z_w0 计算参考球面
- 当 z_w0 ≠ 0 时，PROPER 可能会在传播过程中切换参考面类型
- 或者 PROPER 的传播算法会考虑 z_w0 的影响
""")

print("\n" + "=" * 70)
print("【检查 PROPER 传播算法】")
print("=" * 70)

# 检查 PROPER 的 beam_type_old 属性
wfo3 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
wfo3.z_w0 = -0.04

print(f"\n传播前:")
print(f"  beam_type_old = {wfo3.beam_type_old}")
print(f"  reference_surface = {wfo3.reference_surface}")

# 检查 rayleigh_factor
z_R = wfo3.z_Rayleigh
z_diff_before = abs(wfo3.z - wfo3.z_w0)
threshold = proper.rayleigh_factor * z_R
print(f"  |z - z_w0| = {z_diff_before * 1e3:.2f} mm")
print(f"  rayleigh_factor * z_R = {threshold * 1e3:.2f} mm")
print(f"  |z - z_w0| < threshold: {z_diff_before < threshold}")

proper.prop_propagate(wfo3, distance_m)

print(f"\n传播后:")
print(f"  beam_type_old = {wfo3.beam_type_old}")
print(f"  reference_surface = {wfo3.reference_surface}")

z_diff_after = abs(wfo3.z - wfo3.z_w0)
print(f"  |z - z_w0| = {z_diff_after * 1e3:.2f} mm")
print(f"  |z - z_w0| < threshold: {z_diff_after < threshold}")
