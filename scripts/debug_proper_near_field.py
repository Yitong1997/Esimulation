"""
调试 PROPER 近场传播行为

关键发现：PROPER 在近场传播时似乎会自动处理菲涅尔相位
但在实际系统中，传播后相位变成了 [-π, π]
需要找出差异在哪里
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
print("调试 PROPER 近场传播行为")
print("=" * 70)

wavelength_m = 0.55e-6
grid_size = 512
distance_m = 0.1  # 100 mm

print("\n" + "=" * 70)
print("【测试 1】简单的近场传播（z_w0 = 0）")
print("=" * 70)

wfo1 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
print(f"\n初始: z={wfo1.z*1e3:.2f}mm, z_w0={wfo1.z_w0*1e3:.2f}mm, ref={wfo1.reference_surface}")

phase1_before = proper.prop_get_phase(wfo1)
print(f"传播前相位范围: [{np.min(phase1_before):.6f}, {np.max(phase1_before):.6f}] rad")

proper.prop_propagate(wfo1, distance_m)
print(f"传播后: z={wfo1.z*1e3:.2f}mm, z_w0={wfo1.z_w0*1e3:.2f}mm, ref={wfo1.reference_surface}")

phase1_after = proper.prop_get_phase(wfo1)
print(f"传播后相位范围: [{np.min(phase1_after):.6f}, {np.max(phase1_after):.6f}] rad")

print("\n" + "=" * 70)
print("【测试 2】设置 z_w0 = -40mm（模拟实际系统）")
print("=" * 70)

wfo2 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
wfo2.z_w0 = -0.04  # -40 mm
print(f"\n初始: z={wfo2.z*1e3:.2f}mm, z_w0={wfo2.z_w0*1e3:.2f}mm, ref={wfo2.reference_surface}")

phase2_before = proper.prop_get_phase(wfo2)
print(f"传播前相位范围: [{np.min(phase2_before):.6f}, {np.max(phase2_before):.6f}] rad")

proper.prop_propagate(wfo2, distance_m)
print(f"传播后: z={wfo2.z*1e3:.2f}mm, z_w0={wfo2.z_w0*1e3:.2f}mm, ref={wfo2.reference_surface}")

phase2_after = proper.prop_get_phase(wfo2)
print(f"传播后相位范围: [{np.min(phase2_after):.6f}, {np.max(phase2_after):.6f}] rad")


print("\n" + "=" * 70)
print("【测试 3】设置初始相位（模拟实际系统的像差）")
print("=" * 70)

wfo3 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
wfo3.z_w0 = -0.04  # -40 mm

# 设置一个小的初始相位
sampling_m = proper.prop_get_sampling(wfo3)
n = grid_size
coords_m = (np.arange(n) - n // 2) * sampling_m
X_m, Y_m = np.meshgrid(coords_m, coords_m)
r_sq_m = X_m**2 + Y_m**2

# 模拟实际系统的初始相位（很小的像差）
initial_phase = -0.038 * (r_sq_m / np.max(r_sq_m))  # 类似实际系统
initial_complex = np.exp(1j * initial_phase)

# 获取原始 wfarr 的振幅
original_amp = proper.prop_get_amplitude(wfo3)

# 写入新的复振幅
new_complex = original_amp * initial_complex
wfo3.wfarr = proper.prop_shift_center(new_complex)

print(f"\n初始: z={wfo3.z*1e3:.2f}mm, z_w0={wfo3.z_w0*1e3:.2f}mm, ref={wfo3.reference_surface}")

phase3_before = proper.prop_get_phase(wfo3)
print(f"传播前相位范围: [{np.min(phase3_before):.6f}, {np.max(phase3_before):.6f}] rad")

proper.prop_propagate(wfo3, distance_m)
print(f"传播后: z={wfo3.z*1e3:.2f}mm, z_w0={wfo3.z_w0*1e3:.2f}mm, ref={wfo3.reference_surface}")

phase3_after = proper.prop_get_phase(wfo3)
print(f"传播后相位范围: [{np.min(phase3_after):.6f}, {np.max(phase3_after):.6f}] rad")

print("\n" + "=" * 70)
print("【测试 4】检查 PROPER 的 rayleigh_factor")
print("=" * 70)

print(f"\nproper.rayleigh_factor = {proper.rayleigh_factor}")

# 检查 z - z_w0 与 rayleigh_factor * z_Rayleigh 的关系
z_R = wfo3.z_Rayleigh
z_diff = wfo3.z - wfo3.z_w0
threshold = proper.rayleigh_factor * z_R

print(f"z_Rayleigh = {z_R * 1e3:.2f} mm")
print(f"|z - z_w0| = {abs(z_diff) * 1e3:.2f} mm")
print(f"rayleigh_factor * z_Rayleigh = {threshold * 1e3:.2f} mm")
print(f"|z - z_w0| < threshold: {abs(z_diff) < threshold}")

print("\n" + "=" * 70)
print("【测试 5】强制设置为 SPHERI 参考面")
print("=" * 70)

wfo5 = proper.prop_begin(0.01, wavelength_m, grid_size, 0.25)
wfo5.z_w0 = -0.04  # -40 mm
wfo5.reference_surface = "SPHERI"  # 强制设置为 SPHERI
wfo5.beam_type_old = "OUTSIDE"

# 设置初始相位
new_complex5 = original_amp * initial_complex
wfo5.wfarr = proper.prop_shift_center(new_complex5)

print(f"\n初始: z={wfo5.z*1e3:.2f}mm, z_w0={wfo5.z_w0*1e3:.2f}mm, ref={wfo5.reference_surface}")

phase5_before = proper.prop_get_phase(wfo5)
print(f"传播前相位范围: [{np.min(phase5_before):.6f}, {np.max(phase5_before):.6f}] rad")

proper.prop_propagate(wfo5, distance_m)
print(f"传播后: z={wfo5.z*1e3:.2f}mm, z_w0={wfo5.z_w0*1e3:.2f}mm, ref={wfo5.reference_surface}")

phase5_after = proper.prop_get_phase(wfo5)
print(f"传播后相位范围: [{np.min(phase5_after):.6f}, {np.max(phase5_after):.6f}] rad")

print("\n" + "=" * 70)
print("【分析】")
print("=" * 70)

print("""
关键发现：
1. 当 z_w0 = 0 时，PROPER 传播后相位保持很小
2. 当 z_w0 = -40mm 时，PROPER 传播后相位变成 [-π, π]
3. 这说明 PROPER 的传播行为与 z_w0 的设置有关

可能的原因：
- PROPER 在传播时会根据 z_w0 计算参考球面
- 当 z_w0 ≠ 0 时，参考球面的曲率半径会影响传播结果
""")
