"""
调试 PROPER 初始化时的波前相位

检查 prop_begin 创建的波前是否有相位，以及传播后相位如何变化。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 测试 1: prop_begin 创建的初始波前
# ============================================================================

print_section("测试 1: prop_begin 创建的初始波前")

wavelength_m = 0.55e-6
grid_size = 256
physical_size_m = 0.03  # 30 mm
beam_ratio = 0.5

wfo = proper.prop_begin(physical_size_m, wavelength_m, grid_size, beam_ratio)

print(f"初始 PROPER 状态:")
print(f"  wfo.z = {wfo.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.4f} mm")
print(f"  wfo.w0 = {wfo.w0 * 1e3:.4f} mm")
print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh * 1e3:.4f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# 检查 wfarr
wfarr_centered = proper.prop_shift_center(wfo.wfarr)
amplitude = np.abs(wfarr_centered)
phase = np.angle(wfarr_centered)

print(f"\n初始 wfarr:")
print(f"  振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
print(f"  相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")

# 使用 prop_get_phase
phase_proper = proper.prop_get_phase(wfo)
print(f"\nprop_get_phase:")
print(f"  相位范围: [{np.min(phase_proper):.6f}, {np.max(phase_proper):.6f}] rad")


# ============================================================================
# 测试 2: 传播 40mm 后的波前
# ============================================================================

print_section("测试 2: 传播 40mm 后的波前")

wfo2 = proper.prop_begin(physical_size_m, wavelength_m, grid_size, beam_ratio)
proper.prop_propagate(wfo2, 0.04)  # 40mm

print(f"传播后 PROPER 状态:")
print(f"  wfo.z = {wfo2.z * 1e3:.4f} mm")
print(f"  wfo.z_w0 = {wfo2.z_w0 * 1e3:.4f} mm")
print(f"  wfo.w0 = {wfo2.w0 * 1e3:.4f} mm")
print(f"  wfo.z_Rayleigh = {wfo2.z_Rayleigh * 1e3:.4f} mm")
print(f"  wfo.reference_surface = {wfo2.reference_surface}")

# 检查 wfarr
wfarr_centered2 = proper.prop_shift_center(wfo2.wfarr)
amplitude2 = np.abs(wfarr_centered2)
phase2 = np.angle(wfarr_centered2)

print(f"\n传播后 wfarr:")
print(f"  振幅范围: [{np.min(amplitude2):.6f}, {np.max(amplitude2):.6f}]")
print(f"  相位范围: [{np.min(phase2):.6f}, {np.max(phase2):.6f}] rad")
print(f"  相位范围（波长数）: [{np.min(phase2)/(2*np.pi):.6f}, {np.max(phase2)/(2*np.pi):.6f}] waves")

# 使用 prop_get_phase
phase_proper2 = proper.prop_get_phase(wfo2)
print(f"\nprop_get_phase:")
print(f"  相位范围: [{np.min(phase_proper2):.6f}, {np.max(phase_proper2):.6f}] rad")


# ============================================================================
# 测试 3: 检查相位的空间分布
# ============================================================================

