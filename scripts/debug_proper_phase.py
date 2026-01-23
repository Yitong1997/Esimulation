"""
调试 PROPER 相位提取问题

分析 PROPER 的参考面类型和相位折叠问题。

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import proper


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 测试 PROPER 的参考面行为
# ============================================================================

print_section("测试 PROPER 的参考面行为")

# 参数
wavelength_um = 0.55
wavelength_m = wavelength_um * 1e-6
w0_mm = 5.0
w0_m = w0_mm * 1e-3
grid_size = 256
physical_size_mm = 40.0
physical_size_m = physical_size_mm * 1e-3

# 计算瑞利长度
wavelength_mm = wavelength_um * 1e-3
z_R_mm = np.pi * w0_mm**2 / wavelength_mm
z_R_m = z_R_mm * 1e-3

print(f"""
参数:
  波长: {wavelength_um} μm
  束腰半径: {w0_mm} mm
  瑞利长度: {z_R_mm:.1f} mm
  网格大小: {grid_size} × {grid_size}
  物理尺寸: {physical_size_mm} mm
""")


# ============================================================================
# 创建初始波前
# ============================================================================

print_section("创建初始波前")

wfo = proper.prop_begin(
    physical_size_m,
    wavelength_m,
    grid_size,
    0.5,  # beam_ratio
)

print(f"初始状态:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z * 1e3:.2f} mm")
print(f"  z_w0: {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z_Rayleigh: {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"  w0: {wfo.w0 * 1e3:.4f} mm")

# 获取采样
sampling_m = proper.prop_get_sampling(wfo)
sampling_mm = sampling_m * 1e3
actual_physical_size_mm = sampling_mm * grid_size

print(f"  采样间隔: {sampling_mm:.4f} mm")
print(f"  实际物理尺寸: {actual_physical_size_mm:.2f} mm")


# ============================================================================
# 传播 10mm 并检查相位
# ============================================================================

print_section("传播 10mm 并检查相位")

# 传播
distance_mm = 10.0
distance_m = distance_mm * 1e-3
proper.prop_propagate(wfo, distance_m)

print(f"传播后状态:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z * 1e3:.2f} mm")
print(f"  z_w0: {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z_Rayleigh: {wfo.z_Rayleigh * 1e3:.2f} mm")

# 提取相位
amplitude = proper.prop_get_amplitude(wfo)
phase = proper.prop_get_phase(wfo)

print(f"\n提取的相位:")
print(f"  最大值: {np.max(phase):.4f} rad")
print(f"  最小值: {np.min(phase):.4f} rad")
print(f"  范围: {np.max(phase) - np.min(phase):.4f} rad")

# 检查是否折叠
if np.max(phase) - np.min(phase) > 2 * np.pi - 0.1:
    print(f"  [WARNING] 相位范围接近 2π，可能已折叠!")


# ============================================================================
# 计算理论相位
# ============================================================================

print_section("计算理论相位")

# 创建坐标网格
half_size = actual_physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
R_sq = X**2 + Y**2

# 计算理论曲率半径（严格公式）
z = distance_mm
R_theory = z * (1 + (z_R_mm / z)**2) if abs(z) > 1e-10 else np.inf

print(f"理论曲率半径: {R_theory:.2f} mm")

# 计算理论相位
k = 2 * np.pi / wavelength_mm
if np.isinf(R_theory):
    theory_phase = np.zeros_like(R_sq)
else:
    theory_phase = k * R_sq / (2 * R_theory)

print(f"理论相位:")
print(f"  最大值: {np.max(theory_phase):.4f} rad")
print(f"  最小值: {np.min(theory_phase):.4f} rad")
print(f"  范围: {np.max(theory_phase) - np.min(theory_phase):.4f} rad")


# ============================================================================
# 计算 PROPER 参考面相位
# ============================================================================

print_section("计算 PROPER 参考面相位")

if wfo.reference_surface == "PLANAR":
    proper_ref_phase = np.zeros((grid_size, grid_size))
    print("PROPER 参考面: PLANAR (相位 = 0)")
else:
    R_ref_m = wfo.z - wfo.z_w0
    R_ref_mm = R_ref_m * 1e3
    print(f"PROPER 参考面: SPHERI")
    print(f"  参考曲率半径: {R_ref_mm:.2f} mm")
    
    # 计算参考相位
    r_sq_m = (X * 1e-3)**2 + (Y * 1e-3)**2
    k_m = 2 * np.pi / wavelength_m
    proper_ref_phase = -k_m * r_sq_m / (2 * R_ref_m)
    
    print(f"  参考相位最大值: {np.max(proper_ref_phase):.4f} rad")
    print(f"  参考相位最小值: {np.min(proper_ref_phase):.4f} rad")


# ============================================================================
# 重建绝对相位
# ============================================================================

print_section("重建绝对相位")

absolute_phase = proper_ref_phase + phase

print(f"重建的绝对相位:")
print(f"  最大值: {np.max(absolute_phase):.4f} rad")
print(f"  最小值: {np.min(absolute_phase):.4f} rad")
print(f"  范围: {np.max(absolute_phase) - np.min(absolute_phase):.4f} rad")

# 与理论相位比较
phase_diff = np.angle(np.exp(1j * (absolute_phase - theory_phase)))
valid_mask = amplitude > 0.01 * np.max(amplitude)

if np.sum(valid_mask) > 0:
    phase_rms = np.sqrt(np.mean(phase_diff[valid_mask]**2))
    print(f"\n与理论相位的差异:")
    print(f"  RMS: {phase_rms / (2*np.pi):.6f} waves")
else:
    print(f"\n[WARNING] 无有效数据!")


# ============================================================================
# 绘制分析图
# ============================================================================

print_section("绘制分析图")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

extent = [-half_size, half_size, -half_size, half_size]

# 振幅
im1 = axes[0, 0].imshow(amplitude, extent=extent, cmap='viridis')
axes[0, 0].set_title('Amplitude')
axes[0, 0].set_xlabel('X (mm)')
axes[0, 0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0, 0])

# PROPER 提取的相位
im2 = axes[0, 1].imshow(phase, extent=extent, cmap='twilight')
axes[0, 1].set_title(f'PROPER Phase (prop_get_phase)\nRange: {np.max(phase) - np.min(phase):.2f} rad')
axes[0, 1].set_xlabel('X (mm)')
axes[0, 1].set_ylabel('Y (mm)')
plt.c