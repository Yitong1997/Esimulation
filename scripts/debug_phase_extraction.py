"""
详细分析从 PROPER 提取相位的每一步

理论上：
1. PROPER 存储的是相对于参考波面的残差相位（prop_get_phase 返回）
2. PROPER 参考波面的相位应当解析计算（不用复数，避免折叠）
3. 绝对相位 = 残差相位 + 参考波面相位（解析计算）
4. 结果应当是非折叠的

测试每一步的中间结果，找出哪一步出现了折叠。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import proper


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 创建测试场景：传播 40mm 后的高斯光束
# ============================================================================

print_section("创建测试场景")

wavelength_um = 0.55
wavelength_m = wavelength_um * 1e-6
wavelength_mm = wavelength_um * 1e-3
w0_mm = 5.0
w0_m = w0_mm * 1e-3
grid_size = 256
physical_size_mm = 80.0
physical_size_m = physical_size_mm * 1e-3

# 计算瑞利长度
z_R_mm = np.pi * w0_mm**2 / wavelength_mm
z_R_m = z_R_mm * 1e-3

print(f"波长: {wavelength_um} μm")
print(f"束腰半径: {w0_mm} mm")
print(f"瑞利长度: {z_R_mm:.1f} mm")
print(f"网格大小: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")


# ============================================================================
# 步骤 1: 创建初始 PROPER 波前并传播
# ============================================================================

print_section("步骤 1: 创建初始 PROPER 波前并传播")

# 创建 PROPER 波前
beam_diameter_m = physical_size_m
wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)

print(f"初始 PROPER 状态:")
print(f"  wfo.z = {wfo.z} m")
print(f"  wfo.z_w0 = {wfo.z_w0} m")
print(f"  wfo.w0 = {wfo.w0} m")
print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh} m")
print(f"  wfo.reference_surface = {wfo.reference_surface}")
print(f"  wfo.lamda = {wfo.lamda} m")

# 获取初始采样
sampling_m = proper.prop_get_sampling(wfo)
sampling_mm = sampling_m * 1e3
print(f"  采样间隔: {sampling_mm:.6f} mm")

# 传播 40mm
propagation_distance_mm = 40.0
propagation_distance_m = propagation_distance_mm * 1e-3

print(f"\n传播距离: {propagation_distance_mm} mm")
proper.prop_propagate(wfo, propagation_distance_m)

print(f"\n传播后 PROPER 状态:")
print(f"  wfo.z = {wfo.z} m = {wfo.z * 1e3} mm")
print(f"  wfo.z_w0 = {wfo.z_w0} m = {wfo.z_w0 * 1e3} mm")
print(f"  wfo.w0 = {wfo.w0} m = {wfo.w0 * 1e3} mm")
print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh} m = {wfo.z_Rayleigh * 1e3} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# 计算 PROPER 的参考曲率半径（远场近似）
R_proper_m = wfo.z - wfo.z_w0
R_proper_mm = R_proper_m * 1e3
print(f"  PROPER 参考曲率半径 (z - z_w0): {R_proper_mm:.2f} mm")


# ============================================================================
# 步骤 2: 从 PROPER 提取残差相位
# ============================================================================

print_section("步骤 2: 从 PROPER 提取残差相位")

# prop_get_phase 返回的是相对于 PROPER 参考面的残差相位
residual_phase = proper.prop_get_phase(wfo)
amplitude = proper.prop_get_amplitude(wfo)

print(f"残差相位统计:")
print(f"  形状: {residual_phase.shape}")
print(f"  最小值: {np.min(residual_phase):.6f} rad")
print(f"  最大值: {np.max(residual_phase):.6f} rad")
print(f"  范围: {np.max(residual_phase) - np.min(residual_phase):.6f} rad")
print(f"  范围（波长数）: {(np.max(residual_phase) - np.min(residual_phase))/(2*np.pi):.6f} waves")

# 检查残差相位是否在 [-π, π] 范围内
if np.min(residual_phase) >= -np.pi and np.max(residual_phase) <= np.pi:
    print(f"  [INFO] 残差相位在 [-π, π] 范围内")
else:
    print(f"  [WARNING] 残差相位超出 [-π, π] 范围")

# 有效区域
valid_mask = amplitude > 0.01 * np.max(amplitude)
residual_phase_valid = residual_phase[valid_mask]
print(f"\n有效区域内残差相位统计:")
print(f"  有效像素数: {np.sum(valid_mask)}")
print(f"  最小值: {np.min(residual_phase_valid):.6f} rad")
print(f"  最大值: {np.max(residual_phase_valid):.6f} rad")
print(f"  RMS: {np.std(residual_phase_valid):.6f} rad")


# ============================================================================
# 步骤 3: 解析计算 PROPER 参考波面相位（不用复数！）
# ============================================================================

print_section("步骤 3: 解析计算 PROPER 参考波面相位")

# 创建坐标网格
n = grid_size
half_size_mm = physical_size_mm / 2
coords_mm = np.linspace(-half_size_mm, half_size_mm, n)
X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
r_sq_mm = X_mm**2 + Y_mm**2
r_sq_m = r_sq_mm * 1e-6  # 转换为 m²

print(f"坐标网格:")
print(f"  范围: [{-half_size_mm}, {half_size_mm}] mm")
print(f"  r² 最大值: {np.max(r_sq_mm):.2f} mm²")

# PROPER 参考波面相位公式（球面参考）
# φ_ref = -k × r² / (2 × R_ref)
# 注意：PROPER 使用负号！

k = 2 * np.pi / wavelength_m  # 波数 (1/m)

if wfo.reference_surface == "PLANAR":
    print(f"\nPROPER 参考面类型: PLANAR（平面）")
    proper_ref_phase = np.zeros((n, n))
elif wfo.reference_surface == "SPHERI":
    print(f"\nPROPER 参考面类型: SPHERI（球面）")
    print(f"  参考曲率半径 R_ref = z - z_w0 = {R_proper_m:.6f} m = {R_proper_mm:.2f} mm")
    
    if abs(R_proper_m) < 1e-10:
        print(f"  [WARNING] 曲率半径接近零，使用平面参考")
        proper_ref_phase = np.zeros((n, n))
    else:
        # 解析计算参考相位（不用复数！）
        # φ_ref = -k × r² / (2 × R_ref)
        proper_ref_phase = -k * r_sq_m / (2 * R_proper_m)
else:
    print(f"\n[WARNING] 未知参考面类型: {wfo.reference_surface}")
    proper_ref_phase = np.zeros((n, n))

print(f"\nPROPER 参考波面相位（解析计算）:")
print(f"  最小值: {np.min(proper_ref_phase):.6f} rad")
print(f"  最大值: {np.max(proper_ref_phase):.6f} rad")
print(f"  范围: {np.max(proper_ref_phase) - np.min(proper_ref_phase):.6f} rad")
print(f"  范围（波长数）: {(np.max(proper_ref_phase) - np.min(proper_ref_phase))/(2*np.pi):.6f} waves")

# 检查参考相位是否超出 [-π, π]
if np.max(np.abs(proper_ref_phase)) > np.pi:
    print(f"  [INFO] 参考相位超出 [-π, π] 范围（这是正常的，因为是解析计算）")
else:
    print(f"  [INFO] 参考相位在 [-π, π] 范围内")


# ============================================================================
# 步骤 4: 计算绝对相位 = 残差相位 + 参考相位
# ============================================================================

print_section("步骤 4: 计算绝对相位")

# 绝对相位 = 残差相位 + 参考相位
absolute_phase = residual_phase + proper_ref_phase

print(f"绝对相位 = 残差相位 + 参考相位:")
print(f"  最小值: {np.min(absolute_phase):.6f} rad")
print(f"  最大值: {np.max(absolute_phase):.6f} rad")
print(f"  范围: {np.max(absolute_phase) - np.min(absolute_phase):.6f} rad")
print(f"  范围（波长数）: {(np.max(absolute_phase) - np.min(absolute_phase))/(2*np.pi):.6f} waves")

# 有效区域内
absolute_phase_valid = absolute_phase[valid_mask]
print(f"\n有效区域内绝对相位统计:")
print(f"  最小值: {np.min(absolute_phase_valid):.6f} rad")
print(f"  最大值: {np.max(absolute_phase_valid):.6f} rad")
print(f"  范围: {np.max(absolute_phase_valid) - np.min(absolute_phase_valid):.6f} rad")

# 检查绝对相位是否被折叠
if np.max(absolute_phase) - np.min(absolute_phase) > 2 * np.pi:
    print(f"  [INFO] 绝对相位范围超过 2π（这是正常的，表示非折叠）")
elif np.max(absolute_phase) - np.min(absolute_phase) < 2 * np.pi * 0.01:
    print(f"  [INFO] 绝对相位范围很小（接近平面波）")
else:
    print(f"  [WARNING] 绝对相位范围在 2π 附近，可能存在折叠问题")


# ============================================================================
# 步骤 5: 与理论高斯光束相位比较
# ============================================================================

print_section("步骤 5: 与理论高斯光束相位比较")

# 理论高斯光束相位（使用严格公式）
# 传播距离 z = 40mm，束腰在 z=0
z_mm = propagation_distance_mm
z_m = z_mm * 1e-3

# 严格曲率半径公式: R = z × (1 + (z_R/z)²)
if abs(z_mm) < 1e-10:
    R_theory_mm = np.inf
else:
    R_theory_mm = z_mm * (1 + (z_R_mm / z_mm)**2)
    R_theory_m = R_theory_mm * 1e-3

print(f"理论高斯光束参数:")
print(f"  传播距离 z: {z_mm} mm")
print(f"  瑞利长度 z_R: {z_R_mm:.2f} mm")
print(f"  z/z_R: {z_mm/z_R_mm:.6f}")
print(f"  严格曲率半径 R = z × (1 + (z_R/z)²): {R_theory_mm:.2f} mm")
print(f"  PROPER 远场近似曲率半径 R = z - z_w0: {R_proper_mm:.2f} mm")
print(f"  曲率半径差异: {abs(R_theory_mm - R_proper_mm):.2f} mm")

# 理论相位（相对于主光线）
# φ_theory(r) = k × r² / (2 × R)
if np.isinf(R_theory_mm):
    theory_phase = np.zeros((n, n))
else:
    theory_phase = k * r_sq_m / (2 * R_theory_m)

print(f"\n理论高斯光束相位（严格公式）:")
print(f"  最小值: {np.min(theory_phase):.6f} rad")
print(f"  最大值: {np.max(theory_phase):.6f} rad")
print(f"  范围: {np.max(theory_phase) - np.min(theory_phase):.6f} rad")
print(f"  范围（波长数）: {(np.max(theory_phase) - np.min(theory_phase))/(2*np.pi):.6f} waves")

# 比较绝对相位与理论相位
phase_diff = absolute_phase - theory_phase
phase_diff_valid = phase_diff[valid_mask]

print(f"\n绝对相位与理论相位的差异:")
print(f"  最小值: {np.min(phase_diff_valid):.6f} rad")
print(f"  最大值: {np.max(phase_diff_valid):.6f} rad")
print(f"  RMS: {np.std(phase_diff_valid):.6f} rad = {np.std(phase_diff_valid)/(2*np.pi):.6f} waves")


# ============================================================================
# 步骤 6: 检查当前 StateConverter 的实现
# ============================================================================

print_section("步骤 6: 检查当前 StateConverter 的实现")

from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.data_models import GridSampling, PilotBeamParams

# 创建 StateConverter
state_converter = StateConverter(wavelength_um)

# 创建 GridSampling
grid_sampling = GridSampling(
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
    sampling_mm=sampling_mm,
    beam_ratio=0.5,
)

# 使用 StateConverter 提取相位
amplitude_sc, phase_sc = state_converter.proper_to_amplitude_phase(
    wfo, grid_sampling, pilot_beam_params=None
)

print(f"StateConverter 提取的相位:")
print(f"  最小值: {np.min(phase_sc):.6f} rad")
print(f"  最大值: {np.max(phase_sc):.6f} rad")
print(f"  范围: {np.max(phase_sc) - np.min(phase_sc):.6f} rad")
print(f"  范围（波长数）: {(np.max(phase_sc) - np.min(phase_sc))/(2*np.pi):.6f} waves")

# 与我们手动计算的绝对相位比较
phase_diff_sc = phase_sc - absolute_phase
print(f"\nStateConverter 相位与手动计算的差异:")
print(f"  最大差异: {np.max(np.abs(phase_diff_sc)):.6f} rad")

# 检查 StateConverter 计算的 PROPER 参考相位
proper_ref_phase_sc = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"\nStateConverter 计算的 PROPER 参考相位:")
print(f"  最小值: {np.min(proper_ref_phase_sc):.6f} rad")
print(f"  最大值: {np.max(proper_ref_phase_sc):.6f} rad")

# 与我们手动计算的参考相位比较
ref_diff = proper_ref_phase_sc - proper_ref_phase
print(f"\n参考相位差异:")
print(f"  最大差异: {np.max(np.abs(ref_diff)):.6f} rad")
