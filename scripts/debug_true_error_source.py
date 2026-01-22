"""
重新分析真正的误差来源

关键问题：
- 振幅不应该影响相位插值精度
- PROPER 计算的相位网格应该是精确的
- 那么误差到底来自哪里？

需要验证：
1. 相位网格本身与 Pilot Beam 的差异（在网格点上）
2. 插值是否引入额外误差
3. 边缘区域的相位网格值是否正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)
from scipy.interpolate import RegularGridInterpolator

print("=" * 80)
print("真正的误差来源分析")
print("=" * 80)

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

# 传播到 Surface 3
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):
    propagator._propagate_to_surface(i)

# 获取入射面状态
state_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_entrance = state
        break

grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm
phase_grid = state_entrance.phase
pb = state_entrance.pilot_beam_params
R = pb.curvature_radius_mm

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

# 创建网格坐标
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_sq_grid = X**2 + Y**2

# Pilot Beam 相位网格
pilot_phase_grid = k * r_sq_grid / (2 * R) if not np.isinf(R) else np.zeros_like(r_sq_grid)

print(f"Pilot Beam 曲率半径: {R:.2f} mm")
print(f"网格分辨率: {grid_size}")

print("\n" + "=" * 60)
print("【分析 1】网格点上的相位差异")
print("=" * 60)

# 在网格点上比较相位网格与 Pilot Beam
diff_grid = phase_grid - pilot_phase_grid

# 检查不同半径处的差异
r_grid = np.sqrt(r_sq_grid)
r_values = [0, 5, 10, 15, 20]

print("\n不同半径处的相位差异（网格点上）:")
for r_target in r_values:
    # 找到最接近目标半径的网格点
    mask = np.abs(r_grid - r_target) < 0.5
    if np.sum(mask) > 0:
        mean_diff = np.mean(diff_grid[mask])
        std_diff = np.std(diff_grid[mask])
        mean_phase = np.mean(phase_grid[mask])
        mean_pilot = np.mean(pilot_phase_grid[mask])
        print(f"  r ≈ {r_target:2d} mm: "
              f"网格相位 = {mean_phase:.6f}, "
              f"Pilot = {mean_pilot:.6f}, "
              f"差异 = {mean_diff/(2*np.pi)*1000:.4f} milli-waves")

print("\n" + "=" * 60)
print("【分析 2】边缘区域的相位网格值分析")
print("=" * 60)

# 检查边缘区域（r > 15 mm）的相位网格
edge_mask = r_grid > 15
center_mask = r_grid < 5

print(f"中心区域 (r < 5 mm):")
print(f"  相位网格范围: [{np.min(phase_grid[center_mask]):.6f}, {np.max(phase_grid[center_mask]):.6f}] rad")
print(f"  Pilot Beam 范围: [{np.min(pilot_phase_grid[center_mask]):.6f}, {np.max(pilot_phase_grid[center_mask]):.6f}] rad")

print(f"\n边缘区域 (r > 15 mm):")
print(f"  相位网格范围: [{np.min(phase_grid[edge_mask]):.6f}, {np.max(phase_grid[edge_mask]):.6f}] rad")
print(f"  Pilot Beam 范围: [{np.min(pilot_phase_grid[edge_mask]):.6f}, {np.max(pilot_phase_grid[edge_mask]):.6f}] rad")

# 边缘区域的差异
edge_diff = phase_grid[edge_mask] - pilot_phase_grid[edge_mask]
print(f"  差异范围: [{np.min(edge_diff)/(2*np.pi)*1000:.4f}, {np.max(edge_diff)/(2*np.pi)*1000:.4f}] milli-waves")

print("\n" + "=" * 60)
print("【分析 3】相位网格的二次项拟合（全网格）")
print("=" * 60)

# 拟合整个网格
r_flat = r_grid.flatten()
phase_flat = phase_grid.flatten()

A = np.column_stack([r_flat**2, np.ones_like(r_flat)])
coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_flat, rcond=None)
a_fit, b_fit = coeffs

# 从二次项系数反推曲率半径
R_from_fit = k / (2 * a_fit) if a_fit != 0 else np.inf

print(f"相位网格拟合: phase = {a_fit:.9f} * r² + {b_fit:.9f}")
print(f"Pilot Beam 公式: phase = {k/(2*R):.9f} * r² + 0")
print(f"从拟合反推的曲率半径: {R_from_fit:.2f} mm")
print(f"Pilot Beam 曲率半径: {R:.2f} mm")
print(f"曲率半径差异: {(R_from_fit - R)/R * 100:.4f}%")

# 计算拟合残差
phase_fitted = a_fit * r_flat**2 + b_fit
residual = phase_flat - phase_fitted
print(f"拟合残差 RMS: {np.std(residual)/(2*np.pi)*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【分析 4】检查相位网格是否真的是二次形式")
print("=" * 60)

# 如果相位网格是精确的二次形式，那么残差应该很小
# 如果残差很大，说明相位网格不是纯二次形式

# 检查残差的空间分布
residual_grid = phase_grid - (a_fit * r_sq_grid + b_fit)
print(f"残差网格范围: [{np.min(residual_grid):.9f}, {np.max(residual_grid):.9f}] rad")
print(f"残差网格 RMS: {np.std(residual_grid)/(2*np.pi)*1000:.6f} milli-waves")

# 检查残差是否有径向依赖
print("\n残差的径向分布:")
for r_target in [0, 5, 10, 15, 20, 25]:
    mask = np.abs(r_grid - r_target) < 1
    if np.sum(mask) > 0:
        mean_res = np.mean(residual_grid[mask])
        print(f"  r ≈ {r_target:2d} mm: 残差 = {mean_res/(2*np.pi)*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【分析 5】检查 PROPER 相位的真实形式")
print("=" * 60)

import proper
wfo = state_entrance.proper_wfo
proper_phase = proper.prop_get_phase(wfo)

# PROPER 相位与 state.phase 应该相同
diff_proper_state = proper_phase - phase_grid
print(f"PROPER phase vs state.phase 差异: {np.max(np.abs(diff_proper_state)):.9f} rad")

# 检查 PROPER 相位的形式
# 高斯光束的相位应该是 k*r²/(2R) + Gouy 相位
# Gouy 相位是常数（在同一 z 平面上）

# 计算 Gouy 相位
z = -pb.waist_position_mm  # 当前位置相对于束腰
z_R = pb.rayleigh_length_mm
gouy_phase = np.arctan(z / z_R) if z_R > 0 else 0
print(f"\nGouy 相位: {gouy_phase:.9f} rad = {gouy_phase/(2*np.pi)*1000:.4f} milli-waves")

# 检查相位网格是否等于 Pilot Beam + Gouy 相位
expected_phase = pilot_phase_grid + gouy_phase
diff_with_gouy = phase_grid - expected_phase
print(f"相位网格 vs (Pilot Beam + Gouy) 差异 RMS: {np.std(diff_with_gouy)/(2*np.pi)*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【分析 6】检查边缘区域的异常相位来源")
print("=" * 60)

# 边缘区域的相位值很大，但 Pilot Beam 相位很小
# 这说明边缘区域的相位不是来自 Pilot Beam

# 检查边缘区域的相位是否来自其他来源
# 可能是 PROPER 的数值误差，或者是衍射效应

# 检查角落的相位值
corner_phase = phase_grid[-1, -1]
corner_r = r_grid[-1, -1]
corner_pilot = pilot_phase_grid[-1, -1]
corner_expected = corner_pilot + gouy_phase

print(f"角落 (r = {corner_r:.2f} mm):")
print(f"  相位网格: {corner_phase:.6f} rad")
print(f"  Pilot Beam: {corner_pilot:.6f} rad")
print(f"  Pilot Beam + Gouy: {corner_expected:.6f} rad")
print(f"  差异: {(corner_phase - corner_expected)/(2*np.pi)*1000:.4f} milli-waves")

# 检查边缘的相位值
edge_x_phase = phase_grid[grid_size//2, -1]
edge_x_r = r_grid[grid_size//2, -1]
edge_x_pilot = pilot_phase_grid[grid_size//2, -1]
edge_x_expected = edge_x_pilot + gouy_phase

print(f"\n边缘 x=20 (r = {edge_x_r:.2f} mm):")
print(f"  相位网格: {edge_x_phase:.6f} rad")
print(f"  Pilot Beam: {edge_x_pilot:.6f} rad")
print(f"  Pilot Beam + Gouy: {edge_x_expected:.6f} rad")
print(f"  差异: {(edge_x_phase - edge_x_expected)/(2*np.pi)*1000:.4f} milli-waves")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
print(f"""
关键发现：
1. 相位网格在中心区域与 Pilot Beam + Gouy 相位一致
2. 但在边缘区域（r > 15 mm），相位网格有额外的相位成分
3. 这个额外相位不是来自 Pilot Beam，也不是来自 Gouy 相位

可能的原因：
1. PROPER 的衍射计算在边缘区域产生了额外的相位
2. 高斯光束在远离光轴的区域不再是纯二次相位
3. 数值计算的边界效应

这不是插值误差，而是相位网格本身在边缘区域与 Pilot Beam 不一致。
""")
