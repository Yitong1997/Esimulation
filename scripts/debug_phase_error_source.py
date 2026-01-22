"""
深入分析相位误差的真正来源

关键问题：
- 网格相位 vs Pilot Beam 网格相位: 0.000000 waves（精确）
- 插值 Pilot Beam 网格 vs Pilot Beam 解析: 0.000000 waves（精确）
- 插值相位 vs Pilot Beam: 0.000647 waves（有误差）

这说明误差来自于相位网格本身与 Pilot Beam 的差异，而不是插值过程
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
print("相位误差真正来源分析")
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

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm
grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm
pb = state_entrance.pilot_beam_params
R = pb.curvature_radius_mm

print(f"网格分辨率: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")
print(f"Pilot Beam 曲率半径: {R:.2f} mm")
print(f"Pilot Beam 曲率半径是否无穷大: {np.isinf(R)}")

# 创建网格坐标
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_sq_grid = X**2 + Y**2

# 相位网格
phase_grid = state_entrance.phase
amplitude_grid = state_entrance.amplitude
mask = amplitude_grid > 0.01 * np.max(amplitude_grid)

# Pilot Beam 网格相位
if np.isinf(R):
    pilot_phase_grid = np.zeros_like(r_sq_grid)
else:
    pilot_phase_grid = k * r_sq_grid / (2 * R)

print("\n" + "=" * 60)
print("【分析 1】网格上的相位差异")
print("=" * 60)

diff_grid = phase_grid - pilot_phase_grid
print(f"相位网格范围: [{np.min(phase_grid[mask]):.6f}, {np.max(phase_grid[mask]):.6f}] rad")
print(f"Pilot Beam 网格范围: [{np.min(pilot_phase_grid[mask]):.6f}, {np.max(pilot_phase_grid[mask]):.6f}] rad")
print(f"差异范围: [{np.min(diff_grid[mask]):.6f}, {np.max(diff_grid[mask]):.6f}] rad")
print(f"差异 RMS: {np.std(diff_grid[mask]):.9f} rad = {np.std(diff_grid[mask])/(2*np.pi):.9f} waves")

# 检查中心点
center_idx = grid_size // 2
print(f"\n中心点 ({center_idx}, {center_idx}):")
print(f"  相位网格: {phase_grid[center_idx, center_idx]:.9f} rad")
print(f"  Pilot Beam: {pilot_phase_grid[center_idx, center_idx]:.9f} rad")
print(f"  差异: {diff_grid[center_idx, center_idx]:.9f} rad")

print("\n" + "=" * 60)
print("【分析 2】相位网格的特征")
print("=" * 60)

# 相位网格是否有常数偏移？
mean_diff = np.mean(diff_grid[mask])
print(f"差异的平均值: {mean_diff:.9f} rad = {mean_diff/(2*np.pi)*1000:.6f} milli-waves")

# 减去平均值后的 RMS
diff_centered = diff_grid - mean_diff
rms_centered = np.std(diff_centered[mask])
print(f"减去平均值后的 RMS: {rms_centered:.9f} rad = {rms_centered/(2*np.pi):.9f} waves")

print("\n" + "=" * 60)
print("【分析 3】检查相位网格是否包含 Gouy 相位或其他偏移")
print("=" * 60)

# 计算 Gouy 相位
# PilotBeamParams 属性: waist_position_mm, waist_radius_mm, rayleigh_length_mm
z_w0 = pb.waist_position_mm  # 束腰位置（相对于当前位置）
z_R = pb.rayleigh_length_mm
z = -z_w0  # 当前位置相对于束腰的距离
gouy_phase = np.arctan(z / z_R) if z_R > 0 else 0

print(f"Pilot Beam 参数:")
print(f"  waist_position_mm = {z_w0:.2f} mm")
print(f"  z (相对于束腰) = {z:.2f} mm")
print(f"  z_R = {z_R:.2f} mm")
print(f"  Gouy 相位: {gouy_phase:.9f} rad = {gouy_phase/(2*np.pi)*1000:.6f} milli-waves")

# 检查差异是否等于 Gouy 相位
print(f"\n差异平均值 vs Gouy 相位:")
print(f"  差异平均值: {mean_diff:.9f} rad")
print(f"  Gouy 相位: {gouy_phase:.9f} rad")
print(f"  比值: {mean_diff/gouy_phase if gouy_phase != 0 else 'N/A'}")

print("\n" + "=" * 60)
print("【分析 4】检查 PROPER 输出的相位特征")
print("=" * 60)

# 检查相位网格是否有径向依赖
# 如果相位网格 = Pilot Beam + 常数，那么差异应该是常数
# 如果相位网格有其他成分，差异会有径向依赖

r_grid = np.sqrt(r_sq_grid)
r_bins = np.linspace(0, 20, 21)

print("\n径向差异分布:")
print("-" * 60)
for i in range(len(r_bins) - 1):
    mask_r = mask & (r_grid >= r_bins[i]) & (r_grid < r_bins[i+1])
    if np.sum(mask_r) > 0:
        mean_r = np.mean(diff_grid[mask_r])
        std_r = np.std(diff_grid[mask_r])
        print(f"  r = {r_bins[i]:5.1f} - {r_bins[i+1]:5.1f} mm: "
              f"Mean = {mean_r/(2*np.pi)*1000:.6f} milli-waves, "
              f"Std = {std_r/(2*np.pi)*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【分析 5】检查相位网格的二次项系数")
print("=" * 60)

# 拟合相位网格为 a*r^2 + b 的形式
r_flat = r_grid[mask].flatten()
phase_flat = phase_grid[mask].flatten()

# 使用最小二乘拟合
A = np.column_stack([r_flat**2, np.ones_like(r_flat)])
coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_flat, rcond=None)
a_fit, b_fit = coeffs

print(f"相位网格拟合: phase = {a_fit:.9f} * r^2 + {b_fit:.9f}")
print(f"Pilot Beam 公式: phase = {k/(2*R):.9f} * r^2 + 0")
print(f"二次项系数比值: {a_fit / (k/(2*R)) if not np.isinf(R) else 'N/A'}")
print(f"常数项: {b_fit:.9f} rad = {b_fit/(2*np.pi)*1000:.6f} milli-waves")

# 计算拟合残差
phase_fitted = a_fit * r_flat**2 + b_fit
residual_rms = np.std(phase_flat - phase_fitted)
print(f"拟合残差 RMS: {residual_rms:.9f} rad = {residual_rms/(2*np.pi):.9f} waves")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
print(f"""
1. 相位网格与 Pilot Beam 的差异:
   - 平均值: {mean_diff/(2*np.pi)*1000:.6f} milli-waves
   - RMS: {np.std(diff_grid[mask])/(2*np.pi)*1000:.6f} milli-waves
   - 减去平均值后 RMS: {rms_centered/(2*np.pi)*1000:.6f} milli-waves

2. 误差来源分析:
   - 如果差异主要是常数偏移，说明是 Gouy 相位或参考点选择问题
   - 如果差异有径向依赖，说明曲率半径计算有误差
   - 如果拟合残差很小，说明相位网格确实是二次形式

3. 关键发现:
   - 常数项 (b_fit): {b_fit/(2*np.pi)*1000:.6f} milli-waves
   - 这个常数项就是误差的主要来源
   - 它可能来自 PROPER 的 Gouy 相位或参考面处理
""")
