"""
深入分析 WavefrontToRaysSampler 的误差来源

已知：
1. 相位网格 vs Pilot Beam 解析解：~0.000001 waves（几乎无误差）
2. 插值方法本身精度：~0.000001 waves（几乎无误差）
3. WavefrontToRaysSampler 输出：~0.0004 waves（有误差）

问题：误差来自哪里？
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
from wavefront_to_rays import WavefrontToRaysSampler
from scipy.interpolate import RegularGridInterpolator

print("=" * 80)
print("WavefrontToRaysSampler 误差来源分析")
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

# =============================================================================
# 步骤 1: 创建采样器并获取光线数据
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 1】创建采样器")
print("=" * 60)

sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)

output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
n_rays = len(ray_x)
print(f"采样光线数量: {n_rays}")

# =============================================================================
# 步骤 2: 分析光线 OPD 的来源
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 2】分析光线 OPD 的来源")
print("=" * 60)

# 光线 OPD（来自 sampler）
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase_from_opd = k * ray_opd_mm

print(f"光线 OPD 范围: [{np.min(ray_opd_mm):.6f}, {np.max(ray_opd_mm):.6f}] mm")
print(f"光线相位范围: [{np.min(ray_phase_from_opd):.6f}, {np.max(ray_phase_from_opd):.6f}] rad")

# =============================================================================
# 步骤 3: 直接从相位网格插值（与 sampler 使用相同的方法）
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 3】直接从相位网格插值")
print("=" * 60)

phase_grid = state_entrance.phase
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)

# 使用与 sampler 相同的插值方法
interpolator = RegularGridInterpolator(
    (coords, coords),
    phase_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

points = np.column_stack([ray_y, ray_x])
phase_interpolated = interpolator(points)

print(f"插值相位范围: [{np.min(phase_interpolated):.6f}, {np.max(phase_interpolated):.6f}] rad")

# 比较 sampler 的 OPD 与直接插值的相位
# sampler 的 OPD 计算: opd_mm = phase_at_rays * wavelength_mm / (2 * np.pi)
# 所以 phase = opd_mm * (2 * np.pi) / wavelength_mm = k * opd_mm
diff_sampler_vs_interp = ray_phase_from_opd - phase_interpolated
rms_sampler_vs_interp = np.std(diff_sampler_vs_interp) / (2 * np.pi)
print(f"\nsampler OPD 转相位 vs 直接插值相位:")
print(f"  差异范围: [{np.min(diff_sampler_vs_interp):.6f}, {np.max(diff_sampler_vs_interp):.6f}] rad")
print(f"  RMS 差异: {rms_sampler_vs_interp:.6f} waves")

# =============================================================================
# 步骤 4: 计算 Pilot Beam 在光线位置的相位
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 4】计算 Pilot Beam 在光线位置的相位")
print("=" * 60)

r_sq_rays = ray_x**2 + ray_y**2
if np.isinf(R):
    pilot_phase_rays = np.zeros_like(r_sq_rays)
else:
    pilot_phase_rays = k * r_sq_rays / (2 * R)

print(f"Pilot Beam 相位范围: [{np.min(pilot_phase_rays):.6f}, {np.max(pilot_phase_rays):.6f}] rad")

# =============================================================================
# 步骤 5: 比较各种相位
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 5】比较各种相位")
print("=" * 60)

# 1. 插值相位 vs Pilot Beam
diff_interp_vs_pilot = phase_interpolated - pilot_phase_rays
rms_interp_vs_pilot = np.std(diff_interp_vs_pilot) / (2 * np.pi)
print(f"插值相位 vs Pilot Beam:")
print(f"  RMS 差异: {rms_interp_vs_pilot:.6f} waves")

# 2. sampler OPD 转相位 vs Pilot Beam
diff_opd_vs_pilot = ray_phase_from_opd - pilot_phase_rays
rms_opd_vs_pilot = np.std(diff_opd_vs_pilot) / (2 * np.pi)
print(f"\nsampler OPD 转相位 vs Pilot Beam:")
print(f"  RMS 差异: {rms_opd_vs_pilot:.6f} waves")

# 3. 在网格上计算 Pilot Beam 相位
X, Y = np.meshgrid(coords, coords)
r_sq_grid = X**2 + Y**2
if np.isinf(R):
    pilot_phase_grid = np.zeros_like(r_sq_grid)
else:
    pilot_phase_grid = k * r_sq_grid / (2 * R)

# 4. 网格相位 vs Pilot Beam 网格相位
amplitude_grid = state_entrance.amplitude
mask = amplitude_grid > 0.01 * np.max(amplitude_grid)
diff_grid = phase_grid - pilot_phase_grid
rms_grid = np.std(diff_grid[mask]) / (2 * np.pi)
print(f"\n网格相位 vs Pilot Beam 网格相位:")
print(f"  RMS 差异: {rms_grid:.6f} waves")

# 5. 插值 Pilot Beam 网格相位到光线位置
pilot_interp = RegularGridInterpolator(
    (coords, coords),
    pilot_phase_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
pilot_phase_interpolated = pilot_interp(points)

diff_pilot_interp_vs_analytic = pilot_phase_interpolated - pilot_phase_rays
rms_pilot_interp = np.std(diff_pilot_interp_vs_analytic) / (2 * np.pi)
print(f"\n插值 Pilot Beam 网格 vs Pilot Beam 解析:")
print(f"  RMS 差异: {rms_pilot_interp:.6f} waves")

# =============================================================================
# 步骤 6: 分析误差的空间分布
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 6】分析误差的空间分布")
print("=" * 60)

# 计算每条光线的误差
errors = diff_opd_vs_pilot / (2 * np.pi)  # 转换为 waves

# 按径向距离分组
r_rays = np.sqrt(ray_x**2 + ray_y**2)
r_bins = np.linspace(0, 20, 11)

print("\n径向误差分布:")
print("-" * 50)
for i in range(len(r_bins) - 1):
    mask_r = (r_rays >= r_bins[i]) & (r_rays < r_bins[i+1])
    if np.sum(mask_r) > 0:
        rms_r = np.std(errors[mask_r])
        mean_r = np.mean(errors[mask_r])
        print(f"  r = {r_bins[i]:5.1f} - {r_bins[i+1]:5.1f} mm: "
              f"RMS = {rms_r*1000:.4f} milli-waves, "
              f"Mean = {mean_r*1000:.4f} milli-waves, "
              f"N = {np.sum(mask_r)}")

# =============================================================================
# 步骤 7: 检查主光线
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 7】检查主光线")
print("=" * 60)

distances = np.sqrt(ray_x**2 + ray_y**2)
chief_idx = np.argmin(distances)

print(f"主光线索引: {chief_idx}")
print(f"主光线位置: ({ray_x[chief_idx]:.6f}, {ray_y[chief_idx]:.6f}) mm")
print(f"主光线 OPD: {ray_opd_mm[chief_idx]:.9f} mm")
print(f"主光线相位 (from OPD): {ray_phase_from_opd[chief_idx]:.9f} rad")
print(f"主光线相位 (插值): {phase_interpolated[chief_idx]:.9f} rad")
print(f"主光线 Pilot Beam 相位: {pilot_phase_rays[chief_idx]:.9f} rad")

# =============================================================================
# 步骤 8: 检查边缘光线
# =============================================================================
print("\n" + "=" * 60)
print("【步骤 8】检查边缘光线")
print("=" * 60)

# 找到最远的光线
edge_idx = np.argmax(distances)
print(f"边缘光线索引: {edge_idx}")
print(f"边缘光线位置: ({ray_x[edge_idx]:.6f}, {ray_y[edge_idx]:.6f}) mm")
print(f"边缘光线距离: {distances[edge_idx]:.6f} mm")
print(f"边缘光线 OPD: {ray_opd_mm[edge_idx]:.9f} mm")
print(f"边缘光线相位 (from OPD): {ray_phase_from_opd[edge_idx]:.9f} rad")
print(f"边缘光线相位 (插值): {phase_interpolated[edge_idx]:.9f} rad")
print(f"边缘光线 Pilot Beam 相位: {pilot_phase_rays[edge_idx]:.9f} rad")
print(f"边缘光线误差: {errors[edge_idx]*1000:.6f} milli-waves")

# =============================================================================
# 结论
# =============================================================================
print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)

print(f"""
误差来源分析结果：

1. sampler OPD 转相位 vs 直接插值相位: {rms_sampler_vs_interp:.6f} waves
   → sampler 的 OPD 计算与直接插值一致

2. 插值相位 vs Pilot Beam 解析: {rms_interp_vs_pilot:.6f} waves
   → 这是主要误差来源！

3. 网格相位 vs Pilot Beam 网格相位: {rms_grid:.6f} waves
   → 网格上的相位与 Pilot Beam 几乎一致

4. 插值 Pilot Beam 网格 vs Pilot Beam 解析: {rms_pilot_interp:.6f} waves
   → 插值 Pilot Beam 网格也有误差！

关键发现：
- 误差来自于**插值过程本身**，而不是相位网格的内容
- 当从网格插值到任意位置时，即使网格上的值是精确的，
  插值结果也会有误差
- 这是因为 Pilot Beam 相位是 r² 的函数，而线性插值
  无法精确表示二次函数

解决方案：
- 使用更高阶的插值方法（如 cubic 或 quintic）
- 或者直接使用 Pilot Beam 解析公式计算参考相位
""")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
