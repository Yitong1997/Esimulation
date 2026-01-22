"""
分析 Gouy 相位对误差计算的影响

关键问题：
- 相位网格与 Pilot Beam 的差异是常数偏移（Gouy 相位）
- 但 test_sampler_error_source.py 显示的误差是 0.6466 milli-waves
- 这个误差是怎么来的？

假设：误差计算方式不正确，应该使用相对于主光线的相位
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
print("Gouy 相位对误差计算的影响分析")
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

print(f"Pilot Beam 曲率半径: {R:.2f} mm")
print(f"曲率半径是否接近无穷大: {R > 1e8}")

# 创建采样器
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

# 光线相位（从 OPD 转换）
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase = k * ray_opd_mm

# Pilot Beam 相位
r_sq_rays = ray_x**2 + ray_y**2
pilot_phase_rays = k * r_sq_rays / (2 * R) if not np.isinf(R) else np.zeros_like(r_sq_rays)

# 找到主光线
distances = np.sqrt(ray_x**2 + ray_y**2)
chief_idx = np.argmin(distances)

print(f"\n主光线位置: ({ray_x[chief_idx]:.6f}, {ray_y[chief_idx]:.6f}) mm")
print(f"主光线相位: {ray_phase[chief_idx]:.9f} rad")
print(f"主光线 Pilot Beam 相位: {pilot_phase_rays[chief_idx]:.9f} rad")

print("\n" + "=" * 60)
print("【方法 1】原始误差计算（绝对相位）")
print("=" * 60)

diff_absolute = ray_phase - pilot_phase_rays
rms_absolute = np.std(diff_absolute) / (2 * np.pi)
mean_absolute = np.mean(diff_absolute) / (2 * np.pi)
print(f"差异平均值: {mean_absolute*1000:.6f} milli-waves")
print(f"差异 RMS: {rms_absolute*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【方法 2】相对相位误差（相对于主光线）")
print("=" * 60)

# 相对于主光线的相位
ray_phase_relative = ray_phase - ray_phase[chief_idx]
pilot_phase_relative = pilot_phase_rays - pilot_phase_rays[chief_idx]

diff_relative = ray_phase_relative - pilot_phase_relative
rms_relative = np.std(diff_relative) / (2 * np.pi)
mean_relative = np.mean(diff_relative) / (2 * np.pi)
print(f"差异平均值: {mean_relative*1000:.6f} milli-waves")
print(f"差异 RMS: {rms_relative*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【方法 3】减去常数偏移后的误差")
print("=" * 60)

# 减去常数偏移（Gouy 相位）
gouy_offset = np.mean(diff_absolute)
diff_corrected = diff_absolute - gouy_offset
rms_corrected = np.std(diff_corrected) / (2 * np.pi)
print(f"Gouy 偏移: {gouy_offset/(2*np.pi)*1000:.6f} milli-waves")
print(f"修正后 RMS: {rms_corrected*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【分析】为什么 RMS 误差这么大？")
print("=" * 60)

# 检查差异的分布
print(f"\n差异统计:")
print(f"  最小值: {np.min(diff_absolute)/(2*np.pi)*1000:.6f} milli-waves")
print(f"  最大值: {np.max(diff_absolute)/(2*np.pi)*1000:.6f} milli-waves")
print(f"  范围: {(np.max(diff_absolute)-np.min(diff_absolute))/(2*np.pi)*1000:.6f} milli-waves")

# 检查相位范围
print(f"\n相位范围:")
print(f"  光线相位: [{np.min(ray_phase):.6f}, {np.max(ray_phase):.6f}] rad")
print(f"  Pilot Beam 相位: [{np.min(pilot_phase_rays):.6f}, {np.max(pilot_phase_rays):.6f}] rad")

# 检查 Pilot Beam 相位是否正确
print(f"\n验证 Pilot Beam 相位计算:")
print(f"  曲率半径 R = {R:.2f} mm")
print(f"  最大 r² = {np.max(r_sq_rays):.2f} mm²")
print(f"  期望最大相位 = k * r²_max / (2R) = {k * np.max(r_sq_rays) / (2 * R):.9f} rad")
print(f"  实际最大 Pilot Beam 相位 = {np.max(pilot_phase_rays):.9f} rad")

# 检查光线相位的来源
print(f"\n光线相位来源分析:")
print(f"  光线 OPD 范围: [{np.min(ray_opd_mm):.9f}, {np.max(ray_opd_mm):.9f}] mm")
print(f"  光线相位范围: [{np.min(ray_phase):.9f}, {np.max(ray_phase):.9f}] rad")

# 直接从相位网格插值验证
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
interpolator = RegularGridInterpolator(
    (coords, coords),
    state_entrance.phase,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
points = np.column_stack([ray_y, ray_x])
phase_interpolated = interpolator(points)

print(f"\n直接插值相位:")
print(f"  范围: [{np.min(phase_interpolated):.9f}, {np.max(phase_interpolated):.9f}] rad")
print(f"  与光线相位差异: {np.max(np.abs(ray_phase - phase_interpolated)):.9f} rad")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
print(f"""
1. 绝对相位误差 RMS: {rms_absolute*1000:.6f} milli-waves
2. 相对相位误差 RMS: {rms_relative*1000:.6f} milli-waves
3. 减去 Gouy 偏移后 RMS: {rms_corrected*1000:.6f} milli-waves

关键发现：
- 如果 RMS 误差在方法 2 和 3 中显著降低，说明误差主要来自 Gouy 相位偏移
- 如果 RMS 误差没有显著变化，说明误差来自其他来源

实际情况：
- 曲率半径 R = {R:.2f} mm（非常大，接近平面波）
- 当 R 很大时，Pilot Beam 相位 ≈ 0
- 所以误差主要来自光线相位本身（即相位网格的值）
""")
