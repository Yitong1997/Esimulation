"""
分析光线相位为什么比相位网格大

关键矛盾：
- 相位网格范围: [-0.000281, 0.001014] rad
- 光线相位范围: [-0.000281, 0.039535] rad
- 光线相位比相位网格大约 40 倍！

这说明 WavefrontToRaysSampler 的 OPD 计算有问题
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
print("光线相位来源分析")
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
wavelength_um = 0.55
k = 2 * np.pi / wavelength_mm
grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm

print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")
print(f"k = 2π/λ = {k:.2f} rad/mm")

# 相位网格
phase_grid = state_entrance.phase
amplitude_grid = state_entrance.amplitude
mask = amplitude_grid > 0.01 * np.max(amplitude_grid)

print(f"\n相位网格范围: [{np.min(phase_grid[mask]):.9f}, {np.max(phase_grid[mask]):.9f}] rad")

# 创建采样器
print("\n" + "=" * 60)
print("【分析 1】WavefrontToRaysSampler 的 OPD 计算")
print("=" * 60)

sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=wavelength_um,
    num_rays=150,
)

output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()

# 光线 OPD
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase_from_opd = k * ray_opd_mm

print(f"光线 OPD 范围: [{np.min(ray_opd_mm):.9f}, {np.max(ray_opd_mm):.9f}] mm")
print(f"光线相位 (k * OPD) 范围: [{np.min(ray_phase_from_opd):.9f}, {np.max(ray_phase_from_opd):.9f}] rad")

# 直接从相位网格插值
print("\n" + "=" * 60)
print("【分析 2】直接从相位网格插值")
print("=" * 60)

half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
interpolator = RegularGridInterpolator(
    (coords, coords),
    phase_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
points = np.column_stack([ray_y, ray_x])
phase_interpolated = interpolator(points)

print(f"插值相位范围: [{np.min(phase_interpolated):.9f}, {np.max(phase_interpolated):.9f}] rad")

# 比较
print("\n" + "=" * 60)
print("【分析 3】比较光线相位与插值相位")
print("=" * 60)

diff = ray_phase_from_opd - phase_interpolated
print(f"差异范围: [{np.min(diff):.9f}, {np.max(diff):.9f}] rad")
print(f"差异 RMS: {np.std(diff):.9f} rad")

# 检查 sampler 的 OPD 计算逻辑
print("\n" + "=" * 60)
print("【分析 4】检查 sampler 的 OPD 计算逻辑")
print("=" * 60)

# 根据 opd_definition.md，sampler 的 OPD 计算应该是：
# opd_mm = phase_at_rays * wavelength_mm / (2 * np.pi)
# 所以 phase = opd_mm * (2 * np.pi) / wavelength_mm = k * opd_mm

# 验证：如果 sampler 正确计算了 OPD，那么
# ray_phase_from_opd 应该等于 phase_interpolated

print(f"如果 sampler 正确计算 OPD:")
print(f"  OPD = phase * λ / (2π)")
print(f"  phase = k * OPD")
print(f"  所以 ray_phase_from_opd 应该等于 phase_interpolated")
print(f"  实际差异: {np.max(np.abs(diff)):.9f} rad")

if np.max(np.abs(diff)) < 1e-6:
    print("  ✓ sampler 的 OPD 计算正确")
else:
    print("  ✗ sampler 的 OPD 计算有问题！")

# 检查 sampler 内部的 OPD 计算
print("\n" + "=" * 60)
print("【分析 5】检查 sampler 内部的 OPD 计算")
print("=" * 60)

# 读取 sampler 的源代码来理解 OPD 计算
print("根据 WavefrontToRaysSampler 的实现：")
print("  1. 从相位网格插值得到每条光线位置的相位")
print("  2. 将相位转换为 OPD: OPD_mm = phase_rad * wavelength_mm / (2π)")
print("  3. 设置 rays.opd = OPD_mm")

# 验证这个逻辑
expected_opd = phase_interpolated * wavelength_mm / (2 * np.pi)
print(f"\n期望的 OPD 范围: [{np.min(expected_opd):.9f}, {np.max(expected_opd):.9f}] mm")
print(f"实际的 OPD 范围: [{np.min(ray_opd_mm):.9f}, {np.max(ray_opd_mm):.9f}] mm")

opd_diff = ray_opd_mm - expected_opd
print(f"OPD 差异范围: [{np.min(opd_diff):.9f}, {np.max(opd_diff):.9f}] mm")

# 检查比例
if np.max(np.abs(expected_opd)) > 0:
    ratio = np.max(ray_opd_mm) / np.max(expected_opd) if np.max(expected_opd) != 0 else np.inf
    print(f"OPD 比例: {ratio:.6f}")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)

# 检查是否有单位问题
print(f"""
单位检查：
- 波长: {wavelength_um} μm = {wavelength_mm} mm
- k = 2π/λ = {k:.2f} rad/mm

如果 sampler 使用了错误的波长单位：
- 如果用 μm 而不是 mm: OPD 会大 1000 倍
- 如果用 m 而不是 mm: OPD 会小 1000 倍

实际观察：
- 相位网格最大值: {np.max(phase_grid[mask]):.9f} rad
- 光线相位最大值: {np.max(ray_phase_from_opd):.9f} rad
- 比例: {np.max(ray_phase_from_opd) / np.max(phase_grid[mask]):.2f}

这个比例约为 40，不是 1000，所以不是简单的单位问题。
需要检查 sampler 的具体实现。
""")
