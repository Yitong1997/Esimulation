"""
分析插值范围问题

关键发现：
- 相位网格范围: [-0.000281, 0.001014] rad
- 插值相位范围: [-0.000281, 0.039535] rad
- 插值结果比网格值大 39 倍！

这说明插值器在某些位置返回了异常值
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
print("插值范围问题分析")
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
amplitude_grid = state_entrance.amplitude

print(f"网格分辨率: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")

# 创建采样器
sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)

ray_x, ray_y = sampler.get_ray_positions()

print("\n" + "=" * 60)
print("【分析 1】光线位置范围")
print("=" * 60)

print(f"光线 X 范围: [{np.min(ray_x):.6f}, {np.max(ray_x):.6f}] mm")
print(f"光线 Y 范围: [{np.min(ray_y):.6f}, {np.max(ray_y):.6f}] mm")

half_size = physical_size_mm / 2
print(f"网格坐标范围: [{-half_size:.6f}, {half_size:.6f}] mm")

# 检查是否有光线超出网格范围
out_of_range_x = (ray_x < -half_size) | (ray_x > half_size)
out_of_range_y = (ray_y < -half_size) | (ray_y > half_size)
out_of_range = out_of_range_x | out_of_range_y

print(f"超出范围的光线数量: {np.sum(out_of_range)}")

print("\n" + "=" * 60)
print("【分析 2】检查相位网格的实际范围")
print("=" * 60)

mask = amplitude_grid > 0.01 * np.max(amplitude_grid)
print(f"有效区域内相位范围: [{np.min(phase_grid[mask]):.9f}, {np.max(phase_grid[mask]):.9f}] rad")
print(f"整个网格相位范围: [{np.min(phase_grid):.9f}, {np.max(phase_grid):.9f}] rad")

# 检查网格边缘的相位值
print(f"\n网格边缘相位值:")
print(f"  左边缘 (x=-20): {phase_grid[grid_size//2, 0]:.9f} rad")
print(f"  右边缘 (x=+20): {phase_grid[grid_size//2, -1]:.9f} rad")
print(f"  上边缘 (y=+20): {phase_grid[-1, grid_size//2]:.9f} rad")
print(f"  下边缘 (y=-20): {phase_grid[0, grid_size//2]:.9f} rad")
print(f"  角落 (x=+20, y=+20): {phase_grid[-1, -1]:.9f} rad")

print("\n" + "=" * 60)
print("【分析 3】直接插值测试")
print("=" * 60)

# 创建插值器
coords = np.linspace(-half_size, half_size, grid_size)
interpolator = RegularGridInterpolator(
    (coords, coords),
    phase_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

# 测试几个特定位置
test_points = [
    (0, 0),      # 中心
    (10, 0),     # x=10
    (0, 10),     # y=10
    (10, 10),    # 对角
    (20, 0),     # 边缘
    (20, 20),    # 角落
]

print("测试点插值结果:")
for x, y in test_points:
    # 注意：RegularGridInterpolator 的输入顺序是 (y, x)
    result = interpolator([[y, x]])[0]
    print(f"  ({x:5.1f}, {y:5.1f}) mm: {result:.9f} rad")

print("\n" + "=" * 60)
print("【分析 4】检查光线位置对应的相位值")
print("=" * 60)

# 找到相位最大的光线
output_rays = sampler.get_output_rays()
ray_opd_mm = np.asarray(output_rays.opd)
wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm
ray_phase = k * ray_opd_mm

max_phase_idx = np.argmax(ray_phase)
print(f"相位最大的光线:")
print(f"  索引: {max_phase_idx}")
print(f"  位置: ({ray_x[max_phase_idx]:.6f}, {ray_y[max_phase_idx]:.6f}) mm")
print(f"  相位: {ray_phase[max_phase_idx]:.9f} rad")

# 直接插值该位置
interp_phase = interpolator([[ray_y[max_phase_idx], ray_x[max_phase_idx]]])[0]
print(f"  直接插值相位: {interp_phase:.9f} rad")

# 检查该位置附近的网格值
x_idx = int((ray_x[max_phase_idx] + half_size) / physical_size_mm * grid_size)
y_idx = int((ray_y[max_phase_idx] + half_size) / physical_size_mm * grid_size)
x_idx = np.clip(x_idx, 0, grid_size-1)
y_idx = np.clip(y_idx, 0, grid_size-1)
print(f"  最近网格索引: ({x_idx}, {y_idx})")
print(f"  最近网格相位: {phase_grid[y_idx, x_idx]:.9f} rad")

print("\n" + "=" * 60)
print("【分析 5】检查 sampler 使用的相位网格")
print("=" * 60)

# sampler 可能对相位网格做了处理
# 检查 sampler 内部的相位
print("检查 sampler 是否修改了相位网格...")

# 直接从 state_entrance.phase 插值
direct_interp = RegularGridInterpolator(
    (coords, coords),
    state_entrance.phase,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

points = np.column_stack([ray_y, ray_x])
direct_phase = direct_interp(points)

print(f"直接插值相位范围: [{np.min(direct_phase):.9f}, {np.max(direct_phase):.9f}] rad")
print(f"光线相位范围: [{np.min(ray_phase):.9f}, {np.max(ray_phase):.9f}] rad")
print(f"差异: {np.max(np.abs(ray_phase - direct_phase)):.9f} rad")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
