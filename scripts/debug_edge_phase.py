"""
分析边缘相位问题

关键发现：
- 有效区域内相位范围: [-0.000281, 0.001014] rad
- 网格边缘相位值: ~0.0395 rad
- 边缘相位比有效区域大 39 倍！

问题：为什么边缘相位这么大？
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

print("=" * 80)
print("边缘相位问题分析")
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
pb = state_entrance.pilot_beam_params

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

# 创建网格坐标
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_grid = np.sqrt(X**2 + Y**2)

print("\n" + "=" * 60)
print("【分析 1】振幅分布")
print("=" * 60)

# 高斯光束的 1/e² 半径
w = pb.spot_size_mm
print(f"Pilot Beam 光斑大小 (1/e² 半径): {w:.2f} mm")
print(f"网格半尺寸: {half_size:.2f} mm")
print(f"边缘距离 / 光斑大小: {half_size / w:.2f}")

# 检查振幅分布
print(f"\n振幅分布:")
print(f"  最大值: {np.max(amplitude_grid):.6f}")
print(f"  中心值: {amplitude_grid[grid_size//2, grid_size//2]:.6f}")
print(f"  边缘值 (x=20): {amplitude_grid[grid_size//2, -1]:.9f}")
print(f"  边缘值 (x=-20): {amplitude_grid[grid_size//2, 0]:.9f}")

# 计算边缘振幅相对于中心的比例
edge_amp = amplitude_grid[grid_size//2, -1]
center_amp = amplitude_grid[grid_size//2, grid_size//2]
print(f"  边缘/中心比例: {edge_amp/center_amp:.2e}")

print("\n" + "=" * 60)
print("【分析 2】相位分布与振幅的关系")
print("=" * 60)

# 检查不同振幅阈值下的相位范围
thresholds = [0.01, 0.001, 0.0001, 0.00001, 0]
print("\n不同振幅阈值下的相位范围:")
for thresh in thresholds:
    if thresh > 0:
        mask = amplitude_grid > thresh * np.max(amplitude_grid)
    else:
        mask = np.ones_like(amplitude_grid, dtype=bool)
    
    if np.sum(mask) > 0:
        phase_min = np.min(phase_grid[mask])
        phase_max = np.max(phase_grid[mask])
        r_max = np.max(r_grid[mask])
        print(f"  阈值 {thresh:.5f}: 相位 [{phase_min:.6f}, {phase_max:.6f}] rad, "
              f"最大半径 {r_max:.2f} mm, 点数 {np.sum(mask)}")

print("\n" + "=" * 60)
print("【分析 3】PROPER 相位在低振幅区域的行为")
print("=" * 60)

import proper
wfo = state_entrance.proper_wfo
proper_phase = proper.prop_get_phase(wfo)
proper_amp = proper.prop_get_amplitude(wfo)

print(f"PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
print(f"PROPER 振幅范围: [{np.min(proper_amp):.9f}, {np.max(proper_amp):.6f}]")

# 检查低振幅区域的相位
low_amp_mask = proper_amp < 0.001 * np.max(proper_amp)
if np.sum(low_amp_mask) > 0:
    print(f"\n低振幅区域 (amp < 0.001 * max):")
    print(f"  点数: {np.sum(low_amp_mask)}")
    print(f"  相位范围: [{np.min(proper_phase[low_amp_mask]):.6f}, "
          f"{np.max(proper_phase[low_amp_mask]):.6f}] rad")

print("\n" + "=" * 60)
print("【分析 4】光线采样位置与振幅的关系")
print("=" * 60)

from wavefront_to_rays import WavefrontToRaysSampler
from scipy.interpolate import RegularGridInterpolator

sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)

ray_x, ray_y = sampler.get_ray_positions()

# 插值振幅到光线位置
amp_interp = RegularGridInterpolator(
    (coords, coords),
    amplitude_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
points = np.column_stack([ray_y, ray_x])
ray_amp = amp_interp(points)

print(f"光线位置的振幅范围: [{np.min(ray_amp):.9f}, {np.max(ray_amp):.6f}]")
print(f"振幅 < 0.01 * max 的光线数量: {np.sum(ray_amp < 0.01 * np.max(ray_amp))}")
print(f"振幅 < 0.001 * max 的光线数量: {np.sum(ray_amp < 0.001 * np.max(ray_amp))}")

# 找到振幅最小的光线
min_amp_idx = np.argmin(ray_amp)
print(f"\n振幅最小的光线:")
print(f"  位置: ({ray_x[min_amp_idx]:.6f}, {ray_y[min_amp_idx]:.6f}) mm")
print(f"  振幅: {ray_amp[min_amp_idx]:.9f}")
print(f"  距离: {np.sqrt(ray_x[min_amp_idx]**2 + ray_y[min_amp_idx]**2):.2f} mm")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
print(f"""
问题根源：
1. 光线采样位置包括了网格边缘（r = 20 mm）
2. 在边缘位置，振幅非常小（{ray_amp[min_amp_idx]:.2e}）
3. 但相位值仍然很大（~0.04 rad）
4. 这些低振幅区域的相位是 PROPER 数值计算的结果，不是物理上有意义的

解决方案：
1. 只在有效振幅区域内采样光线
2. 或者在计算误差时，只考虑有效振幅区域内的光线
3. 或者使用振幅加权的误差计算

这不是真正的"误差"，而是在低振幅区域比较相位没有物理意义。
""")
