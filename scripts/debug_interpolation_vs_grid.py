"""
分析插值精度问题

关键问题：
- PROPER 计算的相位网格是精确的
- 为什么插值后在边缘区域会有误差？
- 振幅不应该影响相位插值精度

需要验证：
1. 相位网格在边缘的值是否正确（与 Pilot Beam 一致）
2. 插值本身是否引入误差
3. 误差的真正来源是什么
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
print("插值精度 vs 网格精度分析")
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
print(f"物理尺寸: {physical_size_mm} mm")
