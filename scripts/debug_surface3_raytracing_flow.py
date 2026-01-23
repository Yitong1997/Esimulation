"""
调试 Surface 3 的几何光线追迹流程

误差出现在 Surface 4 入射面，说明问题在 Surface 3 的处理中。
Surface 3 是第一个 45° 折叠镜。

需要检查：
1. Surface 3 入射面的相位是否正确（应该与 Pilot Beam 一致）
2. Surface 3 出射面的相位是否正确
3. 从 Surface 3 出射面传播到 Surface 4 入射面后相位是否正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import proper

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)

print("=" * 70)
print("调试 Surface 3 的几何光线追迹流程")
print("=" * 70)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

# 打印光学系统信息
print("\n光学系统表面列表:")
for i, surf in enumerate(optical_system):
    print(f"  Surface {i}: {surf.surface_type}, is_mirror={surf.is_mirror}")

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

# 传播到 Surface 4
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(5):  # 传播到 Surface 4
    propagator._propagate_to_surface(i)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("\n" + "=" * 70)
print("各表面状态")
print("=" * 70)

for state in propagator._surface_states:
    phase = state.phase
    amplitude = state.amplitude
    pb = state.pilot_beam_params
    grid_sampling = state.grid_sampling
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    diff = phase - pilot