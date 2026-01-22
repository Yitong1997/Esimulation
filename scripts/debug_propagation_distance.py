"""
调试传播距离

检查 Surface 3 出射面到 Surface 4 入射面的传播距离
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
print("调试传播距离")
print("=" * 70)

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

# 传播到 Surface 3 出射面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):
    propagator._propagate_to_surface(i)

# 找到 Surface 3 出射面状态
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
        break

print(f"\nSurface 3 出射面:")
print(f"  wfo.z = {state_s3_exit.proper_wfo.z * 1e3:.2f} mm")
print(f"  wfo.z_w0 = {state_s3_exit.proper_wfo.z_w0 * 1e3:.2f} mm")

# 检查 wfarr 的相位
phase_s3 = proper.prop_get_phase(state_s3_exit.proper_wfo)
print(f"  prop_get_phase 范围: [{np.min(phase_s3):.6f}, {np.max(phase_s3):.6f}] rad")

# 继续传播到 Surface 4
propagator._propagate_to_surface(4)

# 找到 Surface 4 入射面状态
state_s4_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 4 and state.position == 'entrance':
        state_s4_entrance = state
        break

print(f"\nSurface 4 入射面:")
print(f"  wfo.z = {state_s4_entrance.proper_wfo.z * 1e3:.2f} mm")
print(f"  wfo.z_w0 = {state_s4_entrance.proper_wfo.z_w0 * 1e3:.2f} mm")

# 检查 wfarr 的相位
phase_s4 = proper.prop_get_phase(state_s4_entrance.proper_wfo)
print(f"  prop_get_phase 范围: [{np.min(phase_s4):.6f}, {np.max(phase_s4):.6f}] rad")

# 检查 wfo 是否是同一个对象
print(f"\nwfo 是否是同一个对象: {state_s3_exit.proper_wfo is state_s4_entrance.proper_wfo}")

# 检查传播距离
# Surface 3 的厚度应该是 100 mm
print(f"\n检查 ZMX 文件中 Surface 3 的厚度...")
