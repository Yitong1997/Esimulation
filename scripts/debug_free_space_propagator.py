"""
详细调试 FreeSpacePropagator 的数据流

追踪每一步的相位变化
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
from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.free_space_propagator import FreeSpacePropagator

print("=" * 70)
print("详细调试 FreeSpacePropagator 的数据流")
print("=" * 70)

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

print("\n【步骤 1: Surface 3 出射面状态】")
print(f"  state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")
print(f"  state.amplitude 范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")

wfo = state_s3_exit.proper_wfo
print(f"\n  PROPER wfo:")
print(f"    z = {wfo.z * 1e3:.2f} mm")
print(f"    z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"    reference_surface = {wfo.reference_surface}")

proper_phase = proper.prop_get_phase(wfo)
print(f"    prop_get_phase 范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
