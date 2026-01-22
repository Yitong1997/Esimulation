"""
调试 reference_surface 的设置

关键问题：在实际系统中，reference_surface 是如何被设置的？
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
print("调试 reference_surface 的设置")
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

print("\n追踪每个状态的 reference_surface:")
print("-" * 70)

for i in range(5):  # 传播到 Surface 4
    propagator._propagate_to_surface(i)
    
    # 打印每个状态的 reference_surface
    for state in propagator._surface_states:
        if hasattr(state, 'proper_wfo') and state.proper_wfo is not None:
            wfo = state.proper_wfo
            print(f"Surface {state.surface_index} {state.position}:")
            print(f"  z = {wfo.z * 1e3:.2f} mm")
            print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
            print(f"  z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.2f} mm")
            print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
            print(f"  reference_surface = {wfo.reference_surface}")
            print(f"  beam_type_old = {wfo.beam_type_old}")
            
            # 检查 rayleigh_factor 条件
            threshold = proper.rayleigh_factor * wfo.z_Rayleigh
            z_diff = abs(wfo.z - wfo.z_w0)
            print(f"  |z - z_w0| < rayleigh_factor * z_R: {z_diff < threshold}")
            print()
