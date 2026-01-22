"""
详细调试 Surface 3 出射面的 PROPER wfo

问题：
- state.phase 范围: [-0.038, 0.0] rad
- PROPER 相位范围: [-3.14, 3.14] rad
- 两者不一致！

需要检查：
1. HybridElementPropagator 返回的 state 中的 proper_wfo 是否正确
2. proper_wfo 中的相位是否与 state.phase 一致
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
from hybrid_optical_propagation.data_models import GridSampling

print("=" * 70)
print("详细调试 Surface 3 出射面的 PROPER wfo")
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
