"""简化版 Surface 3 全流程误差分析"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'optiland-master')

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Starting...")
sys.stdout.flush()

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)

print("Imports done")
sys.stdout.flush()

zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)
print(f"Loaded {len(optical_system)} surfaces")
sys.stdout.flush()

source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=256,
    physical_size_mm=40.0,
)
print("Source created")
sys.stdout.flush()
