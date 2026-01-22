"""
验证误差出现的位置，对照面形

Surface 3 是第一个反射镜（45°折叠镜）
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

print("=" * 70)
print("验证误差出现的位置")
print("=" * 70)

zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

# 打印表面信息
print("\n表面列表:")
for i, s in enumerate(optical_system):
    mirror_str = "反射镜" if s.is_mirror else "透射面"
    radius_str = f"R={s.radius:.1f}" if not np.isinf(s.radius) else "平面"
    print(f"  Surface {i}: {mirror_str}, {radius_str}, 位置={s.vertex_position}")

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

# 传播
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(len(optical_system)):
    propagator._propagate_to_surface(i)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("\n" + "=" * 70)
print("各表面的 state.phase 与 Pilot Beam 比较")
print("=" * 70)

for state in propagator._surface_states:
    phase = state.phase
    amplitude = state.amplitude
    pb = state.pilot_beam_params
    grid_sampling = state.grid_sampling
    
    pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
    
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    if np.sum(valid_mask) == 0:
        continue
    
    diff = phase - pilot_phase
    diff_valid = diff[valid_mask]
    mean_diff = np.mean(diff_valid)
    diff_centered = diff_valid - mean_diff
    rms_mw = np.std(diff_centered) / (2 * np.pi) * 1000
    
    # 获取表面信息
    idx = state.surface_index
    if idx >= 0 and idx < len(optical_system):
        s = optical_system[idx]
        mirror_str = "反射镜" if s.is_mirror else "透射面"
        radius_str = f"R={s.radius:.1f}" if not np.isinf(s.radius) else "平面"
        surface_info = f"{mirror_str}, {radius_str}"
    else:
        surface_info = "初始"
    
    surface_name = f"Surface {idx}" if idx >= 0 else "Initial"
    print(f"{surface_name:12s} ({state.position:8s}): RMS = {rms_mw:8.4f} milli-waves  [{surface_info}]")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
误差首次出现的位置：
- 如果是 Surface 3 出射面或 Surface 4 入射面，说明误差在第一个反射镜处引入
- 需要检查 HybridElementPropagator 对反射镜的处理
""")
