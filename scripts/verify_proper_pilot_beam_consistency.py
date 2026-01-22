"""
验证 PROPER 传播与 Pilot Beam 在入射面的精度一致性

目的：确认 PROPER 物理光学传播是准确的，误差不在这里
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
print("验证 PROPER 传播与 Pilot Beam 的一致性")
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

# 传播到各个表面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(6):
    propagator._propagate_to_surface(i)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("\n各表面入射面处 state.phase 与 Pilot Beam 的比较:")
print("-" * 70)

for state in propagator._surface_states:
    if state.position not in ['entrance', 'source']:
        continue
    
    grid_size = state.grid_sampling.grid_size
    physical_size_mm = state.grid_sampling.physical_size_mm
    phase_grid = state.phase
    pb = state.pilot_beam_params
    R = pb.curvature_radius_mm
    
    # 创建网格坐标
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    r = np.sqrt(r_sq)
    
    # Pilot Beam 相位
    if np.isinf(R):
        pilot_phase = np.zeros_like(r_sq)
    else:
        pilot_phase = k * r_sq / (2 * R)
    
    # 计算振幅掩模（只在有效区域比较）
    amplitude = state.amplitude
    valid_mask = amplitude > 0.01 * np.max(amplitude)
    
    # 计算差异
    diff = phase_grid - pilot_phase
    
    # 在有效区域内的差异统计
    if np.sum(valid_mask) > 0:
        diff_valid = diff[valid_mask]
        # 减去常数偏移（Gouy 相位等）
        mean_diff = np.mean(diff_valid)
        diff_centered = diff_valid - mean_diff
        rms_mw = np.std(diff_centered) / (2 * np.pi) * 1000
        max_mw = np.max(np.abs(diff_centered)) / (2 * np.pi) * 1000
        
        surface_name = f"Surface {state.surface_index}" if state.surface_index >= 0 else "Initial"
        print(f"{surface_name:15s} ({state.position:8s}): "
              f"RMS = {rms_mw:8.4f} milli-waves, "
              f"Max = {max_mw:8.4f} milli-waves, "
              f"R = {R:10.2f} mm")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
如果上述 RMS 误差都在 0.01 milli-waves 以下，说明：
1. PROPER 物理光学传播非常准确
2. state.phase 与 Pilot Beam 参考相位高度一致
3. 误差来源不在 PROPER 传播，而在几何光线追迹流程中
""")
