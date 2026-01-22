"""
调试残差相位过大的来源

之前的端到端测试显示残差相位高达 628320 rad，这是一个巨大的数字。
需要找出这个数字是从哪里来的。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.state_converter import StateConverter

print("=" * 70)
print("调试残差相位过大的来源")
print("=" * 70)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

# 创建光源
source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=256,
    physical_size_mm=40.0,
)

# 创建传播器
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=256,
    num_rays=150,
)

# 执行传播，捕获警告
print("\n执行传播...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = propagator.propagate()
    
    print(f"\n捕获到 {len(w)} 个警告:")
    for i, warning in enumerate(w):
        print(f"\n警告 {i+1}:")
        print(f"  消息: {warning.message}")
        print(f"  文件: {warning.filename}:{warning.lineno}")

# 检查每个状态的相位范围
print("\n" + "=" * 70)
print("检查每个状态的相位范围")
print("=" * 70)

for state in result.surface_states:
    name = f"Surface_{state.surface_index}_{state.position}"
    
    # 检查相位范围
    phase_min = np.min(state.phase)
    phase_max = np.max(state.phase)
    phase_range = phase_max - phase_min
    
    # 检查 Pilot Beam 参考相位
    pilot_phase = state.pilot_beam_params.compute_phase_grid(
        state.grid_sampling.grid_size,
        state.grid_sampling.physical_size_mm,
    )
    pilot_min = np.min(pilot_phase)
    pilot_max = np.max(pilot_phase)
    
    # 计算残差
    residual = state.phase - pilot_phase
    residual_min = np.min(residual)
    residual_max = np.max(residual)
    
    print(f"\n{name}:")
    print(f"  相位范围: [{phase_min:.2f}, {phase_max:.2f}] rad")
    print(f"  Pilot Beam 范围: [{pilot_min:.2f}, {pilot_max:.2f}] rad")
    print(f"  残差范围: [{residual_min:.2f}, {residual_max:.2f}] rad")
    
    # 检查 PROPER 参考面相位
    converter = StateConverter(0.55)
    proper_ref_phase = converter.compute_proper_reference_phase(
        state.proper_wfo, state.grid_sampling
    )
    proper_ref_min = np.min(proper_ref_phase)
    proper_ref_max = np.max(proper_ref_phase)
    
    # 计算相对于 PROPER 参考面的残差
    proper_residual = state.phase - proper_ref_phase
    proper_residual_min = np.min(proper_residual)
    proper_residual_max = np.max(proper_residual)
    
    print(f"  PROPER 参考面范围: [{proper_ref_min:.2f}, {proper_ref_max:.2f}] rad")
    print(f"  PROPER 残差范围: [{proper_residual_min:.2f}, {proper_residual_max:.2f}] rad")
    
    # 检查 wfo 参数
    wfo = state.proper_wfo
    print(f"  wfo.z = {wfo.z * 1e3:.2f} mm")
    print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
