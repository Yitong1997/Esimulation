"""
详细调试 Surface 3 出射面的相位问题

检查：
1. reconstructor 返回的相位
2. amplitude_phase_to_proper 的输入
3. PROPER wfo 中的相位
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

# 修改 HybridElementPropagator 来添加调试输出
from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)

print("=" * 70)
print("详细调试 Surface 3 出射面的相位问题")
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

# 创建 propagator 但不运行完整传播
# 我们要手动追踪 Surface 3 的处理过程
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=512,
    num_rays=150,
)

# 传播到 Surface 3 入射面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]

# 传播到 Surface 3
for i in range(4):  # 0, 1, 2, 3
    propagator._propagate_to_surface(i)

# 找到 Surface 3 入射面和出射面状态
state_s3_entrance = None
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_s3_entrance = state
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state

print("\n【Surface 3 入射面】")
if state_s3_entrance:
    print(f"  state.phase 范围: [{np.min(state_s3_entrance.phase):.6f}, {np.max(state_s3_entrance.phase):.6f}] rad")
    proper_phase = proper.prop_get_phase(state_s3_entrance.proper_wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

print("\n【Surface 3 出射面】")
if state_s3_exit:
    print(f"  state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")
    proper_phase = proper.prop_get_phase(state_s3_exit.proper_wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")
    
    # 检查 PROPER 参数
    wfo = state_s3_exit.proper_wfo
    print(f"\n  PROPER 参数:")
    print(f"    z = {wfo.z * 1e3:.2f} mm")
    print(f"    z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
    print(f"    z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"    reference_surface = {wfo.reference_surface}")

    # 手动检查 wfarr 中的相位
    print(f"\n  手动检查 wfarr:")
    wfarr_shifted = proper.prop_shift_center(wfo.wfarr)
    wfarr_phase = np.angle(wfarr_shifted)
    print(f"    wfarr 相位范围: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")
    
    # 检查 state.phase 和 wfarr 相位的关系
    from hybrid_optical_propagation.state_converter import StateConverter
    from hybrid_optical_propagation.data_models import GridSampling
    
    state_converter = StateConverter(0.55)
    grid_sampling = state_s3_exit.grid_sampling
    proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    print(f"\n  PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")
    
    # 重建相位
    reconstructed_phase = proper_ref_phase + proper_phase
    print(f"  重建相位范围: [{np.min(reconstructed_phase):.6f}, {np.max(reconstructed_phase):.6f}] rad")
    
    # 差异
    diff = state_s3_exit.phase - reconstructed_phase
    print(f"\n  state.phase vs 重建相位:")
    print(f"    差异范围: [{np.min(diff):.6f}, {np.max(diff):.6f}] rad")
    print(f"    差异 RMS: {np.std(diff):.6f} rad")

print("\n" + "=" * 70)
print("问题分析")
print("=" * 70)

print("""
如果 state.phase 和重建相位（PROPER ref + PROPER phase）不一致，
说明 amplitude_phase_to_proper 写入时出了问题。

可能的原因：
1. 写入时的相位和 state.phase 不是同一个
2. 写入后 PROPER 内部做了某些处理
3. 读取时的处理有问题
""")
