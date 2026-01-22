"""
调试 Surface 3 出射面的数据流

追踪：
1. HybridElementPropagator 输出的 exit_phase
2. amplitude_phase_to_proper 写入 PROPER
3. prop_get_phase 读取的相位
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
print("调试 Surface 3 出射面的数据流")
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
for i in range(4):  # 传播到 Surface 3
    propagator._propagate_to_surface(i)

# 找到 Surface 3 出射面状态
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'exit':
        state_s3_exit = state
        break

if state_s3_exit is None:
    print("错误：未找到 Surface 3 出射面状态")
    sys.exit(1)

print("\n【Surface 3 出射面状态】")
print(f"  state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")
print(f"  state.amplitude 范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")

# 检查 Pilot Beam 参数
pb = state_s3_exit.pilot_beam_params
print(f"\n  Pilot Beam 参数:")
print(f"    waist_position_mm = {pb.waist_position_mm:.2f} mm")
print(f"    curvature_radius_mm = {pb.curvature_radius_mm:.2f} mm")
print(f"    q = {pb.q_parameter}")

# 检查 PROPER wfo
wfo = state_s3_exit.proper_wfo
print(f"\n  PROPER wfo 参数:")
print(f"    z = {wfo.z * 1e3:.2f} mm")
print(f"    z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"    z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"    reference_surface = {wfo.reference_surface}")

# 检查 PROPER 相位
proper_phase = proper.prop_get_phase(wfo)
print(f"\n  prop_get_phase 范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# 手动验证写入/读取一致性
print("\n" + "=" * 70)
print("手动验证写入/读取一致性")
print("=" * 70)

state_converter = StateConverter(0.55)
grid_sampling = state_s3_exit.grid_sampling

# 使用 state.phase 和 state.amplitude 重新写入 PROPER
wfo_test = state_converter.amplitude_phase_to_proper(
    state_s3_exit.amplitude,
    state_s3_exit.phase,
    grid_sampling,
    pb,
)

print(f"\n重新写入后的 PROPER 参数:")
print(f"  z = {wfo_test.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo_test.z_w0 * 1e3:.2f} mm")
print(f"  reference_surface = {wfo_test.reference_surface}")

# 读取相位
proper_phase_test = proper.prop_get_phase(wfo_test)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_test):.6f}, {np.max(proper_phase_test):.6f}] rad")

# 比较原始 wfo 和测试 wfo
diff_phase = proper_phase - proper_phase_test
print(f"\n原始 wfo vs 测试 wfo 相位差异:")
print(f"  范围: [{np.min(diff_phase):.9f}, {np.max(diff_phase):.9f}] rad")

# 检查 wfarr 是否相同
diff_wfarr = np.abs(wfo.wfarr - wfo_test.wfarr)
print(f"  wfarr 差异最大值: {np.max(diff_wfarr):.9e}")

print("\n" + "=" * 70)
print("检查 HybridElementPropagator 的输出")
print("=" * 70)

# 找到 Surface 3 入射面状态
state_s3_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_s3_entrance = state
        break

if state_s3_entrance:
    print(f"\n【Surface 3 入射面状态】")
    print(f"  state.phase 范围: [{np.min(state_s3_entrance.phase):.6f}, {np.max(state_s3_entrance.phase):.6f}] rad")
    
    # 检查入射面 PROPER 相位
    wfo_entrance = state_s3_entrance.proper_wfo
    proper_phase_entrance = proper.prop_get_phase(wfo_entrance)
    print(f"  prop_get_phase 范围: [{np.min(proper_phase_entrance):.6f}, {np.max(proper_phase_entrance):.6f}] rad")
    
    # 比较入射面 state.phase 和 PROPER 相位
    diff_entrance = state_s3_entrance.phase - proper_phase_entrance
    print(f"  state.phase vs prop_get_phase 差异:")
    print(f"    范围: [{np.min(diff_entrance):.9f}, {np.max(diff_entrance):.9f}] rad")

print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

print("""
【关键问题】

Surface 3 出射面的 state.phase 范围很小（约 0.04 rad），
但 PROPER wfo 中的相位范围是 [-3.14, 3.14] rad。

这说明 HybridElementPropagator._propagate_local_raytracing 方法
在调用 amplitude_phase_to_proper 时，传入的相位与 state.phase 不一致。

需要检查 HybridElementPropagator._propagate_local_raytracing 的实现。
""")
