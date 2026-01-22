"""
调试 Surface 3 出射面的 state.phase 来源

关键发现：
- PROPER 提取的相位范围: [-3.14, 3.14] rad（有折叠）
- state.phase 范围: [-0.038, 0.0] rad（很小）
- 两者完全不同！

需要找出 state.phase 是从哪里来的
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
print("调试 Surface 3 出射面的 state.phase 来源")
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

# 只传播到 Surface 3
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):  # 0, 1, 2, 3
    propagator._propagate_to_surface(i)

# 找到 Surface 3 入射面和出射面
state_s3_entrance = None
state_s3_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3:
        if state.position == 'entrance':
            state_s3_entrance = state
        elif state.position == 'exit':
            state_s3_exit = state

print("\n【Surface 3 入射面】")
if state_s3_entrance:
    print(f"  state.phase 范围: [{np.min(state_s3_entrance.phase):.6f}, {np.max(state_s3_entrance.phase):.6f}] rad")
    print(f"  state.amplitude 范围: [{np.min(state_s3_entrance.amplitude):.6f}, {np.max(state_s3_entrance.amplitude):.6f}]")
    
    # 从 PROPER 提取
    wfo = state_s3_entrance.proper_wfo
    proper_phase = proper.prop_get_phase(wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

print("\n【Surface 3 出射面】")
if state_s3_exit:
    print(f"  state.phase 范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}] rad")
    print(f"  state.amplitude 范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")
    
    # 从 PROPER 提取
    wfo = state_s3_exit.proper_wfo
    proper_phase = proper.prop_get_phase(wfo)
    print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

print("\n" + "=" * 70)
print("分析 HybridElementPropagator 的处理流程")
print("=" * 70)

# 查看 reconstructor.reconstruct 返回的复振幅
# 在 _propagate_local_raytracing 中：
# exit_complex = reconstructor.reconstruct(...)
# exit_amplitude = np.abs(exit_complex)
# exit_phase = np.angle(exit_complex)  <-- 这里会折叠！
# exit_phase = self._state_converter.unwrap_with_pilot_beam(...)

print("""
在 HybridElementPropagator._propagate_local_raytracing 中：

1. reconstructor.reconstruct() 返回复振幅
2. exit_phase = np.angle(exit_complex)  <-- 这里会折叠到 [-π, π]！
3. exit_phase = unwrap_with_pilot_beam(exit_phase, ...)

问题：如果 reconstructor 返回的相位本身就是折叠的，
那么 unwrap_with_pilot_beam 能否正确解包裹？
""")

# 检查 Pilot Beam 参数
pb = state_s3_exit.pilot_beam_params
print(f"\nSurface 3 出射面 Pilot Beam 参数:")
print(f"  曲率半径 R: {pb.curvature_radius_mm:.2f} mm")
print(f"  束腰半径 w0: {pb.waist_radius_mm:.2f} mm")
print(f"  束腰位置 z0: {pb.waist_position_mm:.2f} mm")

# 计算 Pilot Beam 相位
grid_sampling = state_s3_exit.grid_sampling
pilot_phase = pb.compute_phase_grid(grid_sampling.grid_size, grid_sampling.physical_size_mm)
print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")

# 如果 Pilot Beam 相位范围很小（接近 0），而实际相位范围很大（接近 ±π），
# 那么 unwrap_with_pilot_beam 会失败！

print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

print("""
【问题根源】

1. reconstructor.reconstruct() 返回的相位是 -2π × OPD
   - OPD 是相对于主光线的光程差
   - 对于平面镜，OPD 应该很小（接近 0）
   
2. 但是 PROPER 中存储的相位范围是 [-π, π]
   - 这说明 PROPER 中的相位是折叠的
   
3. 问题：state.phase 应该来自哪里？
   - 如果来自 reconstructor：应该是小值（OPD 很小）
   - 如果来自 PROPER：应该是折叠值（接近 ±π）
   
4. 实际情况：
   - state.phase 范围: [-0.038, 0.0] rad（小值）
   - PROPER 相位范围: [-3.14, 3.14] rad（折叠值）
   
   这说明 state.phase 来自 reconstructor，而不是 PROPER！
   
5. 但是 PROPER wfo 也被更新了（通过 amplitude_phase_to_proper）
   - 这会导致 PROPER 中的相位与 state.phase 不一致！
""")

# 验证：检查 state.phase 是否与 reconstructor 的输出一致
print("\n【验证】检查 state.phase 的来源")

# state.phase 范围很小，说明它来自 reconstructor
# reconstructor 的相位 = -2π × OPD
# 对于平面镜，OPD 应该很小

# 计算期望的 OPD 范围
state_phase = state_s3_exit.phase
expected_opd_waves = -state_phase / (2 * np.pi)
print(f"  从 state.phase 反推的 OPD 范围: [{np.min(expected_opd_waves):.6f}, {np.max(expected_opd_waves):.6f}] waves")

# 这个 OPD 范围是否合理？
# 对于平面镜，OPD 应该接近 0
print(f"  OPD 范围是否合理（平面镜应接近 0）: {'是' if np.max(np.abs(expected_opd_waves)) < 0.1 else '否'}")
