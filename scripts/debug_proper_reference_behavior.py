"""
分析 PROPER 的参考面行为

关键发现：
- PROPER 传播后相位是 0
- 这说明 PROPER 使用参考球面跟踪相位

需要理解：
1. PROPER 的 reference_surface 设置
2. 相位是如何存储的
3. 为什么测试案例中的相位不是 0
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

print("=" * 80)
print("PROPER 参考面行为分析")
print("=" * 80)

# 创建高斯光束
wavelength_um = 0.55
wavelength_m = wavelength_um * 1e-6
w0_mm = 5.0
w0_m = w0_mm * 1e-3
grid_size = 512
physical_size_mm = 40.0
physical_size_m = physical_size_mm * 1e-3

beam_diameter_m = 2 * w0_m
beam_diam_fraction = beam_diameter_m / physical_size_m

wfo = proper.prop_begin(
    beam_diameter_m,
    wavelength_m,
    grid_size,
    beam_diam_fraction,
)

print(f"初始状态:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z} m")
print(f"  z_w0: {wfo.z_w0} m")
print(f"  z_Rayleigh: {wfo.z_Rayleigh} m")

# 传播
proper.prop_propagate(wfo, 0.04)  # 40 mm

print(f"\n传播后:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z} m")
print(f"  z_w0: {wfo.z_w0} m")

# 获取相位
phase = proper.prop_get_phase(wfo)
print(f"  相位范围: [{np.min(phase):.9f}, {np.max(phase):.9f}] rad")

print("\n" + "=" * 60)
print("【分析】PROPER 的相位存储方式")
print("=" * 60)

# PROPER 使用 PLANAR 参考面时，相位是相对于平面波的
# PROPER 使用 SPHERI 参考面时，相位是相对于球面波的

# 检查 rayleigh_factor
print(f"proper.rayleigh_factor: {proper.rayleigh_factor}")

# 判断参考面类型
z = wfo.z
z_w0 = wfo.z_w0
z_R = wfo.z_Rayleigh
rayleigh_factor = proper.rayleigh_factor

if abs(z - z_w0) < rayleigh_factor * z_R:
    expected_ref = "PLANAR"
else:
    expected_ref = "SPHERI"

print(f"|z - z_w0| = {abs(z - z_w0):.6f} m")
print(f"rayleigh_factor * z_R = {rayleigh_factor * z_R:.6f} m")
print(f"期望的参考面类型: {expected_ref}")
print(f"实际的参考面类型: {wfo.reference_surface}")

print("\n" + "=" * 60)
print("【分析】为什么测试案例中的相位不是 0")
print("=" * 60)

# 重新加载测试案例
from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)

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

# 传播到 Surface 3
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):
    propagator._propagate_to_surface(i)

# 获取入射面状态
state_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_entrance = state
        break

wfo_test = state_entrance.proper_wfo
phase_test = proper.prop_get_phase(wfo_test)

print(f"\n测试案例的 PROPER 状态:")
print(f"  reference_surface: {wfo_test.reference_surface}")
print(f"  z: {wfo_test.z} m")
print(f"  z_w0: {wfo_test.z_w0} m")
print(f"  z_Rayleigh: {wfo_test.z_Rayleigh} m")
print(f"  相位范围: [{np.min(phase_test):.6f}, {np.max(phase_test):.6f}] rad")

# 检查 state.phase 与 PROPER 相位的关系
print(f"\nstate.phase 范围: [{np.min(state_entrance.phase):.6f}, {np.max(state_entrance.phase):.6f}] rad")
print(f"state.phase 与 PROPER 相位差异: {np.max(np.abs(state_entrance.phase - phase_test)):.9f} rad")

print("\n" + "=" * 60)
print("【分析】相位差异的来源")
print("=" * 60)

# 检查初始状态
state_initial = propagator._surface_states[0]
wfo_initial = state_initial.proper_wfo
phase_initial = proper.prop_get_phase(wfo_initial)

print(f"初始状态:")
print(f"  reference_surface: {wfo_initial.reference_surface}")
print(f"  z: {wfo_initial.z} m")
print(f"  相位范围: [{np.min(phase_initial):.6f}, {np.max(phase_initial):.6f}] rad")
print(f"  state.phase 范围: [{np.min(state_initial.phase):.6f}, {np.max(state_initial.phase):.6f}] rad")

# 检查每个表面的状态
print("\n各表面状态:")
for state in propagator._surface_states:
    wfo_s = state.proper_wfo
    phase_s = proper.prop_get_phase(wfo_s)
    print(f"  Surface {state.surface_index} ({state.position}):")
    print(f"    reference_surface: {wfo_s.reference_surface}")
    print(f"    PROPER 相位范围: [{np.min(phase_s):.6f}, {np.max(phase_s):.6f}] rad")
    print(f"    state.phase 范围: [{np.min(state.phase):.6f}, {np.max(state.phase):.6f}] rad")
