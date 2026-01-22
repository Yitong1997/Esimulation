"""
分析相位网格与 Pilot Beam 的不匹配

关键发现：
- 光线相位范围: [-0.000281, 0.039535] rad
- Pilot Beam 相位范围: [0.000000, 0.004482] rad
- 相位网格的值比 Pilot Beam 大约 9 倍！

这说明相位网格不是纯粹的 Pilot Beam 相位，而是包含了其他成分。
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

print("=" * 80)
print("相位网格与 Pilot Beam 不匹配分析")
print("=" * 80)

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

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm
grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm
pb = state_entrance.pilot_beam_params
R_pilot = pb.curvature_radius_mm

print(f"网格分辨率: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")
print(f"Pilot Beam 曲率半径: {R_pilot:.2f} mm")

# 创建网格坐标
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_sq_grid = X**2 + Y**2
r_grid = np.sqrt(r_sq_grid)

# 相位网格
phase_grid = state_entrance.phase
amplitude_grid = state_entrance.amplitude
mask = amplitude_grid > 0.01 * np.max(amplitude_grid)

# Pilot Beam 网格相位
pilot_phase_grid = k * r_sq_grid / (2 * R_pilot) if not np.isinf(R_pilot) else np.zeros_like(r_sq_grid)

print("\n" + "=" * 60)
print("【分析 1】相位网格的二次项拟合")
print("=" * 60)

# 拟合相位网格为 a*r^2 + b 的形式
r_flat = r_grid[mask].flatten()
phase_flat = phase_grid[mask].flatten()

A = np.column_stack([r_flat**2, np.ones_like(r_flat)])
coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_flat, rcond=None)
a_fit, b_fit = coeffs

# 从二次项系数反推曲率半径
# phase = k * r² / (2 * R) => a = k / (2 * R) => R = k / (2 * a)
R_from_fit = k / (2 * a_fit) if a_fit != 0 else np.inf

print(f"相位网格拟合: phase = {a_fit:.9f} * r² + {b_fit:.9f}")
print(f"从拟合反推的曲率半径: {R_from_fit:.2f} mm")
print(f"Pilot Beam 曲率半径: {R_pilot:.2f} mm")
print(f"曲率半径比值: {R_from_fit / R_pilot:.6f}")

print("\n" + "=" * 60)
print("【分析 2】检查 PROPER 的参考面类型")
print("=" * 60)

wfo = state_entrance.proper_wfo
print(f"PROPER reference_surface: {wfo.reference_surface}")
print(f"PROPER z: {wfo.z} m = {wfo.z * 1e3} mm")
print(f"PROPER z_w0: {wfo.z_w0} m = {wfo.z_w0 * 1e3} mm")
print(f"PROPER z_Rayleigh: {wfo.z_Rayleigh} m = {wfo.z_Rayleigh * 1e3} mm")

# 计算 PROPER 的参考球面曲率半径
R_proper_ref = (wfo.z - wfo.z_w0) * 1e3  # 转换为 mm
print(f"PROPER 参考球面曲率半径 (z - z_w0): {R_proper_ref:.2f} mm")

print("\n" + "=" * 60)
print("【分析 3】检查相位网格是否包含参考球面相位")
print("=" * 60)

# 如果 PROPER 使用 SPHERI 参考面，相位网格应该是残差相位
# 完整相位 = 残差相位 + 参考球面相位
# 参考球面相位 = k * r² / (2 * R_ref)

if wfo.reference_surface == "SPHERI":
    print("PROPER 使用 SPHERI 参考面")
    print("相位网格存储的是残差相位（相对于参考球面）")
    
    # 计算参考球面相位
    R_ref_mm = R_proper_ref
    ref_phase_grid = k * r_sq_grid / (2 * R_ref_mm) if not np.isinf(R_ref_mm) else np.zeros_like(r_sq_grid)
    
    # 完整相位 = 残差相位 + 参考球面相位
    full_phase_grid = phase_grid + ref_phase_grid
    
    print(f"\n参考球面相位范围: [{np.min(ref_phase_grid[mask]):.6f}, {np.max(ref_phase_grid[mask]):.6f}] rad")
    print(f"残差相位范围: [{np.min(phase_grid[mask]):.6f}, {np.max(phase_grid[mask]):.6f}] rad")
    print(f"完整相位范围: [{np.min(full_phase_grid[mask]):.6f}, {np.max(full_phase_grid[mask]):.6f}] rad")
    
    # 比较完整相位与 Pilot Beam
    diff_full = full_phase_grid - pilot_phase_grid
    rms_full = np.std(diff_full[mask]) / (2 * np.pi)
    print(f"\n完整相位 vs Pilot Beam RMS: {rms_full*1000:.6f} milli-waves")
    
else:
    print("PROPER 使用 PLANAR 参考面")
    print("相位网格存储的是完整相位")

print("\n" + "=" * 60)
print("【分析 4】检查 state_entrance.phase 的来源")
print("=" * 60)

# state_entrance.phase 应该是从 PROPER 提取并处理后的相位
# 根据 amplitude_conversion.md，如果是 SPHERI 参考面，应该已经加回了参考球面相位

import proper
proper_phase = proper.prop_get_phase(wfo)
print(f"PROPER prop_get_phase 范围: [{np.min(proper_phase[mask]):.6f}, {np.max(proper_phase[mask]):.6f}] rad")
print(f"state_entrance.phase 范围: [{np.min(phase_grid[mask]):.6f}, {np.max(phase_grid[mask]):.6f}] rad")

# 检查两者是否相同
diff_proper_state = proper_phase - phase_grid
print(f"PROPER phase vs state.phase 差异: {np.max(np.abs(diff_proper_state[mask])):.9f} rad")

print("\n" + "=" * 60)
print("【分析 5】检查 Pilot Beam 参数的来源")
print("=" * 60)

print(f"Pilot Beam 参数:")
print(f"  wavelength_um: {pb.wavelength_um}")
print(f"  waist_radius_mm: {pb.waist_radius_mm}")
print(f"  waist_position_mm: {pb.waist_position_mm}")
print(f"  curvature_radius_mm: {pb.curvature_radius_mm}")
print(f"  spot_size_mm: {pb.spot_size_mm}")
print(f"  q_parameter: {pb.q_parameter}")
print(f"  rayleigh_length_mm: {pb.rayleigh_length_mm}")

# 验证曲率半径计算
z = -pb.waist_position_mm  # 当前位置相对于束腰的距离
z_R = pb.rayleigh_length_mm
if abs(z) < 1e-15:
    R_calc = np.inf
else:
    R_calc = z * (1 + (z_R / z)**2)
print(f"\n验证曲率半径计算:")
print(f"  z (相对于束腰): {z:.2f} mm")
print(f"  z_R: {z_R:.2f} mm")
print(f"  计算的 R = z * (1 + (z_R/z)²): {R_calc:.2f} mm")
print(f"  存储的 R: {pb.curvature_radius_mm:.2f} mm")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
