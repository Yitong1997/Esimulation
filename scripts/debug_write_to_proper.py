"""
调试 amplitude_phase_to_proper 写入 PROPER 的问题

问题：
- 输入 state.phase 范围: [-0.038, 0.0] rad
- 输出 PROPER 相位范围: [-3.14, 3.14] rad
- 两者完全不一致！

需要追踪：
1. amplitude_phase_to_proper 的每一步
2. 检查 residual_field 的相位
3. 检查 prop_shift_center 的影响
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

from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.data_models import PilotBeamParams, GridSampling

print("=" * 70)
print("调试 amplitude_phase_to_proper 写入 PROPER 的问题")
print("=" * 70)

# 创建测试数据（模拟 Surface 3 出射面）
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0

# 创建简单的振幅和相位
amplitude = np.ones((grid_size, grid_size))
# 相位范围很小，类似 Surface 3 出射面
# 使用一个简单的二次相位
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_sq = X**2 + Y**2
# 相位范围约 [-0.04, 0] rad
phase = -r_sq / 10000.0

print(f"\n【输入数据】")
print(f"  振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
print(f"  相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")

# 创建 Pilot Beam 参数
pb = PilotBeamParams.from_gaussian_source(wavelength_um, 5.0, -40.0)
print(f"\n【Pilot Beam 参数】")
print(f"  q = {pb.q_parameter}")
print(f"  R = {pb.curvature_radius_mm:.2f} mm")
print(f"  z_Rayleigh = {pb.rayleigh_length_mm:.2f} mm")

# 创建 GridSampling
grid_sampling = GridSampling.create(grid_size, physical_size_mm)

# 手动执行 amplitude_phase_to_proper 的每一步
print("\n" + "=" * 70)
print("手动执行 amplitude_phase_to_proper 的每一步")
print("=" * 70)

# Step 1: 初始化 PROPER 对象
beam_diameter_m = grid_sampling.physical_size_mm * 1e-3
wavelength_m = wavelength_um * 1e-6
w0_m = pb.waist_radius_mm * 1e-3
beam_diam_fraction = (2 * w0_m) / beam_diameter_m
beam_diam_fraction = max(0.1, min(0.9, beam_diam_fraction))

print(f"\n【Step 1: prop_begin】")
print(f"  beam_diameter_m = {beam_diameter_m}")
print(f"  wavelength_m = {wavelength_m}")
print(f"  beam_diam_fraction = {beam_diam_fraction}")

wfo = proper.prop_begin(
    beam_diameter_m,
    wavelength_m,
    grid_size,
    beam_diam_fraction,
)

print(f"  wfo.z = {wfo.z * 1e3:.2f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")


# Step 2: 同步高斯光束参数
print(f"\n【Step 2: 同步高斯光束参数】")
wfo.w0 = pb.waist_radius_mm * 1e-3
wfo.z_Rayleigh = pb.rayleigh_length_mm * 1e-3
wfo.z_w0 = wfo.z - pb.waist_position_mm * 1e-3

rayleigh_factor = proper.rayleigh_factor
print(f"  rayleigh_factor = {rayleigh_factor}")
print(f"  |wfo.z_w0 - wfo.z| = {abs(wfo.z_w0 - wfo.z) * 1e3:.2f} mm")
print(f"  rayleigh_factor * wfo.z_Rayleigh = {rayleigh_factor * wfo.z_Rayleigh * 1e3:.2f} mm")

if abs(wfo.z_w0 - wfo.z) < rayleigh_factor * wfo.z_Rayleigh:
    wfo.beam_type_old = "INSIDE_"
    wfo.reference_surface = "PLANAR"
else:
    wfo.beam_type_old = "OUTSIDE"
    wfo.reference_surface = "SPHERI"

print(f"  wfo.z = {wfo.z * 1e3:.2f} mm")
print(f"  wfo.z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  wfo.reference_surface = {wfo.reference_surface}")

# Step 3: 计算 PROPER 参考面相位
print(f"\n【Step 3: 计算 PROPER 参考面相位】")
state_converter = StateConverter(wavelength_um)
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"  PROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# Step 4: 计算残差相位
print(f"\n【Step 4: 计算残差相位】")
residual_phase = phase - proper_ref_phase
print(f"  残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")

# Step 5: 构建残差场
print(f"\n【Step 5: 构建残差场】")
residual_field = amplitude * np.exp(1j * residual_phase)
residual_field_phase = np.angle(residual_field)
print(f"  残差场相位范围 (np.angle): [{np.min(residual_field_phase):.6f}, {np.max(residual_field_phase):.6f}] rad")

# Step 6: prop_shift_center
print(f"\n【Step 6: prop_shift_center】")
shifted_field = proper.prop_shift_center(residual_field)
shifted_field_phase = np.angle(shifted_field)
print(f"  移位后场相位范围 (np.angle): [{np.min(shifted_field_phase):.6f}, {np.max(shifted_field_phase):.6f}] rad")

# Step 7: 写入 wfarr
print(f"\n【Step 7: 写入 wfarr】")
wfo.wfarr = shifted_field

# Step 8: 读取 PROPER 相位
print(f"\n【Step 8: 读取 PROPER 相位】")
proper_phase = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# 检查 prop_get_phase 的实现
print(f"\n【检查 prop_get_phase 的实现】")
# prop_get_phase 内部会调用 prop_shift_center 移回中心
# 然后调用 np.angle 提取相位
wfarr_shifted_back = proper.prop_shift_center(wfo.wfarr)
wfarr_phase = np.angle(wfarr_shifted_back)
print(f"  手动移回中心后相位范围: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")

# 比较
print(f"\n【比较】")
print(f"  输入相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
print(f"  残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")
print(f"  PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# 重建相位
reconstructed_phase = proper_ref_phase + proper_phase
print(f"  重建相位范围: [{np.min(reconstructed_phase):.6f}, {np.max(reconstructed_phase):.6f}] rad")

# 差异
diff = reconstructed_phase - phase
print(f"  差异范围: [{np.min(diff):.6f}, {np.max(diff):.6f}] rad")
