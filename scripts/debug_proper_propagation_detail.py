"""
详细调试 PROPER 传播

检查 PROPER 传播引入的相位变化
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

from hybrid_optical_propagation.data_models import GridSampling, PilotBeamParams
from hybrid_optical_propagation.state_converter import StateConverter

print("=" * 70)
print("详细调试 PROPER 传播")
print("=" * 70)

# 参数设置（与实际传播一致）
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0
sampling_mm = physical_size_mm / grid_size
w0_mm = 5.0

grid_sampling = GridSampling(
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
    sampling_mm=sampling_mm,
    beam_ratio=0.25,
)

# 创建初始高斯光束（模拟 Surface 3 出射面）
# 使用 prop_begin 初始化
beam_diameter_m = physical_size_mm * 1e-3
wavelength_m = wavelength_um * 1e-6
w0_m = w0_mm * 1e-3
beam_diam_fraction = (2 * w0_m) / beam_diameter_m

wfo = proper.prop_begin(
    beam_diameter_m,
    wavelength_m,
    grid_size,
    beam_diam_fraction,
)

# 设置高斯光束参数（模拟 Surface 3 出射面的状态）
# 当前在 z=0（PROPER 内部坐标），束腰在 z=-40mm
wfo.w0 = w0_m
wfo.z_Rayleigh = np.pi * w0_m**2 / wavelength_m
wfo.z_w0 = -40e-3  # 束腰在 -40mm 处

print(f"\n初始 PROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.2f} mm")
print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

# 添加小的像差相位（模拟 Surface 3 出射面的相位）
X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
r_sq = X_mm**2 + Y_mm**2
aberration_phase = -0.04 * r_sq / (physical_size_mm/2)**2

# 高斯振幅
gaussian_amplitude = np.exp(-r_sq / (2 * (w0_mm * 2)**2))

# 将像差相位写入 wfarr
# 注意：wfarr 存储的是相对于参考面的残差
residual_field = gaussian_amplitude * np.exp(1j * aberration_phase)
wfo.wfarr = proper.prop_shift_center(residual_field)

print(f"\n写入像差相位后:")
proper_phase_before = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_before):.6f}, {np.max(proper_phase_before):.6f}] rad")

# 保存传播前的状态
wfarr_before = wfo.wfarr.copy()
z_before = wfo.z

print("\n" + "=" * 70)
print("执行 PROPER 传播 (40 mm)")
print("=" * 70)

# 传播 40 mm
distance_m = 40e-3
proper.prop_propagate(wfo, distance_m)

print(f"\n传播后 PROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z - z_w0 = {(wfo.z - wfo.z_w0) * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

proper_phase_after = proper.prop_get_phase(wfo)
print(f"  prop_get_phase 范围: [{np.min(proper_phase_after):.6f}, {np.max(proper_phase_after):.6f}] rad")

# 分析相位变化
print("\n" + "=" * 70)
print("分析相位变化")
print("=" * 70)

# 检查 wfarr 的变化
wfarr_after = wfo.wfarr
wfarr_phase_before = np.angle(proper.prop_shift_center(wfarr_before))
wfarr_phase_after = np.angle(proper.prop_shift_center(wfarr_after))

print(f"\nwfarr 相位变化:")
print(f"  传播前: [{np.min(wfarr_phase_before):.6f}, {np.max(wfarr_phase_before):.6f}] rad")
print(f"  传播后: [{np.min(wfarr_phase_after):.6f}, {np.max(wfarr_phase_after):.6f}] rad")

# 检查振幅变化
wfarr_amp_before = np.abs(proper.prop_shift_center(wfarr_before))
wfarr_amp_after = np.abs(proper.prop_shift_center(wfarr_after))

print(f"\nwfarr 振幅变化:")
print(f"  传播前: [{np.min(wfarr_amp_before):.6f}, {np.max(wfarr_amp_before):.6f}]")
print(f"  传播后: [{np.min(wfarr_amp_after):.6f}, {np.max(wfarr_amp_after):.6f}]")
