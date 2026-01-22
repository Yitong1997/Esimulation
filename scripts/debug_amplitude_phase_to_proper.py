"""
调试 amplitude_phase_to_proper 方法

问题：
- 输入 state.phase 范围: [-0.038, 0.0] rad
- 输出 PROPER 相位范围: [-3.14, 3.14] rad
- 两者完全不一致！
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
print("调试 amplitude_phase_to_proper 方法")
print("=" * 70)

# 创建测试数据
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0

# 创建简单的振幅和相位
amplitude = np.ones((grid_size, grid_size))
# 相位范围很小，类似 Surface 3 出射面
phase = np.zeros((grid_size, grid_size))
phase[256, 256] = -0.038  # 中心有一点相位

# 创建 Pilot Beam 参数（类似 Surface 3 出射面）
pb = PilotBeamParams.from_gaussian_source(wavelength_um, 5.0, -40.0)
print(f"Pilot Beam 参数:")
print(f"  q = {pb.q_parameter}")
print(f"  R = {pb.curvature_radius_mm:.2f} mm")
print(f"  z_w0 = {pb.waist_position_mm:.2f} mm")

# 创建 GridSampling
grid_sampling = GridSampling.create(grid_size, physical_size_mm)

# 调用 amplitude_phase_to_proper
state_converter = StateConverter(wavelength_um)
wfo = state_converter.amplitude_phase_to_proper(
    amplitude, phase, grid_sampling, pilot_beam_params=pb
)

# 检查结果
proper_phase = proper.prop_get_phase(wfo)
print(f"\n输入相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
print(f"输出 PROPER 相位范围: [{np.min(proper_phase):.6f}, {np.max(proper_phase):.6f}] rad")

# 检查 PROPER 参数
print(f"\nPROPER 参数:")
print(f"  z = {wfo.z * 1e3:.2f} mm")
print(f"  z_w0 = {wfo.z_w0 * 1e3:.2f} mm")
print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"  reference_surface = {wfo.reference_surface}")

# 检查 PROPER 参考面相位
proper_ref_phase = state_converter.compute_proper_reference_phase(wfo, grid_sampling)
print(f"\nPROPER 参考面相位范围: [{np.min(proper_ref_phase):.6f}, {np.max(proper_ref_phase):.6f}] rad")

# 重建相位
reconstructed_phase = proper_ref_phase + proper_phase
print(f"重建相位范围: [{np.min(reconstructed_phase):.6f}, {np.max(reconstructed_phase):.6f}] rad")

# 与输入相位比较
diff = reconstructed_phase - phase
print(f"差异范围: [{np.min(diff):.6f}, {np.max(diff):.6f}] rad")

print("\n" + "=" * 70)
print("分析 amplitude_phase_to_proper 的实现")
print("=" * 70)

print("""
amplitude_phase_to_proper 的流程：
1. 使用 prop_begin 初始化 PROPER 对象
2. 同步高斯光束参数（_sync_gaussian_params）
3. 计算 PROPER 参考面相位
4. 计算残差相位 = 输入相位 - PROPER 参考面相位
5. 将残差写入 PROPER（使用 prop_shift_center）

问题可能在于：
- 如果 reference_surface = PLANAR，PROPER 参考面相位 = 0
- 那么残差相位 = 输入相位
- 但 PROPER 相位范围是 [-π, π]，如果输入相位超出这个范围，会被折叠

让我们检查实际的残差相位...
""")

# 手动计算残差相位
residual_phase = phase - proper_ref_phase
print(f"残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")

# 检查 wfarr 中的相位
wfarr_phase = np.angle(proper.prop_shift_center(wfo.wfarr))
print(f"wfarr 相位范围: [{np.min(wfarr_phase):.6f}, {np.max(wfarr_phase):.6f}] rad")
