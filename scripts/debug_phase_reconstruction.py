"""
调试相位重建问题

问题：
1. reconstructor.reconstruct() 返回复振幅
2. np.angle() 提取相位时会折叠到 [-π, π]
3. unwrap_with_pilot_beam 无法正确解包裹（因为 Pilot Beam 相位范围很小）

验证：
1. 检查 reconstructor 内部的相位范围
2. 检查 np.angle() 后的相位范围
3. 检查 unwrap_with_pilot_beam 后的相位范围
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.data_models import PilotBeamParams, GridSampling

print("=" * 70)
print("调试相位重建问题")
print("=" * 70)

# 创建测试数据
wavelength_um = 0.55
grid_size = 512
physical_size_mm = 40.0
sampling_mm = physical_size_mm / grid_size

# 创建 reconstructor
reconstructor = RayToWavefrontReconstructor(
    grid_size=grid_size,
    sampling_mm=sampling_mm,
    wavelength_um=wavelength_um,
)

# 创建测试光线数据
# 模拟一个简单的情况：光线从入射面到出射面，OPD 很小
num_rays = 100
np.random.seed(42)

# 入射面光线位置（均匀分布）
ray_x_in = np.linspace(-10, 10, int(np.sqrt(num_rays)))
ray_y_in = np.linspace(-10, 10, int(np.sqrt(num_rays)))
ray_x_in, ray_y_in = np.meshgrid(ray_x_in, ray_y_in)
ray_x_in = ray_x_in.flatten()
ray_y_in = ray_y_in.flatten()

# 出射面光线位置（与入射面相同，模拟平面镜）
ray_x_out = ray_x_in.copy()
ray_y_out = ray_y_in.copy()

# OPD（波长数）- 模拟一个小的 OPD
# 使用一个二次函数模拟球面波前
r_sq = ray_x_in**2 + ray_y_in**2
opd_waves = r_sq / 1000.0  # 很小的 OPD

# 有效光线掩模
valid_mask = np.ones(len(ray_x_in), dtype=bool)

print(f"\n【输入数据】")
print(f"  光线数量: {len(ray_x_in)}")
print(f"  OPD 范围: [{np.min(opd_waves):.6f}, {np.max(opd_waves):.6f}] waves")
print(f"  OPD 对应相位范围: [{np.min(opd_waves)*2*np.pi:.6f}, {np.max(opd_waves)*2*np.pi:.6f}] rad")

# 调用 reconstruct
complex_amplitude = reconstructor.reconstruct(
    ray_x_in, ray_y_in,
    ray_x_out, ray_y_out,
    opd_waves, valid_mask,
)

# 提取相位
exit_phase_angle = np.angle(complex_amplitude)

print(f"\n【reconstruct 输出】")
print(f"  复振幅形状: {complex_amplitude.shape}")
print(f"  np.angle() 相位范围: [{np.min(exit_phase_angle):.6f}, {np.max(exit_phase_angle):.6f}] rad")

# 创建 Pilot Beam 参数（模拟 Surface 3 出射面）
pb = PilotBeamParams.from_gaussian_source(wavelength_um, 5.0, -40.0)
grid_sampling = GridSampling.create(grid_size, physical_size_mm)

print(f"\n【Pilot Beam 参数】")
print(f"  q = {pb.q_parameter}")
print(f"  R = {pb.curvature_radius_mm:.2f} mm")
pilot_phase = pb.compute_phase_grid(grid_size, physical_size_mm)
print(f"  Pilot Beam 相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")

# 使用 Pilot Beam 解包裹
state_converter = StateConverter(wavelength_um)
unwrapped_phase = state_converter.unwrap_with_pilot_beam(
    exit_phase_angle, pb, grid_sampling
)

print(f"\n【解包裹后】")
print(f"  解包裹相位范围: [{np.min(unwrapped_phase):.6f}, {np.max(unwrapped_phase):.6f}] rad")

# 检查解包裹是否正确
# 期望的相位应该是 -2π × OPD（根据 reconstructor 的定义）
# 但由于 OPD 很小，相位应该也很小

print("\n" + "=" * 70)
print("问题分析")
print("=" * 70)

print("""
【问题根源】

1. reconstructor._compute_amplitude_phase_jacobian 计算相位：
   phase = -2π × OPD
   这个相位是非折叠的。

2. reconstructor._resample_to_grid_separate 插值相位：
   phase_grid = griddata(..., valid_phase, ...)
   插值后的相位仍然是非折叠的。

3. reconstructor.reconstruct 返回复振幅：
   complex_amplitude = amp_grid * np.exp(1j * phase_grid)
   这里 exp(1j * phase_grid) 会将相位"隐藏"在复数中。

4. HybridElementPropagator._propagate_local_raytracing 提取相位：
   exit_phase = np.angle(exit_complex)
   np.angle() 返回 [-π, π] 范围的相位！

5. unwrap_with_pilot_beam 尝试解包裹：
   但 Pilot Beam 相位范围很小（约 0.009 rad），
   而 np.angle() 返回的相位可能已经折叠，
   导致解包裹失败。

【解决方案】

修改 reconstructor.reconstruct() 方法，直接返回振幅和相位，
而不是返回复振幅。这样可以避免 np.angle() 导致的相位折叠。

或者，修改 HybridElementPropagator._propagate_local_raytracing，
直接使用 reconstructor 内部的相位网格，而不是从复振幅中提取。
""")

# 验证：如果直接使用 reconstructor 内部的相位，是否正确？
print("\n" + "=" * 70)
print("验证：直接使用 reconstructor 内部的相位")
print("=" * 70)

# 手动调用 reconstructor 的内部方法
amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
    ray_x_in, ray_y_in,
    ray_x_out, ray_y_out,
    opd_waves, valid_mask,
)

amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
    ray_x_out, ray_y_out,
    amplitude, phase,
    valid_mask,
)

print(f"  内部相位范围: [{np.min(phase_grid):.6f}, {np.max(phase_grid):.6f}] rad")
print(f"  np.angle() 相位范围: [{np.min(exit_phase_angle):.6f}, {np.max(exit_phase_angle):.6f}] rad")

# 比较
diff = phase_grid - exit_phase_angle
print(f"  差异范围: [{np.min(diff):.6f}, {np.max(diff):.6f}] rad")

# 检查是否有 2π 跳变
has_2pi_jump = np.any(np.abs(diff) > np.pi)
print(f"  是否有 2π 跳变: {has_2pi_jump}")
