"""
调试 Surface 3 的混合元件传播流程

关键检查点：
1. 入射面的 state.phase 范围
2. WavefrontToRaysSampler 输出的 OPD
3. ElementRaytracer 输出的 OPD
4. RayToWavefrontReconstructor 重建的相位
5. amplitude_phase_to_proper 的残差相位
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

print("=" * 70)
print("调试 Surface 3 的混合元件传播流程")
print("=" * 70)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

print(f"\n光学系统表面:")
for surface in optical_system:
    mirror_str = " [MIRROR]" if surface.is_mirror else ""
    print(f"  Surface {surface.index}: {surface.surface_type}{mirror_str}, "
          f"comment='{surface.comment}'")

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

# 获取 Surface 3 入射面的状态
print("\n" + "=" * 70)
print("【步骤 1】获取 Surface 3 入射面状态")
print("=" * 70)

# 手动执行传播到 Surface 3 入射面
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]

# 传播到 Surface 0, 1, 2
for i in range(3):
    surface = optical_system[i]
    propagator._propagate_to_surface(i)

# 现在 _current_state 应该是 Surface 2 exit
print(f"\n当前状态: Surface {propagator._current_state.surface_index} {propagator._current_state.position}")

# 传播到 Surface 3 入射面
surface_3_index = 3  # Surface 3 是第 4 个表面（索引 3）
# 找到 Surface 3 的实际索引
for i, s in enumerate(optical_system):
    if s.index == 4:  # Surface 4 是 M1 镜
        surface_3_index = i
        break

print(f"\nSurface 3 (M1) 在 optical_system 中的索引: {surface_3_index}")
surface_3 = optical_system[surface_3_index]
print(f"Surface 3 信息: index={surface_3.index}, is_mirror={surface_3.is_mirror}")

# 获取 Surface 3 入射面的光轴状态
entrance_axis = propagator.get_optical_axis_at_surface(surface_3_index, 'entrance')
exit_axis = propagator.get_optical_axis_at_surface(surface_3_index, 'exit')

print(f"\n入射光轴方向: {entrance_axis.direction.to_array()}")
print(f"出射光轴方向: {exit_axis.direction.to_array()}")

# 传播到 Surface 3 入射面
propagator._propagate_free_space_to_entrance(surface_3_index, entrance_axis)

state_s3_entrance = propagator._current_state
print(f"\nSurface 3 入射面状态:")
print(f"  振幅范围: [{np.min(state_s3_entrance.amplitude):.6f}, {np.max(state_s3_entrance.amplitude):.6f}]")
print(f"  相位范围: [{np.min(state_s3_entrance.phase):.6f}, {np.max(state_s3_entrance.phase):.6f}] rad")
print(f"  相位范围（波长数）: [{np.min(state_s3_entrance.phase)/(2*np.pi):.6f}, {np.max(state_s3_entrance.phase)/(2*np.pi):.6f}] waves")

# 检查 Pilot Beam 参数
pilot_params = state_s3_entrance.pilot_beam_params
print(f"\nPilot Beam 参数:")
print(f"  束腰半径: {pilot_params.waist_radius_mm:.4f} mm")
print(f"  束腰位置: {pilot_params.waist_position_mm:.4f} mm")
print(f"  曲率半径: {pilot_params.curvature_radius_mm:.4f} mm")
print(f"  瑞利长度: {pilot_params.rayleigh_length_mm:.4f} mm")

# 计算 Pilot Beam 参考相位
grid_sampling = state_s3_entrance.grid_sampling
pilot_phase = pilot_params.compute_phase_grid(
    grid_sampling.grid_size, grid_sampling.physical_size_mm
)
print(f"\nPilot Beam 参考相位范围: [{np.min(pilot_phase):.6f}, {np.max(pilot_phase):.6f}] rad")

# 计算残差相位
residual_phase = state_s3_entrance.phase - pilot_phase
print(f"残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")

print("\n" + "=" * 70)
print("【步骤 2】执行混合元件传播")
print("=" * 70)

# 使用 HybridElementPropagator 处理 Surface 3
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor

# 2.1 从振幅/相位采样光线
print("\n【2.1】从振幅/相位采样光线")

sampler = WavefrontToRaysSampler(
    amplitude=state_s3_entrance.amplitude,
    phase=state_s3_entrance.phase,
    physical_size=grid_sampling.physical_size_mm,
    wavelength=0.55,
    num_rays=150,
    distribution="hexapolar",
)

input_rays = sampler.get_output_rays()
opd_waves_from_sampler = sampler.get_ray_opd()

print(f"  采样光线数: {len(np.asarray(input_rays.x))}")
print(f"  光线 x 范围: [{np.min(input_rays.x):.4f}, {np.max(input_rays.x):.4f}] mm")
print(f"  光线 y 范围: [{np.min(input_rays.y):.4f}, {np.max(input_rays.y):.4f}] mm")
print(f"  OPD 范围（波长数）: [{np.min(opd_waves_from_sampler):.6f}, {np.max(opd_waves_from_sampler):.6f}]")

# 检查 OPD 是否与输入相位一致
wavelength_mm = 0.55 * 1e-3
expected_opd_waves = state_s3_entrance.phase / (2 * np.pi)
print(f"  输入相位对应的 OPD 范围（波长数）: [{np.min(expected_opd_waves):.6f}, {np.max(expected_opd_waves):.6f}]")

# 2.2 创建表面定义并进行光线追迹
print("\n【2.2】光线追迹")

# 创建表面定义
hybrid_propagator = HybridElementPropagator(
    wavelength_um=0.55,
    num_rays=150,
)

surface_def = hybrid_propagator._create_surface_definition(surface_3, entrance_axis)
print(f"  表面类型: {surface_def.surface_type}")
print(f"  曲率半径: {surface_def.radius}")
print(f"  倾斜角 X: {np.degrees(surface_def.tilt_x):.2f}°")
print(f"  倾斜角 Y: {np.degrees(surface_def.tilt_y):.2f}°")

raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=0.55,
    chief_ray_direction=tuple(entrance_axis.direction.to_array()),
    entrance_position=tuple(entrance_axis.position.to_array()),
    exit_chief_direction=tuple(exit_axis.direction.to_array()),
)

# 设置输入光线的 OPD
input_rays.opd = opd_waves_from_sampler * wavelength_mm

output_rays = raytracer.trace(input_rays)

print(f"  输出光线 x 范围: [{np.min(output_rays.x):.4f}, {np.max(output_rays.x):.4f}] mm")
print(f"  输出光线 y 范围: [{np.min(output_rays.y):.4f}, {np.max(output_rays.y):.4f}] mm")
print(f"  输出 OPD 范围（mm）: [{np.min(output_rays.opd):.6f}, {np.max(output_rays.opd):.6f}]")

# 转换为波长数
output_opd_waves = np.asarray(output_rays.opd) / wavelength_mm
print(f"  输出 OPD 范围（波长数）: [{np.min(output_opd_waves):.6f}, {np.max(output_opd_waves):.6f}]")

# 2.3 重建波前
print("\n【2.3】重建波前")

reconstructor = RayToWavefrontReconstructor(
    grid_size=grid_sampling.grid_size,
    sampling_mm=grid_sampling.sampling_mm,
    wavelength_um=0.55,
)

valid_mask = np.ones(len(np.asarray(input_rays.x)), dtype=bool)

exit_amplitude, exit_phase = reconstructor.reconstruct_amplitude_phase(
    ray_x_in=np.asarray(input_rays.x),
    ray_y_in=np.asarray(input_rays.y),
    ray_x_out=np.asarray(output_rays.x),
    ray_y_out=np.asarray(output_rays.y),
    opd_waves=output_opd_waves,
    valid_mask=valid_mask,
)

print(f"  重建振幅范围: [{np.min(exit_amplitude):.6f}, {np.max(exit_amplitude):.6f}]")
print(f"  重建相位范围: [{np.min(exit_phase):.6f}, {np.max(exit_phase):.6f}] rad")
print(f"  重建相位范围（波长数）: [{np.min(exit_phase)/(2*np.pi):.6f}, {np.max(exit_phase)/(2*np.pi):.6f}]")

# 2.4 更新 Pilot Beam 参数
print("\n【2.4】更新 Pilot Beam 参数")

new_pilot_params = hybrid_propagator._update_pilot_beam(pilot_params, surface_3)
print(f"  新束腰半径: {new_pilot_params.waist_radius_mm:.4f} mm")
print(f"  新束腰位置: {new_pilot_params.waist_position_mm:.4f} mm")
print(f"  新曲率半径: {new_pilot_params.curvature_radius_mm:.4f} mm")

# 计算新的 Pilot Beam 参考相位
new_pilot_phase = new_pilot_params.compute_phase_grid(
    grid_sampling.grid_size, grid_sampling.physical_size_mm
)
print(f"  新 Pilot Beam 参考相位范围: [{np.min(new_pilot_phase):.6f}, {np.max(new_pilot_phase):.6f}] rad")

# 2.5 计算残差相位
print("\n【2.5】计算残差相位（写入 PROPER 前）")

residual_phase_exit = exit_phase - new_pilot_phase
print(f"  残差相位范围: [{np.min(residual_phase_exit):.6f}, {np.max(residual_phase_exit):.6f}] rad")
print(f"  残差相位范围（波长数）: [{np.min(residual_phase_exit)/(2*np.pi):.6f}, {np.max(residual_phase_exit)/(2*np.pi):.6f}]")

# 检查有效区域的残差
valid_mask_grid = exit_amplitude > 0.01 * np.max(exit_amplitude)
if np.any(valid_mask_grid):
    residual_valid = residual_phase_exit[valid_mask_grid]
    print(f"  有效区域残差相位范围: [{np.min(residual_valid):.6f}, {np.max(residual_valid):.6f}] rad")
    print(f"  有效区域残差相位 RMS: {np.std(residual_valid):.6f} rad")

print("\n" + "=" * 70)
print("【分析】")
print("=" * 70)

print("""
关键发现：
1. 输入相位范围很小（接近 0）
2. 采样器输出的 OPD 也很小
3. 光线追迹后的 OPD 应该也很小（平面镜不引入额外 OPD）
4. 重建的相位应该与输入相位接近

如果残差相位很大，问题可能在于：
- Pilot Beam 参数更新不正确
- 重建相位的符号或单位问题
- OPD 计算的符号问题
""")
