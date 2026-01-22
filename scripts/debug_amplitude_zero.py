"""
调试振幅变为零的问题

详细追踪 Surface 7 处振幅变为零的原因。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 导入模块
# ============================================================================

print_section("导入模块")

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    PilotBeamParams,
    GridSampling,
    PropagationState,
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
from hybrid_optical_propagation.free_space_propagator import FreeSpacePropagator, compute_propagation_distance
from hybrid_optical_propagation.state_converter import StateConverter
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

print("[OK] 模块导入成功")


# ============================================================================
# 加载光学系统
# ============================================================================

print_section("加载光学系统")

zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

print(f"表面数量: {len(optical_system)}")
for surface in optical_system:
    mirror_str = "反射镜" if surface.is_mirror else "透射面"
    print(f"  - 表面 {surface.index}: {surface.surface_type}, "
          f"R={surface.radius:.2f}mm, {mirror_str}, "
          f"位置={surface.vertex_position}")


# ============================================================================
# 创建光源
# ============================================================================

print_section("创建光源")

wavelength_um = 0.55
w0_mm = 5.0
grid_size = 256
physical_size_mm = 50.0
num_rays = 150

source = SourceDefinition(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=0.0,
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
)

# 计算瑞利长度
wavelength_mm = wavelength_um * 1e-3
z_R = np.pi * w0_mm**2 / wavelength_mm

print(f"波长: {wavelength_um} μm")
print(f"束腰半径: {w0_mm} mm")
print(f"瑞利长度: {z_R:.2f} mm")


# ============================================================================
# 创建初始波前
# ============================================================================

print_section("创建初始波前")

amplitude, phase, pilot_beam_params, proper_wfo = source.create_initial_wavefront()

print(f"振幅形状: {amplitude.shape}")
print(f"振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
print(f"相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}]")


# ============================================================================
# 创建传播器并传播到 Surface 4 (M1)
# ============================================================================

print_section("传播到 Surface 4 (M1)")

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=wavelength_um,
    grid_size=grid_size,
    num_rays=num_rays,
)

# 传播到 Surface 4
state_4 = propagator.propagate_to_surface(4)

print(f"Surface 4 振幅范围: [{np.min(state_4.amplitude):.6f}, {np.max(state_4.amplitude):.6f}]")
print(f"Surface 4 相位范围: [{np.min(state_4.phase):.6f}, {np.max(state_4.phase):.6f}]")
print(f"Surface 4 Pilot Beam R: {state_4.pilot_beam_params.curvature_radius_mm:.2f} mm")


# ============================================================================
# 详细调试 Surface 7 的混合元件传播
# ============================================================================

print_section("详细调试 Surface 7 的混合元件传播")

# 获取 Surface 7 的信息
surface_7 = optical_system[5]  # 索引 5 对应 Surface 7
print(f"Surface 7: {surface_7.surface_type}, R={surface_7.radius}, is_mirror={surface_7.is_mirror}")
print(f"Surface 7 位置: {surface_7.vertex_position}")
print(f"Surface 7 法向量: {surface_7.surface_normal}")

# 获取光轴状态
entrance_axis_7 = propagator.get_optical_axis_at_surface(5, 'entrance')
exit_axis_7 = propagator.get_optical_axis_at_surface(5, 'exit')

print(f"\n入射光轴:")
print(f"  位置: {entrance_axis_7.position.to_array()}")
print(f"  方向: {entrance_axis_7.direction.to_array()}")

print(f"\n出射光轴:")
print(f"  位置: {exit_axis_7.position.to_array()}")
print(f"  方向: {exit_axis_7.direction.to_array()}")

# 首先传播到 Surface 7 的入射面
print("\n--- 自由空间传播到 Surface 7 入射面 ---")

# 获取 Surface 4 出射后的状态
state_after_4 = propagator._current_state

# 计算传播距离
current_pos = state_after_4.optical_axis_state.position.to_array()
target_pos = entrance_axis_7.position.to_array()
current_dir = state_after_4.optical_axis_state.direction.to_array()

distance = compute_propagation_distance(current_pos, target_pos, current_dir)
print(f"传播距离: {distance:.2f} mm")

# 执行自由空间传播
free_space_propagator = FreeSpacePropagator(wavelength_um)
state_entrance_7 = free_space_propagator.propagate(
    state_after_4,
    entrance_axis_7,
    5,
    'entrance',
)

print(f"入射面振幅范围: [{np.min(state_entrance_7.amplitude):.6f}, {np.max(state_entrance_7.amplitude):.6f}]")
print(f"入射面相位范围: [{np.min(state_entrance_7.phase):.6f}, {np.max(state_entrance_7.phase):.6f}]")


# ============================================================================
# 详细调试光线采样
# ============================================================================

print_section("详细调试光线采样")

# 使用 WavefrontToRaysSampler 采样光线
sampler = WavefrontToRaysSampler(
    amplitude=state_entrance_7.amplitude,
    phase=state_entrance_7.phase,
    physical_size=state_entrance_7.grid_sampling.physical_size_mm,
    wavelength=wavelength_um,
    num_rays=num_rays,
    distribution="hexapolar",
)

input_rays = sampler.get_output_rays()

print(f"采样光线数量: {len(np.asarray(input_rays.x))}")
print(f"光线 x 范围: [{np.min(input_rays.x):.4f}, {np.max(input_rays.x):.4f}] mm")
print(f"光线 y 范围: [{np.min(input_rays.y):.4f}, {np.max(input_rays.y):.4f}] mm")
print(f"光线 z 范围: [{np.min(input_rays.z):.4f}, {np.max(input_rays.z):.4f}] mm")

# 检查光线方向
L = np.asarray(input_rays.L)
M = np.asarray(input_rays.M)
N = np.asarray(input_rays.N)
print(f"光线 L 范围: [{np.min(L):.6f}, {np.max(L):.6f}]")
print(f"光线 M 范围: [{np.min(M):.6f}, {np.max(M):.6f}]")
print(f"光线 N 范围: [{np.min(N):.6f}, {np.max(N):.6f}]")

# 检查 OPD
opd_waves = sampler.get_ray_opd()
print(f"OPD 范围: [{np.min(opd_waves):.6f}, {np.max(opd_waves):.6f}] waves")


# ============================================================================
# 详细调试光线追迹
# ============================================================================

print_section("详细调试光线追迹")

# 创建表面定义
surface_def = SurfaceDefinition(
    surface_type='mirror',
    radius=surface_7.radius,
    thickness=surface_7.thickness,
    material='mirror',
    semi_aperture=surface_7.semi_aperture,
    conic=surface_7.conic,
)

print(f"表面定义:")
print(f"  类型: {surface_def.surface_type}")
print(f"  曲率半径: {surface_def.radius}")
print(f"  厚度: {surface_def.thickness}")
print(f"  半口径: {surface_def.semi_aperture}")

# 创建光线追迹器
raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=wavelength_um,
    chief_ray_direction=tuple(entrance_axis_7.direction.to_array()),
    entrance_position=tuple(entrance_axis_7.position.to_array()),
)

print(f"\n光线追迹器:")
print(f"  主光线方向: {raytracer.chief_ray_direction}")
print(f"  入射位置: {raytracer.entrance_position}")

# 执行光线追迹
output_rays = raytracer.trace(input_rays)

print(f"\n出射光线:")
print(f"  x 范围: [{np.min(output_rays.x):.4f}, {np.max(output_rays.x):.4f}] mm")
print(f"  y 范围: [{np.min(output_rays.y):.4f}, {np.max(output_rays.y):.4f}] mm")
print(f"  z 范围: [{np.min(output_rays.z):.4f}, {np.max(output_rays.z):.4f}] mm")

# 检查出射光线方向
L_out = np.asarray(output_rays.L)
M_out = np.asarray(output_rays.M)
N_out = np.asarray(output_rays.N)
print(f"  L 范围: [{np.min(L_out):.6f}, {np.max(L_out):.6f}]")
print(f"  M 范围: [{np.min(M_out):.6f}, {np.max(M_out):.6f}]")
print(f"  N 范围: [{np.min(N_out):.6f}, {np.max(N_out):.6f}]")

# 检查 OPD
opd_out = np.asarray(output_rays.opd)
print(f"  OPD 范围: [{np.min(opd_out):.6f}, {np.max(opd_out):.6f}] mm")


# ============================================================================
# 详细调试重建
# ============================================================================

print_section("详细调试重建")

# 计算 OPD（波长数）
opd_waves_out = opd_out / wavelength_mm
print(f"OPD (waves) 范围: [{np.min(opd_waves_out):.6f}, {np.max(opd_waves_out):.6f}]")

# 创建重建器
reconstructor = RayToWavefrontReconstructor(
    grid_size=state_entrance_7.grid_sampling.grid_size,
    sampling_mm=state_entrance_7.grid_sampling.sampling_mm,
    wavelength_um=wavelength_um,
)

print(f"\n重建器:")
print(f"  网格大小: {reconstructor.grid_size}")
print(f"  采样间隔: {reconstructor.sampling_mm} mm")
print(f"  网格范围: {reconstructor.grid_extent_mm}")

# 创建有效光线掩模
valid_mask = np.ones(len(np.asarray(input_rays.x)), dtype=bool)

# 检查输入/输出光线位置
ray_x_in = np.asarray(input_rays.x)
ray_y_in = np.asarray(input_rays.y)
ray_x_out = np.asarray(output_rays.x)
ray_y_out = np.asarray(output_rays.y)

print(f"\n输入光线位置:")
print(f"  x 范围: [{np.min(ray_x_in):.4f}, {np.max(ray_x_in):.4f}] mm")
print(f"  y 范围: [{np.min(ray_y_in):.4f}, {np.max(ray_y_in):.4f}] mm")

print(f"\n输出光线位置:")
print(f"  x 范围: [{np.min(ray_x_out):.4f}, {np.max(ray_x_out):.4f}] mm")
print(f"  y 范围: [{np.min(ray_y_out):.4f}, {np.max(ray_y_out):.4f}] mm")

# 检查输出光线是否在网格范围内
grid_half_size = reconstructor.grid_half_size_mm
in_range_x = (ray_x_out >= -grid_half_size) & (ray_x_out <= grid_half_size)
in_range_y = (ray_y_out >= -grid_half_size) & (ray_y_out <= grid_half_size)
in_range = in_range_x & in_range_y

print(f"\n网格范围: [{-grid_half_size:.4f}, {grid_half_size:.4f}] mm")
print(f"在范围内的光线数量: {np.sum(in_range)} / {len(ray_x_out)}")

if np.sum(in_range) == 0:
    print("\n[ERROR] 所有出射光线都在网格范围外！这是振幅变为零的原因。")
    print("出射光线位置与网格范围不匹配。")
    
    # 分析原因
    print("\n分析原因:")
    print(f"  入射光轴位置: {entrance_axis_7.position.to_array()}")
    print(f"  出射光轴位置: {exit_axis_7.position.to_array()}")
    print(f"  网格中心: (0, 0)")
    print(f"  出射光线中心: ({np.mean(ray_x_out):.4f}, {np.mean(ray_y_out):.4f})")

# 尝试重建
try:
    exit_complex = reconstructor.reconstruct(
        ray_x_in=ray_x_in,
        ray_y_in=ray_y_in,
        ray_x_out=ray_x_out,
        ray_y_out=ray_y_out,
        opd_waves=opd_waves_out,
        valid_mask=valid_mask,
    )
    
    exit_amplitude = np.abs(exit_complex)
    exit_phase = np.angle(exit_complex)
    
    print(f"\n重建结果:")
    print(f"  振幅范围: [{np.min(exit_amplitude):.6f}, {np.max(exit_amplitude):.6f}]")
    print(f"  相位范围: [{np.min(exit_phase):.6f}, {np.max(exit_phase):.6f}]")
    
except Exception as e:
    print(f"\n[ERROR] 重建失败: {e}")


# ============================================================================
# 可视化
# ============================================================================

print_section("生成可视化")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 入射面振幅
ax = axes[0, 0]
im = ax.imshow(state_entrance_7.amplitude, cmap='hot')
ax.set_title('入射面振幅')
plt.colorbar(im, ax=ax)

# 入射面相位
ax = axes[0, 1]
im = ax.imshow(state_entrance_7.phase, cmap='twilight')
ax.set_title('入射面相位')
plt.colorbar(im, ax=ax)

# 光线位置（入射）
ax = axes[0, 2]
ax.scatter(ray_x_in, ray_y_in, c='blue', s=1, alpha=0.5, label='入射')
ax.scatter(ray_x_out, ray_y_out, c='red', s=1, alpha=0.5, label='出射')
ax.axhline(y=-grid_half_size, color='green', linestyle='--', label='网格边界')
ax.axhline(y=grid_half_size, color='green', linestyle='--')
ax.axvline(x=-grid_half_size, color='green', linestyle='--')
ax.axvline(x=grid_half_size, color='green', linestyle='--')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('光线位置')
ax.legend()
ax.set_aspect('equal')

# 光线 OPD
ax = axes[1, 0]
sc = ax.scatter(ray_x_out, ray_y_out, c=opd_waves_out, cmap='viridis', s=5)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('出射光线 OPD (waves)')
plt.colorbar(sc, ax=ax)
ax.set_aspect('equal')

# 光线方向
ax = axes[1, 1]
ax.quiver(ray_x_out[::10], ray_y_out[::10], L_out[::10], M_out[::10], scale=20)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('出射光线方向')
ax.set_aspect('equal')

# 空白
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('debug_amplitude_zero.png', dpi=150)
print(f"图像已保存到 debug_amplitude_zero.png")


print_section("调试完成")
