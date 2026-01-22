"""
Surface_3 误差来源逐步排查

按顺序检查：
1. Surface_3 入射面的仿真复振幅
2. 透过薄相位元件后的光线 OPD 与光线方向
3. 光线抵达出射面处的光线 OPD
4. 重建复振幅

ZMX 文件: optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def print_section(title: str) -> None:
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
from hybrid_optical_propagation.state_converter import StateConverter
from wavefront_to_rays import WavefrontToRaysSampler

print("[OK] 模块导入成功")


# ============================================================================
# 加载光学系统并传播到 Surface_3 入射面
# ============================================================================

print_section("加载光学系统")

zmx_file_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"
optical_system = load_optical_system_from_zmx(zmx_file_path)

print(f"ZMX 文件: {zmx_file_path}")
print(f"表面数量: {len(optical_system)}")
for surface in optical_system:
    print(f"  - 表面 {surface.index}: {surface.surface_type}, "
          f"R={surface.radius:.2f}mm, mirror={surface.is_mirror}, "
          f"vertex={surface.vertex_position}")

# 找到 Surface_3（第一个反射镜 M1，index=4）
mirror_surface = None
for s in optical_system:
    if s.is_mirror:
        mirror_surface = s
        print(f"\n第一个反射镜: Surface {s.index}")
        break


# ============================================================================
# 创建光源并传播到 Surface_3 入射面
# ============================================================================

print_section("创建光源并传播到 Surface_3 入射面")

wavelength_um = 0.55
w0_mm = 5.0
grid_size = 256
physical_size_mm = 40.0

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
print(f"瑞利长度: {z_R:.1f} mm")


# 创建传播器
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=wavelength_um,
    grid_size=grid_size,
    num_rays=150,
)

# 执行传播
result = propagator.propagate()

if not result.success:
    print(f"[FAIL] 传播失败: {result.error_message}")
    sys.exit(1)

print(f"[OK] 传播成功，共 {len(result.surface_states)} 个状态")

# 找到 Surface_3 入射面状态
surface3_entrance = None
surface3_exit = None
for state in result.surface_states:
    if state.surface_index == 3:
        if state.position == 'entrance':
            surface3_entrance = state
        elif state.position == 'exit':
            surface3_exit = state

if surface3_entrance is None:
    print("[FAIL] 未找到 Surface_3 入射面状态")
    sys.exit(1)

print(f"\n找到 Surface_3 入射面状态")


# ============================================================================
# 步骤 1: 检查 Surface_3 入射面的仿真复振幅
# ============================================================================

print_section("步骤 1: Surface_3 入射面的仿真复振幅")

amplitude_in = surface3_entrance.amplitude
phase_in = surface3_entrance.phase
pilot_params_in = surface3_entrance.pilot_beam_params
grid_sampling_in = GridSampling.from_proper(surface3_entrance.proper_wfo)

print(f"网格大小: {amplitude_in.shape}")
print(f"物理尺寸: {grid_sampling_in.physical_size_mm:.2f} mm")
print(f"采样间隔: {grid_sampling_in.sampling_mm:.4f} mm")

print(f"\n振幅统计:")
print(f"  最大值: {np.max(amplitude_in):.6f}")
print(f"  最小值: {np.min(amplitude_in):.6f}")

print(f"\n相位统计:")
print(f"  最大值: {np.max(phase_in):.4f} rad ({np.max(phase_in)/(2*np.pi):.4f} waves)")
print(f"  最小值: {np.min(phase_in):.4f} rad ({np.min(phase_in)/(2*np.pi):.4f} waves)")
print(f"  范围: {np.max(phase_in) - np.min(phase_in):.4f} rad")

print(f"\nPilot Beam 参数:")
print(f"  曲率半径: {pilot_params_in.curvature_radius_mm:.2f} mm")
print(f"  光斑大小: {pilot_params_in.spot_size_mm:.4f} mm")
print(f"  束腰位置: {pilot_params_in.waist_position_mm:.2f} mm")

# 计算 Pilot Beam 参考相位
pilot_phase_in = pilot_params_in.compute_phase_grid(
    grid_sampling_in.grid_size,
    grid_sampling_in.physical_size_mm,
)

print(f"\nPilot Beam 参考相位:")
print(f"  最大值: {np.max(pilot_phase_in):.4f} rad")
print(f"  最小值: {np.min(pilot_phase_in):.4f} rad")

# 计算相位误差
valid_mask = amplitude_in > 0.01 * np.max(amplitude_in)
phase_diff_in = np.angle(np.exp(1j * (phase_in - pilot_phase_in)))
phase_rms_in = np.sqrt(np.mean(phase_diff_in[valid_mask]**2))

print(f"\n入射面相位误差（相对于 Pilot Beam）:")
print(f"  RMS: {phase_rms_in/(2*np.pi):.6f} waves")
print(f"  这个误差应该很小（< 0.001 waves）")


# ============================================================================
# 步骤 2: 透过薄相位元件后的光线 OPD 与光线方向
# ============================================================================

print_section("步骤 2: 透过薄相位元件后的光线 OPD 与光线方向")

# 使用 WavefrontToRaysSampler 采样光线
sampler = WavefrontToRaysSampler(
    amplitude=amplitude_in,
    phase=phase_in,
    physical_size=grid_sampling_in.physical_size_mm,
    wavelength=wavelength_um,
    num_rays=150,
    distribution="hexapolar",
)

# 获取出射光线
output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
ray_L, ray_M, ray_N = sampler.get_ray_directions()
ray_opd_waves = sampler.get_ray_opd()  # 相对于主光线的 OPD（波长数）
ray_opd_raw_mm = sampler.get_ray_opd_raw()  # 原始 OPD（mm）

print(f"采样光线数量: {len(ray_x)}")

print(f"\n光线位置统计:")
print(f"  X 范围: [{np.min(ray_x):.4f}, {np.max(ray_x):.4f}] mm")
print(f"  Y 范围: [{np.min(ray_y):.4f}, {np.max(ray_y):.4f}] mm")

print(f"\n光线方向统计:")
print(f"  L 范围: [{np.min(ray_L):.6f}, {np.max(ray_L):.6f}]")
print(f"  M 范围: [{np.min(ray_M):.6f}, {np.max(ray_M):.6f}]")
print(f"  N 范围: [{np.min(ray_N):.6f}, {np.max(ray_N):.6f}]")

print(f"\n光线 OPD 统计（相对于主光线）:")
print(f"  范围: [{np.min(ray_opd_waves):.6f}, {np.max(ray_opd_waves):.6f}] waves")
print(f"  RMS: {np.std(ray_opd_waves):.6f} waves")

print(f"\n光线 OPD 统计（原始，mm）:")
print(f"  范围: [{np.min(ray_opd_raw_mm):.6f}, {np.max(ray_opd_raw_mm):.6f}] mm")

# 计算期望的 Pilot Beam OPD
r_sq = ray_x**2 + ray_y**2
if np.isinf(pilot_params_in.curvature_radius_mm):
    expected_pilot_opd_mm = np.zeros_like(ray_x)
else:
    expected_pilot_opd_mm = r_sq / (2 * pilot_params_in.curvature_radius_mm)

expected_pilot_opd_waves = expected_pilot_opd_mm / wavelength_mm

print(f"\n期望的 Pilot Beam OPD（波长数）:")
print(f"  范围: [{np.min(expected_pilot_opd_waves):.6f}, {np.max(expected_pilot_opd_waves):.6f}]")

# 比较实际 OPD 与期望 OPD
# 注意：ray_opd_waves 是相对于主光线的，expected_pilot_opd_waves 也是相对于主光线的
opd_diff = ray_opd_waves - expected_pilot_opd_waves
print(f"\n光线 OPD 与 Pilot Beam OPD 的差异:")
print(f"  范围: [{np.min(opd_diff):.6f}, {np.max(opd_diff):.6f}] waves")
print(f"  RMS: {np.std(opd_diff):.6f} waves")


# ============================================================================
# 步骤 3: 光线追迹到出射面
# ============================================================================

print_section("步骤 3: 光线追迹到出射面")

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

# 获取 Surface_3 对应的表面定义
# Surface_3 是 index=4 的反射镜
surface_def = SurfaceDefinition(
    surface_type='mirror',
    radius=mirror_surface.radius,
    thickness=0.0,  # 反射镜厚度为 0
    material='mirror',
    semi_aperture=mirror_surface.semi_aperture,
    conic=mirror_surface.conic,
)

print(f"表面定义:")
print(f"  类型: {surface_def.surface_type}")
print(f"  曲率半径: {surface_def.radius}")
print(f"  圆锥常数: {surface_def.conic}")
print(f"  半口径: {surface_def.semi_aperture}")

# 获取入射光轴状态
entrance_axis = surface3_entrance.optical_axis_state
print(f"\n入射光轴状态:")
print(f"  位置: {entrance_axis.position.to_array()}")
print(f"  方向: {entrance_axis.direction.to_array()}")

# 创建光线追迹器
raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=wavelength_um,
    chief_ray_direction=tuple(entrance_axis.direction.to_array()),
    entrance_position=tuple(entrance_axis.position.to_array()),
)

# 设置输入光线的 OPD
# 注意：这里需要将相对 OPD 转换为绝对 OPD
input_rays = sampler.get_output_rays()
input_rays.opd = ray_opd_waves * wavelength_mm  # 转换为 mm

print(f"\n输入光线 OPD（mm）:")
print(f"  范围: [{np.min(np.asarray(input_rays.opd)):.6f}, {np.max(np.asarray(input_rays.opd)):.6f}]")

# 执行光线追迹
traced_rays = raytracer.trace(input_rays)

# 获取追迹后的光线数据
traced_x = np.asarray(traced_rays.x)
traced_y = np.asarray(traced_rays.y)
traced_z = np.asarray(traced_rays.z)
traced_L = np.asarray(traced_rays.L)
traced_M = np.asarray(traced_rays.M)
traced_N = np.asarray(traced_rays.N)
traced_opd = np.asarray(traced_rays.opd)

print(f"\n追迹后光线位置:")
print(f"  X 范围: [{np.min(traced_x):.4f}, {np.max(traced_x):.4f}] mm")
print(f"  Y 范围: [{np.min(traced_y):.4f}, {np.max(traced_y):.4f}] mm")
print(f"  Z 范围: [{np.min(traced_z):.4f}, {np.max(traced_z):.4f}] mm")

print(f"\n追迹后光线方向:")
print(f"  L 范围: [{np.min(traced_L):.6f}, {np.max(traced_L):.6f}]")
print(f"  M 范围: [{np.min(traced_M):.6f}, {np.max(traced_M):.6f}]")
print(f"  N 范围: [{np.min(traced_N):.6f}, {np.max(traced_N):.6f}]")

print(f"\n追迹后光线 OPD（mm）:")
print(f"  范围: [{np.min(traced_opd):.6f}, {np.max(traced_opd):.6f}]")
print(f"  RMS: {np.std(traced_opd):.6f} mm")

# 转换为波长数
traced_opd_waves = traced_opd / wavelength_mm
print(f"\n追迹后光线 OPD（波长数）:")
print(f"  范围: [{np.min(traced_opd_waves):.6f}, {np.max(traced_opd_waves):.6f}]")
print(f"  RMS: {np.std(traced_opd_waves):.6f} waves")


# ============================================================================
# 步骤 4: 重建复振幅
# ============================================================================

print_section("步骤 4: 重建复振幅")

from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor

# 创建重建器
reconstructor = RayToWavefrontReconstructor(
    grid_size=grid_sampling_in.grid_size,
    sampling_mm=grid_sampling_in.sampling_mm,
    wavelength_um=wavelength_um,
)

# 创建有效光线掩模
valid_mask = np.ones(len(ray_x), dtype=bool)

# 重建复振幅
exit_complex = reconstructor.reconstruct(
    ray_x_in=ray_x,
    ray_y_in=ray_y,
    ray_x_out=traced_x,
    ray_y_out=traced_y,
    opd_waves=traced_opd_waves,
    valid_mask=valid_mask,
)

# 分离振幅和相位
exit_amplitude = np.abs(exit_complex)
exit_phase = np.angle(exit_complex)

print(f"重建后振幅统计:")
print(f"  最大值: {np.max(exit_amplitude):.6f}")
print(f"  最小值: {np.min(exit_amplitude):.6f}")
print(f"  是否有 NaN: {np.any(np.isnan(exit_amplitude))}")

print(f"\n重建后相位统计:")
print(f"  最大值: {np.max(exit_phase):.4f} rad")
print(f"  最小值: {np.min(exit_phase):.4f} rad")
print(f"  范围: {np.max(exit_phase) - np.min(exit_phase):.4f} rad")
print(f"  是否有 NaN: {np.any(np.isnan(exit_phase))}")

# 计算与 Pilot Beam 的误差
# 注意：反射镜后 Pilot Beam 参数应该更新
pilot_params_out = pilot_params_in.apply_mirror(mirror_surface.radius)

print(f"\n反射后 Pilot Beam 参数:")
print(f"  曲率半径: {pilot_params_out.curvature_radius_mm:.2f} mm")
print(f"  光斑大小: {pilot_params_out.spot_size_mm:.4f} mm")

pilot_phase_out = pilot_params_out.compute_phase_grid(
    grid_sampling_in.grid_size,
    grid_sampling_in.physical_size_mm,
)

print(f"\n反射后 Pilot Beam 参考相位:")
print(f"  最大值: {np.max(pilot_phase_out):.4f} rad")
print(f"  最小值: {np.min(pilot_phase_out):.4f} rad")

# 计算相位误差
valid_mask_grid = exit_amplitude > 0.01 * np.max(exit_amplitude)
if np.sum(valid_mask_grid) > 0:
    phase_diff_out = np.angle(np.exp(1j * (exit_phase - pilot_phase_out)))
    phase_rms_out = np.sqrt(np.mean(phase_diff_out[valid_mask_grid]**2))
    print(f"\n出射面相位误差（相对于 Pilot Beam）:")
    print(f"  RMS: {phase_rms_out/(2*np.pi):.6f} waves")
else:
    print(f"\n[WARNING] 无有效数据计算相位误差")


# ============================================================================
# 步骤 5: 与实际 Surface_3 exit 状态比较
# ============================================================================

print_section("步骤 5: 与实际 Surface_3 exit 状态比较")

if surface3_exit is not None:
    actual_amplitude = surface3_exit.amplitude
    actual_phase = surface3_exit.phase
    actual_grid_sampling = GridSampling.from_proper(surface3_exit.proper_wfo)
    
    print(f"实际 Surface_3 exit 状态:")
    print(f"  网格大小: {actual_amplitude.shape}")
    print(f"  物理尺寸: {actual_grid_sampling.physical_size_mm:.2f} mm")
    print(f"  采样间隔: {actual_grid_sampling.sampling_mm:.4f} mm")
    
    print(f"\n实际振幅统计:")
    print(f"  最大值: {np.max(actual_amplitude):.6f}")
    print(f"  最小值: {np.min(actual_amplitude):.6f}")
    
    print(f"\n实际相位统计:")
    print(f"  最大值: {np.max(actual_phase):.4f} rad")
    print(f"  最小值: {np.min(actual_phase):.4f} rad")
    print(f"  范围: {np.max(actual_phase) - np.min(actual_phase):.4f} rad")
    
    # 注意：实际状态的网格大小可能不同（因为 tilted_asm 的 expand=True）
    if actual_amplitude.shape != exit_amplitude.shape:
        print(f"\n[WARNING] 网格大小不同！")
        print(f"  重建: {exit_amplitude.shape}")
        print(f"  实际: {actual_amplitude.shape}")
else:
    print("[WARNING] 未找到 Surface_3 exit 状态")


# ============================================================================
# 绘制诊断图
# ============================================================================

print_section("绘制诊断图")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

extent_in = [-grid_sampling_in.physical_size_mm/2, grid_sampling_in.physical_size_mm/2,
             -grid_sampling_in.physical_size_mm/2, grid_sampling_in.physical_size_mm/2]

# 第一行：入射面
im1 = axes[0, 0].imshow(amplitude_in / np.max(amplitude_in), extent=extent_in, cmap='viridis')
axes[0, 0].set_title('Entrance Amplitude')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(phase_in, extent=extent_in, cmap='twilight')
axes[0, 1].set_title('Entrance Phase (rad)')
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(pilot_phase_in, extent=extent_in, cmap='twilight')
axes[0, 2].set_title('Pilot Beam Phase (rad)')
plt.colorbar(im3, ax=axes[0, 2])

# 计算网格上的相位误差
valid_mask_grid_in = amplitude_in > 0.01 * np.max(amplitude_in)
phase_diff_in_grid = np.angle(np.exp(1j * (phase_in - pilot_phase_in)))
phase_diff_in_masked = np.where(valid_mask_grid_in, phase_diff_in_grid, np.nan)
im4 = axes[0, 3].imshow(phase_diff_in_masked, extent=extent_in, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
axes[0, 3].set_title(f'Entrance Phase Error\nRMS={phase_rms_in/(2*np.pi):.6f} waves')
plt.colorbar(im4, ax=axes[0, 3])

# 第二行：光线数据
axes[1, 0].scatter(ray_x, ray_y, c=ray_opd_waves, cmap='RdBu_r', s=10)
axes[1, 0].set_title('Input Ray OPD (waves)')
axes[1, 0].set_xlabel('X (mm)')
axes[1, 0].set_ylabel('Y (mm)')
axes[1, 0].set_aspect('equal')
plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])

# 删除重复代码
# axes[1, 0].scatter(ray_x, ra
axes[1, 1].scatter(traced_x, traced_y, c=traced_opd_waves, cmap='RdBu_r', s=10)
axes[1, 1].set_title('Traced Ray OPD (waves)')
axes[1, 1].set_xlabel('X (mm)')
axes[1, 1].set_ylabel('Y (mm)')
axes[1, 1].set_aspect('equal')
plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])

axes[1, 2].hist(ray_opd_waves, bins=30, alpha=0.7, label='Input')
axes[1, 2].hist(traced_opd_waves, bins=30, alpha=0.7, label='Traced')
axes[1, 2].set_title('OPD Distribution')
axes[1, 2].set_xlabel('OPD (waves)')
axes[1, 2].legend()

axes[1, 3].scatter(ray_opd_waves, traced_opd_waves, s=10, alpha=0.5)
axes[1, 3].plot([np.min(ray_opd_waves), np.max(ray_opd_waves)],
                [np.min(ray_opd_waves), np.max(ray_opd_waves)], 'r--')
axes[1, 3].set_title('Input vs Traced OPD')
axes[1, 3].set_xlabel('Input OPD (waves)')
axes[1, 3].set_ylabel('Traced OPD (waves)')

# 第三行：重建结果
im9 = axes[2, 0].imshow(exit_amplitude / np.max(exit_amplitude) if np.max(exit_amplitude) > 0 else exit_amplitude,
                        extent=extent_in, cmap='viridis')
axes[2, 0].set_title('Reconstructed Amplitude')
plt.colorbar(im9, ax=axes[2, 0])

im10 = axes[2, 1].imshow(exit_phase, extent=extent_in, cmap='twilight')
axes[2, 1].set_title('Reconstructed Phase (rad)')
plt.colorbar(im10, ax=axes[2, 1])

im11 = axes[2, 2].imshow(pilot_phase_out, extent=extent_in, cmap='twilight')
axes[2, 2].set_title('Exit Pilot Beam Phase (rad)')
plt.colorbar(im11, ax=axes[2, 2])

if np.sum(valid_mask_grid) > 0:
    phase_diff_out_masked = np.where(valid_mask_grid, phase_diff_out, np.nan)
    vmax = max(0.1, np.nanmax(np.abs(phase_diff_out_masked)))
    im12 = axes[2, 3].imshow(phase_diff_out_masked, extent=extent_in, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, 3].set_title(f'Exit Phase Error\nRMS={phase_rms_out/(2*np.pi):.6f} waves')
    plt.colorbar(im12, ax=axes[2, 3])
else:
    axes[2, 3].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[2, 3].transAxes)
    axes[2, 3].set_title('Exit Phase Error')

plt.tight_layout()
fig.savefig('debug_surface3_step_by_step.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n[OK] 诊断图已保存: debug_surface3_step_by_step.png")

print_section("总结")
print(f"""
误差来源排查结果：

1. 入射面仿真复振幅:
   - 相位 RMS 误差: {phase_rms_in/(2*np.pi):.6f} waves
   - 状态: {'正常' if phase_rms_in/(2*np.pi) < 0.001 else '异常'}

2. 光线采样后的 OPD:
   - OPD 范围: [{np.min(ray_opd_waves):.6f}, {np.max(ray_opd_waves):.6f}] waves
   - 与 Pilot Beam 差异 RMS: {np.std(opd_diff):.6f} waves
   - 状态: {'正常' if np.std(opd_diff) < 0.001 else '异常'}

3. 光线追迹后的 OPD:
   - OPD 范围: [{np.min(traced_opd_waves):.6f}, {np.max(traced_opd_waves):.6f}] waves
   - OPD RMS: {np.std(traced_opd_waves):.6f} waves

4. 重建后的相位误差:
   - 相位 RMS 误差: {phase_rms_out/(2*np.pi) if np.sum(valid_mask_grid) > 0 else 'N/A':.6f} waves
""")
