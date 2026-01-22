"""
调试脚本：逐步检查 Surface_0 处的混合光线追迹行为

目标：
1. 确认入射面复振幅与 Pilot Beam 误差很小
2. 检查入射面光线通过薄元件后的光程是否等于 Pilot Beam 相位
3. 检查出射面光线相对于 Pilot Beam 的残差相位是否很小

基于 ZMX 文件定义的系统进行测试。
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
# 第一部分：导入模块并加载 ZMX 系统
# ============================================================================

print_section("第一部分：导入模块并加载 ZMX 系统")

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    PilotBeamParams,
    GridSampling,
    PropagationState,
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
from hybrid_optical_propagation.free_space_propagator import FreeSpacePropagator
from sequential_system.coordinate_tracking import OpticalAxisState, Position3D, RayDirection
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

print("[OK] 模块导入成功")

# 加载 ZMX 文件
zmx_file = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
print(f"加载 ZMX 文件: {zmx_file}")

try:
    optical_system = load_optical_system_from_zmx(zmx_file)
    print(f"[OK] 加载成功，表面数量: {len(optical_system)}")
    
    # 打印表面信息
    for i, surface in enumerate(optical_system):
        print(f"  Surface {i}: type={surface.surface_type}, "
              f"is_mirror={surface.is_mirror}, "
              f"radius={surface.radius:.2f}, "
              f"vertex={surface.vertex_position}")
except Exception as e:
    print(f"[FAIL] 加载失败: {e}")
    sys.exit(1)


# ============================================================================
# 第二部分：创建光源并初始化传播
# ============================================================================

print_section("第二部分：创建光源并初始化传播")

# 光源参数
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

print(f"光源参数:")
print(f"  波长: {wavelength_um} um")
print(f"  束腰半径: {w0_mm} mm")
print(f"  网格大小: {grid_size}")
print(f"  物理尺寸: {physical_size_mm} mm")

# 创建初始波前
simulation_amplitude, pilot_beam_params, proper_wfo = source.create_initial_wavefront()

# 获取实际的网格采样（从 PROPER）
grid_sampling = GridSampling.from_proper(proper_wfo)
print(f"\n实际网格采样:")
print(f"  物理尺寸: {grid_sampling.physical_size_mm:.2f} mm")
print(f"  采样间隔: {grid_sampling.sampling_mm:.4f} mm")

print(f"\n初始 Pilot Beam 参数:")
print(f"  曲率半径: {pilot_beam_params.curvature_radius_mm}")
print(f"  光斑大小: {pilot_beam_params.spot_size_mm:.4f} mm")
print(f"  束腰半径: {pilot_beam_params.waist_radius_mm:.4f} mm")


# ============================================================================
# 第三部分：传播到 Surface_0 入射面
# ============================================================================

print_section("第三部分：传播到 Surface_0 入射面")

# 找到第一个实际的光学表面（跳过坐标断点等）
surface_0 = None
surface_0_index = None
for i, surface in enumerate(optical_system):
    if surface.is_mirror or surface.surface_type == 'standard':
        surface_0 = surface
        surface_0_index = i
        break

if surface_0 is None:
    print("[FAIL] 未找到有效的光学表面")
    sys.exit(1)

print(f"Surface_0 信息:")
print(f"  索引: {surface_0_index}")
print(f"  类型: {surface_0.surface_type}")
print(f"  是否反射镜: {surface_0.is_mirror}")
print(f"  曲率半径: {surface_0.radius}")
print(f"  顶点位置: {surface_0.vertex_position}")

# 计算传播距离
propagation_distance = np.linalg.norm(surface_0.vertex_position)
print(f"  传播距离: {propagation_distance:.2f} mm")

# 创建入射光轴状态
entrance_axis = OpticalAxisState(
    position=Position3D.from_array(surface_0.vertex_position),
    direction=RayDirection(0.0, 0.0, 1.0),
    path_length=propagation_distance,
)

# 执行自由空间传播
free_space_propagator = FreeSpacePropagator(wavelength_um)

initial_axis = OpticalAxisState(
    position=Position3D(0.0, 0.0, 0.0),
    direction=RayDirection(0.0, 0.0, 1.0),
    path_length=0.0,
)

initial_state = PropagationState(
    surface_index=-1,
    position='source',
    simulation_amplitude=simulation_amplitude,
    pilot_beam_params=pilot_beam_params,
    proper_wfo=proper_wfo,
    optical_axis_state=initial_axis,
    grid_sampling=grid_sampling,
)

# 传播到入射面
print("\n执行自由空间传播...")
entrance_state = free_space_propagator.propagate(
    initial_state,
    entrance_axis,
    target_surface_index=surface_0_index,
    target_position='entrance',
)

print(f"[OK] 传播完成")
print(f"\n入射面 Pilot Beam 参数:")
print(f"  曲率半径: {entrance_state.pilot_beam_params.curvature_radius_mm:.2f} mm")
print(f"  光斑大小: {entrance_state.pilot_beam_params.spot_size_mm:.4f} mm")


# ============================================================================
# 第四部分：检查入射面复振幅与 Pilot Beam 的误差
# ============================================================================

print_section("第四部分：检查入射面复振幅与 Pilot Beam 的误差")

# 提取入射面复振幅
entrance_amplitude = np.abs(entrance_state.simulation_amplitude)
entrance_phase = np.angle(entrance_state.simulation_amplitude)

# 计算 Pilot Beam 参考相位
pilot_phase_entrance = entrance_state.pilot_beam_params.compute_phase_grid(
    grid_size, grid_sampling.physical_size_mm
)

# 计算相位误差
phase_diff = np.angle(np.exp(1j * (entrance_phase - pilot_phase_entrance)))

# 有效区域掩模
valid_mask = entrance_amplitude > 0.01 * np.max(entrance_amplitude)

# 计算误差统计
phase_rms_rad = np.sqrt(np.mean(phase_diff[valid_mask]**2))
phase_rms_waves = phase_rms_rad / (2 * np.pi)
phase_pv_rad = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
phase_pv_waves = phase_pv_rad / (2 * np.pi)

print(f"入射面复振幅与 Pilot Beam 的误差:")
print(f"  相位 RMS 误差: {phase_rms_waves:.6f} waves ({phase_rms_rad:.6f} rad)")
print(f"  相位 PV 误差: {phase_pv_waves:.6f} waves ({phase_pv_rad:.6f} rad)")

if phase_rms_waves < 0.01:
    print(f"  [OK] 入射面误差很小，符合预期")
else:
    print(f"  [WARNING] 入射面误差较大，需要检查")


# ============================================================================
# 第五部分：检查入射面光线 OPD 是否等于 Pilot Beam 相位
# ============================================================================

print_section("第五部分：检查入射面光线 OPD 是否等于 Pilot Beam 相位")

# 使用 WavefrontToRaysSampler 采样光线
num_rays = 100
sampler = WavefrontToRaysSampler(
    wavefront_amplitude=entrance_state.simulation_amplitude,
    physical_size=grid_sampling.physical_size_mm,
    wavelength=wavelength_um,
    num_rays=num_rays,
    distribution="hexapolar",
)

# 获取光线位置
ray_x, ray_y = sampler.get_ray_positions()
output_rays = sampler.get_output_rays()

print(f"采样光线数量: {len(ray_x)}")

# 计算光线位置处的 Pilot Beam 相位
r_sq = ray_x**2 + ray_y**2
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

R = entrance_state.pilot_beam_params.curvature_radius_mm
if np.isinf(R):
    pilot_phase_at_rays = np.zeros_like(r_sq)
else:
    pilot_phase_at_rays = k * r_sq / (2 * R)

# 将 Pilot Beam 相位转换为 OPD (mm)
pilot_opd_mm = pilot_phase_at_rays / k

# 获取 WavefrontToRaysSampler 输出的光线 OPD
sampler_opd_mm = np.asarray(output_rays.opd)

# 比较
opd_diff_mm = sampler_opd_mm - pilot_opd_mm
opd_diff_waves = opd_diff_mm / wavelength_mm

print(f"\n光线 OPD 与 Pilot Beam 相位的比较:")
print(f"  Pilot Beam OPD 范围: [{np.min(pilot_opd_mm):.6f}, {np.max(pilot_opd_mm):.6f}] mm")
print(f"  Sampler OPD 范围: [{np.min(sampler_opd_mm):.6f}, {np.max(sampler_opd_mm):.6f}] mm")
print(f"  OPD 差异 RMS: {np.std(opd_diff_waves):.6f} waves")
print(f"  OPD 差异 PV: {np.max(opd_diff_waves) - np.min(opd_diff_waves):.6f} waves")

# 检查 WavefrontToRaysSampler 是否使用了折叠相位
# 从仿真复振幅中提取相位（使用 np.angle，会折叠）
sim_phase_at_rays = np.zeros(len(ray_x))
for i in range(len(ray_x)):
    # 找到最近的网格点
    ix = int((ray_x[i] + grid_sampling.physical_size_mm/2) / grid_sampling.sampling_mm)
    iy = int((ray_y[i] + grid_sampling.physical_size_mm/2) / grid_sampling.sampling_mm)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    sim_phase_at_rays[i] = entrance_phase[iy, ix]

# 将仿真相位转换为 OPD
sim_opd_mm = sim_phase_at_rays / k

print(f"\n仿真复振幅相位（折叠）与 Pilot Beam 相位的比较:")
print(f"  仿真相位 OPD 范围: [{np.min(sim_opd_mm):.6f}, {np.max(sim_opd_mm):.6f}] mm")
print(f"  Pilot Beam OPD 范围: [{np.min(pilot_opd_mm):.6f}, {np.max(pilot_opd_mm):.6f}] mm")

# 检查是否存在相位折叠
phase_diff_at_rays = sim_phase_at_rays - pilot_phase_at_rays
wrapped_diff = np.angle(np.exp(1j * phase_diff_at_rays))
print(f"  相位差（折叠后）RMS: {np.std(wrapped_diff):.6f} rad")


# ============================================================================
# 第六部分：手动执行光线追迹并检查出射面 OPD
# ============================================================================

print_section("第六部分：手动执行光线追迹并检查出射面 OPD")

# 创建表面定义
surface_def = SurfaceDefinition(
    surface_type='mirror' if surface_0.is_mirror else 'refract',
    radius=surface_0.radius,
    thickness=0.0,
    material='mirror' if surface_0.is_mirror else surface_0.material,
    semi_aperture=surface_0.semi_aperture,
    conic=surface_0.conic,
)

print(f"表面定义:")
print(f"  类型: {surface_def.surface_type}")
print(f"  曲率半径: {surface_def.radius}")
print(f"  圆锥常数: {surface_def.conic}")

# 创建光线追迹器
# 入射方向沿 +Z
chief_ray_direction = (0.0, 0.0, 1.0)
entrance_position = tuple(surface_0.vertex_position)

raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=wavelength_um,
    chief_ray_direction=chief_ray_direction,
    entrance_position=entrance_position,
)

print(f"\n光线追迹器配置:")
print(f"  入射方向: {chief_ray_direction}")
print(f"  入射位置: {entrance_position}")
print(f"  出射方向: {raytracer.exit_chief_direction}")

# 准备输入光线
# 关键：将光线 OPD 设置为 Pilot Beam 相位对应的 OPD
from optiland.rays import RealRays

input_rays = RealRays(
    x=ray_x,
    y=ray_y,
    z=np.zeros_like(ray_x),
    L=np.zeros_like(ray_x),
    M=np.zeros_like(ray_x),
    N=np.ones_like(ray_x),
    intensity=np.ones_like(ray_x),
    wavelength=np.full_like(ray_x, wavelength_um),
)

# 设置光线 OPD 为 Pilot Beam 相位对应的 OPD
input_rays.opd = pilot_opd_mm.copy()

print(f"\n输入光线 OPD（Pilot Beam 相位）:")
print(f"  范围: [{np.min(input_rays.opd):.6f}, {np.max(input_rays.opd):.6f}] mm")

# 执行光线追迹
output_rays = raytracer.trace(input_rays)

print(f"\n输出光线:")
print(f"  位置 X 范围: [{np.min(output_rays.x):.4f}, {np.max(output_rays.x):.4f}] mm")
print(f"  位置 Y 范围: [{np.min(output_rays.y):.4f}, {np.max(output_rays.y):.4f}] mm")
print(f"  OPD 范围: [{np.min(output_rays.opd):.6f}, {np.max(output_rays.opd):.6f}] mm")

# 计算出射面 Pilot Beam 参数
# 反射镜会改变 Pilot Beam 参数
exit_pilot_params = entrance_state.pilot_beam_params.apply_mirror(surface_0.radius)

print(f"\n出射面 Pilot Beam 参数:")
print(f"  曲率半径: {exit_pilot_params.curvature_radius_mm:.2f} mm")
print(f"  光斑大小: {exit_pilot_params.spot_size_mm:.4f} mm")

# 计算出射光线位置处的 Pilot Beam 相位
exit_ray_x = np.asarray(output_rays.x)
exit_ray_y = np.asarray(output_rays.y)
exit_r_sq = exit_ray_x**2 + exit_ray_y**2

R_exit = exit_pilot_params.curvature_radius_mm
if np.isinf(R_exit):
    pilot_phase_exit = np.zeros_like(exit_r_sq)
else:
    pilot_phase_exit = k * exit_r_sq / (2 * R_exit)

pilot_opd_exit_mm = pilot_phase_exit / k

# 计算出射光线 OPD 相对于 Pilot Beam 的残差
output_opd_mm = np.asarray(output_rays.opd)
residual_opd_mm = output_opd_mm - pilot_opd_exit_mm
residual_opd_waves = residual_opd_mm / wavelength_mm

print(f"\n出射光线 OPD 与 Pilot Beam 的比较:")
print(f"  出射光线 OPD 范围: [{np.min(output_opd_mm):.6f}, {np.max(output_opd_mm):.6f}] mm")
print(f"  Pilot Beam OPD 范围: [{np.min(pilot_opd_exit_mm):.6f}, {np.max(pilot_opd_exit_mm):.6f}] mm")
print(f"  残差 OPD RMS: {np.std(residual_opd_waves):.6f} waves")
print(f"  残差 OPD PV: {np.max(residual_opd_waves) - np.min(residual_opd_waves):.6f} waves")

# 检查残差是否很小
if np.std(residual_opd_waves) < 0.01:
    print(f"  [OK] 出射面残差 OPD 很小，符合预期")
else:
    print(f"  [WARNING] 出射面残差 OPD 较大，需要检查")
    print(f"\n  可能的问题:")
    print(f"  1. 入射光线 OPD 设置不正确")
    print(f"  2. 光线追迹过程中 OPD 计算有误")
    print(f"  3. Pilot Beam 参数更新不正确")


# ============================================================================
# 第七部分：详细分析 OPD 计算
# ============================================================================

print_section("第七部分：详细分析 OPD 计算")

# 分析入射光线 OPD 的组成
print("入射光线 OPD 分析:")
print(f"  设置的 OPD（Pilot Beam 相位）: {np.mean(pilot_opd_mm):.6f} mm (平均)")

# 分析出射光线 OPD 的组成
# 出射 OPD = 入射 OPD + 传播 OPD
# 对于反射镜，传播 OPD 应该是从入射面到镜面再到出射面的光程

# 计算几何光程
# 入射光线从入射面（z=0）到镜面顶点（z=0）的距离
# 对于平面镜，这个距离是 0
# 对于曲面镜，这个距离取决于光线位置

# 检查光线追迹器的 OPD 增量
print(f"\n光线追迹 OPD 增量:")
opd_increment = output_opd_mm - np.asarray(input_rays.opd)
print(f"  OPD 增量范围: [{np.min(opd_increment):.6f}, {np.max(opd_increment):.6f}] mm")
print(f"  OPD 增量 RMS: {np.std(opd_increment):.6f} mm")

# 对于理想反射镜，OPD 增量应该是 2 * sag（往返光程）
# sag = r^2 / (2*R) 对于球面
if not np.isinf(surface_0.radius):
    expected_sag = (ray_x**2 + ray_y**2) / (2 * surface_0.radius)
    expected_opd_increment = 2 * expected_sag  # 往返
    print(f"\n理论 OPD 增量（2*sag）:")
    print(f"  范围: [{np.min(expected_opd_increment):.6f}, {np.max(expected_opd_increment):.6f}] mm")
    
    # 比较
    opd_increment_diff = opd_increment - expected_opd_increment
    print(f"  与实际增量的差异 RMS: {np.std(opd_increment_diff):.6f} mm")


# ============================================================================
# 第八部分：绘制诊断图
# ============================================================================

print_section("第八部分：绘制诊断图")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 入射面相位误差
extent = [-grid_sampling.physical_size_mm/2, grid_sampling.physical_size_mm/2,
          -grid_sampling.physical_size_mm/2, grid_sampling.physical_size_mm/2]

phase_diff_masked = np.where(valid_mask, phase_diff, np.nan)
im1 = axes[0, 0].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r')
axes[0, 0].set_title(f'Entrance Phase Error\nRMS={phase_rms_waves:.6f} waves')
axes[0, 0].set_xlabel('X (mm)')
axes[0, 0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0, 0], label='rad')

# 2. 入射光线 OPD vs Pilot Beam OPD
axes[0, 1].scatter(pilot_opd_mm, sampler_opd_mm, s=5, alpha=0.5)
axes[0, 1].plot([np.min(pilot_opd_mm), np.max(pilot_opd_mm)],
                [np.min(pilot_opd_mm), np.max(pilot_opd_mm)], 'r--', label='y=x')
axes[0, 1].set_xlabel('Pilot Beam OPD (mm)')
axes[0, 1].set_ylabel('Sampler OPD (mm)')
axes[0, 1].set_title('Input Ray OPD Comparison')
axes[0, 1].legend()

# 3. 出射光线 OPD vs Pilot Beam OPD
axes[0, 2].scatter(pilot_opd_exit_mm, output_opd_mm, s=5, alpha=0.5)
axes[0, 2].plot([np.min(pilot_opd_exit_mm), np.max(pilot_opd_exit_mm)],
                [np.min(pilot_opd_exit_mm), np.max(pilot_opd_exit_mm)], 'r--', label='y=x')
axes[0, 2].set_xlabel('Pilot Beam OPD (mm)')
axes[0, 2].set_ylabel('Output Ray OPD (mm)')
axes[0, 2].set_title(f'Output Ray OPD Comparison\nResidual RMS={np.std(residual_opd_waves):.6f} waves')
axes[0, 2].legend()

# 4. 残差 OPD 分布
axes[1, 0].hist(residual_opd_waves, bins=50, color='steelblue', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Residual OPD (waves)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Residual OPD Distribution')

# 5. 残差 OPD vs 半径
r = np.sqrt(ray_x**2 + ray_y**2)
axes[1, 1].scatter(r, residual_opd_waves, s=5, alpha=0.5)
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Radius (mm)')
axes[1, 1].set_ylabel('Residual OPD (waves)')
axes[1, 1].set_title('Residual OPD vs Radius')

# 6. OPD 增量 vs 理论值
if not np.isinf(surface_0.radius):
    axes[1, 2].scatter(expected_opd_increment, opd_increment, s=5, alpha=0.5)
    axes[1, 2].plot([np.min(expected_opd_increment), np.max(expected_opd_increment)],
                    [np.min(expected_opd_increment), np.max(expected_opd_increment)], 'r--', label='y=x')
    axes[1, 2].set_xlabel('Expected OPD Increment (mm)')
    axes[1, 2].set_ylabel('Actual OPD Increment (mm)')
    axes[1, 2].set_title('OPD Increment Comparison')
    axes[1, 2].legend()
else:
    axes[1, 2].text(0.5, 0.5, 'Flat Mirror\n(No sag)', ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('OPD Increment (N/A for flat mirror)')

plt.tight_layout()
plt.savefig('debug_surface0_raytracing.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: debug_surface0_raytracing.png")


# ============================================================================
# 第九部分：总结
# ============================================================================

print_section("第九部分：总结")

print(f"""
调试结果总结:

1. 入射面复振幅与 Pilot Beam 误差:
   - 相位 RMS 误差: {phase_rms_waves:.6f} waves
   - 状态: {'OK' if phase_rms_waves < 0.01 else 'WARNING'}

2. 入射光线 OPD 与 Pilot Beam 相位:
   - OPD 差异 RMS: {np.std(opd_diff_waves):.6f} waves
   - 状态: {'OK' if np.std(opd_diff_waves) < 0.01 else 'WARNING'}

3. 出射光线 OPD 与 Pilot Beam 残差:
   - 残差 RMS: {np.std(residual_opd_waves):.6f} waves
   - 状态: {'OK' if np.std(residual_opd_waves) < 0.01 else 'WARNING'}

问题定位:
""")

if phase_rms_waves > 0.01:
    print("  - 入射面复振幅与 Pilot Beam 存在较大误差")
    print("    可能原因: 自由空间传播过程中的误差")

if np.std(opd_diff_waves) > 0.01:
    print("  - 入射光线 OPD 与 Pilot Beam 相位不一致")
    print("    可能原因: WavefrontToRaysSampler 使用了折叠相位")

if np.std(residual_opd_waves) > 0.01:
    print("  - 出射光线 OPD 相对于 Pilot Beam 残差较大")
    print("    可能原因:")
    print("      1. 光线追迹 OPD 计算有误")
    print("      2. Pilot Beam 参数更新不正确")
    print("      3. 入射光线 OPD 设置不正确")

