"""
Surface 3（45度折叠镜）完整误差分析

误差因素记录：
1. 入射面插值误差：~0.000401 waves
   - 原因：从网格插值到光线位置的精度不如 Pilot Beam 解析计算
   - 影响：较小，可接受

继续分析：
2. 光线追迹误差
3. 出射面重采样误差
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
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
from scipy.interpolate import RegularGridInterpolator

print("=" * 80)
print("Surface 3（45度折叠镜）完整误差分析")
print("=" * 80)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=256,
    physical_size_mm=40.0,
)

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=256,
    num_rays=150,
)

# 传播到 Surface 3
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):
    propagator._propagate_to_surface(i)

# 找到入射面和出射面状态
state_entrance = None
state_exit = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_entrance = state
    if state.surface_index == 3 and state.position == 'exit':
        state_exit = state

if state_entrance is None or state_exit is None:
    print("ERROR: 未找到 Surface 3 状态")
    sys.exit(1)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm
grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm

# =============================================================================
# 步骤 1: 入射面分析（已知误差：插值误差 ~0.000401 waves）
# =============================================================================
print("\n" + "=" * 80)
print("【步骤 1】入射面分析")
print("=" * 80)

# 入射面 Pilot Beam 参数
pb_entrance = state_entrance.pilot_beam_params
R_entrance = pb_entrance.curvature_radius_mm
print(f"入射面 Pilot Beam 曲率半径: {R_entrance:.2f} mm")

# 创建采样器
sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)

output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
n_rays = len(ray_x)
print(f"采样光线数量: {n_rays}")

# 光线相位（从 OPD 转换）
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase_entrance = k * ray_opd_mm

# Pilot Beam 在光线位置的相位
r_sq_rays = ray_x**2 + ray_y**2
if np.isinf(R_entrance):
    pilot_phase_entrance_rays = np.zeros_like(r_sq_rays)
else:
    pilot_phase_entrance_rays = k * r_sq_rays / (2 * R_entrance)

# 入射面误差
diff_entrance = ray_phase_entrance - pilot_phase_entrance_rays
rms_entrance = np.std(diff_entrance) / (2 * np.pi)
print(f"入射面光线相位 vs Pilot Beam RMS 误差: {rms_entrance:.6f} waves")
print(f"  （主要来源：网格到光线的插值误差）")

# =============================================================================
# 步骤 2: 光线追迹分析
# =============================================================================
print("\n" + "=" * 80)
print("【步骤 2】光线追迹分析")
print("=" * 80)

# 获取 Surface 3 的元件信息
surface = optical_system[3]
entrance_axis = state_entrance.optical_axis_state

# 直接从 GlobalSurfaceDefinition 获取参数
print(f"元件类型: {surface.surface_type}")
print(f"曲率半径: {surface.radius} mm")
print(f"是否反射: {surface.is_mirror}")

# 从 orientation 矩阵提取倾斜信息
# orientation 的列向量是局部坐标系的 X, Y, Z 轴在全局坐标系中的方向
local_z = surface.orientation[:, 2]  # 局部 Z 轴
# 计算与全局 Z 轴的夹角
global_z = np.array([0, 0, 1])
cos_angle = np.dot(local_z, global_z)
angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
print(f"表面法向与全局 Z 轴夹角: {angle_deg:.1f}°")

# 创建表面定义
# 对于平面镜，需要从 orientation 计算倾斜角度
# 使用 scipy 提取欧拉角
from scipy.spatial.transform import Rotation as R
rotation = R.from_matrix(surface.orientation)
euler_angles = rotation.as_euler('xyz')  # 返回 (rx, ry, rz)
tilt_x = euler_angles[0]
tilt_y = euler_angles[1]

surface_def = SurfaceDefinition(
    surface_type='mirror' if surface.is_mirror else 'refract',
    radius=surface.radius,
    thickness=surface.thickness,
    material='mirror' if surface.is_mirror else 'air',
    semi_aperture=surface.semi_aperture if surface.semi_aperture > 0 else None,
    conic=surface.conic,
    tilt_x=tilt_x,
    tilt_y=tilt_y,
)

# 获取出射光轴方向
exit_axis_state = propagator.get_optical_axis_at_surface(3, 'exit')

raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=0.55,
    chief_ray_direction=tuple(entrance_axis.direction.to_array()),
    entrance_position=tuple(entrance_axis.position.to_array()),
    exit_chief_direction=tuple(exit_axis_state.direction.to_array()),
)

# 执行光线追迹
traced_rays = raytracer.trace(output_rays)

# 出射光线数据
exit_x = np.asarray(traced_rays.x)
exit_y = np.asarray(traced_rays.y)
traced_opd_mm = np.asarray(traced_rays.opd)
traced_phase = k * traced_opd_mm

print(f"\n出射光线位置范围:")
print(f"  x: [{np.min(exit_x):.4f}, {np.max(exit_x):.4f}] mm")
print(f"  y: [{np.min(exit_y):.4f}, {np.max(exit_y):.4f}] mm")
print(f"出射光线 OPD 范围: [{np.min(traced_opd_mm):.6f}, {np.max(traced_opd_mm):.6f}] mm")

# 出射面 Pilot Beam 参数（平面镜不改变曲率）
pb_exit = state_exit.pilot_beam_params
R_exit = pb_exit.curvature_radius_mm
print(f"\n出射面 Pilot Beam 曲率半径: {R_exit:.2f} mm")

# 出射面 Pilot Beam 在光线位置的相位
r_sq_exit = exit_x**2 + exit_y**2
if np.isinf(R_exit):
    pilot_phase_exit_rays = np.zeros_like(r_sq_exit)
else:
    pilot_phase_exit_rays = k * r_sq_exit / (2 * R_exit)

# 光线追迹后的相位误差
diff_traced = traced_phase - pilot_phase_exit_rays
rms_traced = np.std(diff_traced) / (2 * np.pi)
print(f"出射光线相位 vs Pilot Beam RMS 误差: {rms_traced:.6f} waves")

# 分析主光线
distances = np.sqrt(ray_x**2 + ray_y**2)
chief_idx = np.argmin(distances)
print(f"\n主光线分析:")
print(f"  入射位置: ({ray_x[chief_idx]:.6f}, {ray_y[chief_idx]:.6f}) mm")
print(f"  出射位置: ({exit_x[chief_idx]:.6f}, {exit_y[chief_idx]:.6f}) mm")
print(f"  入射 OPD: {ray_opd_mm[chief_idx]:.6f} mm")
print(f"  出射 OPD: {traced_opd_mm[chief_idx]:.6f} mm")
print(f"  OPD 变化: {traced_opd_mm[chief_idx] - ray_opd_mm[chief_idx]:.6f} mm")

# 分析相位变化
phase_change = traced_phase - ray_phase_entrance
print(f"\n相位变化分析:")
print(f"  相位变化范围: [{np.min(phase_change):.6f}, {np.max(phase_change):.6f}] rad")
print(f"  相位变化 RMS: {np.std(phase_change) / (2 * np.pi):.6f} waves")

# 对于平面镜，理论上相位变化应该为 0（主光线处）
# 其他光线的相位变化应该只来自几何路径差
print(f"  主光线相位变化: {phase_change[chief_idx]:.6f} rad")

# =============================================================================
# 步骤 3: 出射面重采样分析
# =============================================================================
print("\n" + "=" * 80)
print("【步骤 3】出射面重采样分析")
print("=" * 80)

# 使用 RayToWavefrontReconstructor 重建波前
reconstructor = RayToWavefrontReconstructor(
    grid_size=grid_size,
    sampling_mm=state_entrance.grid_sampling.sampling_mm,
    wavelength_um=0.55,
)

# 计算 OPD（波长数）
opd_waves = traced_opd_mm / wavelength_mm
valid_mask = np.ones(n_rays, dtype=bool)

# 重建波前
exit_complex = reconstructor.reconstruct(
    ray_x_in=ray_x,
    ray_y_in=ray_y,
    ray_x_out=exit_x,
    ray_y_out=exit_y,
    opd_waves=opd_waves,
    valid_mask=valid_mask,
)

exit_amplitude_recon = np.abs(exit_complex)
exit_phase_recon = np.angle(exit_complex)

# 计算出射面 Pilot Beam 相位网格
pilot_phase_exit_grid = pb_exit.compute_phase_grid(grid_size, physical_size_mm)

# 有效区域
mask_exit = exit_amplitude_recon > 0.01 * np.max(exit_amplitude_recon)
print(f"重建后有效像素数: {np.sum(mask_exit)}")

# 重建相位与 Pilot Beam 的误差
diff_recon = exit_phase_recon - pilot_phase_exit_grid
# 处理相位折叠
diff_recon_wrapped = np.angle(np.exp(1j * diff_recon))
rms_recon = np.std(diff_recon_wrapped[mask_exit]) / (2 * np.pi)

print(f"重建相位范围: [{np.min(exit_phase_recon[mask_exit]):.6f}, {np.max(exit_phase_recon[mask_exit]):.6f}] rad")
print(f"Pilot Beam 相位范围: [{np.min(pilot_phase_exit_grid[mask_exit]):.6f}, {np.max(pilot_phase_exit_grid[mask_exit]):.6f}] rad")
print(f"重建相位 vs Pilot Beam RMS 误差: {rms_recon:.6f} waves")

# =============================================================================
# 步骤 4: 与系统输出的出射面状态比较
# =============================================================================
print("\n" + "=" * 80)
print("【步骤 4】与系统输出的出射面状态比较")
print("=" * 80)

# 系统输出的出射面相位
sys_exit_phase = state_exit.phase
sys_exit_amplitude = state_exit.amplitude

# 有效区域
mask_sys = sys_exit_amplitude > 0.01 * np.max(sys_exit_amplitude)
print(f"系统输出有效像素数: {np.sum(mask_sys)}")

# 系统输出相位与 Pilot Beam 的误差
diff_sys = sys_exit_phase - pilot_phase_exit_grid
diff_sys_wrapped = np.angle(np.exp(1j * diff_sys))
rms_sys = np.std(diff_sys_wrapped[mask_sys]) / (2 * np.pi)

print(f"系统输出相位范围: [{np.min(sys_exit_phase[mask_sys]):.6f}, {np.max(sys_exit_phase[mask_sys]):.6f}] rad")
print(f"系统输出相位 vs Pilot Beam RMS 误差: {rms_sys:.6f} waves")

# 比较重建相位与系统输出相位
# 找到两者都有效的区域
mask_both = mask_exit & mask_sys
if np.sum(mask_both) > 0:
    diff_recon_sys = exit_phase_recon - sys_exit_phase
    diff_recon_sys_wrapped = np.angle(np.exp(1j * diff_recon_sys))
    rms_recon_sys = np.std(diff_recon_sys_wrapped[mask_both]) / (2 * np.pi)
    print(f"重建相位 vs 系统输出相位 RMS 误差: {rms_recon_sys:.6f} waves")

# =============================================================================
# 误差汇总
# =============================================================================
print("\n" + "=" * 80)
print("误差汇总")
print("=" * 80)

print("""
误差因素记录：

1. 入射面插值误差: {:.6f} waves
   - 原因：从网格插值到光线位置的精度不如 Pilot Beam 解析计算
   - 这是采样过程固有的误差

2. 光线追迹后误差: {:.6f} waves
   - 原因：光线追迹过程中的 OPD 计算
   - 包含入射面插值误差的传递

3. 出射面重采样误差: {:.6f} waves
   - 原因：从稀疏光线重采样到网格
   - 包含前面所有误差的累积

4. 系统输出误差: {:.6f} waves
   - 这是最终的端到端误差
""".format(rms_entrance, rms_traced, rms_recon, rms_sys))

# 计算各步骤引入的增量误差
print("各步骤引入的增量误差:")
print(f"  步骤 1 (入射面插值): {rms_entrance:.6f} waves")
print(f"  步骤 2 (光线追迹): {rms_traced - rms_entrance:.6f} waves (增量)")
print(f"  步骤 3 (出射面重采样): {rms_recon - rms_traced:.6f} waves (增量)")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
