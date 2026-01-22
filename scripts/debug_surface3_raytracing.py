"""
详细调试表面 3 的光线追迹和重建过程

检查：
1. 入射光线的 OPD
2. 出射光线的 OPD
3. 重建后的相位
"""

import sys
from pathlib import Path

# 添加项目路径
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
from hybrid_optical_propagation.material_detection import is_coordinate_break
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition, compute_rotation_matrix
from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor


def main():
    """主函数"""
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    print("加载光学系统...")
    optical_system = load_optical_system_from_zmx(zmx_file)
    
    # 创建光源定义
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
    
    # 传播到表面 3 入射面
    print("\n传播到表面 3 入射面...")
    
    # 初始化
    propagator._current_state = propagator._initialize_propagation()
    propagator._surface_states = [propagator._current_state]
    
    # 传播到表面 2
    for i in range(3):
        surface = optical_system[i]
        if is_coordinate_break(surface):
            continue
        propagator._propagate_to_surface(i)
    
    # 传播到表面 3 入射面
    propagator._propagate_to_surface(3)
    
    # 找到表面 3 入射面状态
    state_entrance = None
    for state in propagator._surface_states:
        if state.surface_index == 3 and state.position == 'entrance':
            state_entrance = state
            break
    
    if state_entrance is None:
        print("未找到表面 3 入射面状态")
        return
    
    print(f"\n表面 3 入射面状态:")
    print(f"  振幅范围: [{np.min(state_entrance.amplitude):.6f}, {np.max(state_entrance.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(state_entrance.phase):.6f}, {np.max(state_entrance.phase):.6f}]")
    print(f"  Pilot Beam 曲率半径: {state_entrance.pilot_beam_params.curvature_radius_mm:.2f} mm")
    
    # 手动执行光线追迹
    print("\n" + "=" * 60)
    print("手动执行光线追迹")
    print("=" * 60)
    
    # 1. 从波前采样光线
    sampler = WavefrontToRaysSampler(
        amplitude=state_entrance.amplitude,
        phase=state_entrance.phase,
        physical_size=state_entrance.grid_sampling.physical_size_mm,
        wavelength=0.55,
        num_rays=150,
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    ray_opd_waves = sampler.get_ray_opd()
    
    ray_x = np.asarray(input_rays.x)
    ray_y = np.asarray(input_rays.y)
    
    print(f"\n输入光线:")
    print(f"  数量: {len(ray_x)}")
    print(f"  X 范围: [{np.min(ray_x):.4f}, {np.max(ray_x):.4f}]")
    print(f"  Y 范围: [{np.min(ray_y):.4f}, {np.max(ray_y):.4f}]")
    print(f"  OPD 范围 (waves): [{np.min(ray_opd_waves):.6f}, {np.max(ray_opd_waves):.6f}]")
    
    # 设置光线 OPD
    wavelength_mm = 0.55 * 1e-3
    input_rays.opd = ray_opd_waves * wavelength_mm
    
    # 2. 获取光轴状态
    entrance_axis = propagator.get_optical_axis_at_surface(3, 'entrance')
    exit_axis = propagator.get_optical_axis_at_surface(3, 'exit')
    
    print(f"\n入射光轴:")
    print(f"  位置: {entrance_axis.position.to_array()}")
    print(f"  方向: {entrance_axis.direction.to_array()}")
    
    print(f"\n出射光轴:")
    print(f"  位置: {exit_axis.position.to_array()}")
    print(f"  方向: {exit_axis.direction.to_array()}")
    
    # 3. 创建表面定义
    surface_3 = optical_system[3]
    
    # 计算表面在入射面局部坐标系中的倾斜角度
    entrance_dir = entrance_axis.direction.to_array()
    R_entrance = compute_rotation_matrix(tuple(entrance_dir))
    
    surface_normal_global = surface_3.surface_normal
    surface_normal_local = R_entrance.T @ surface_normal_global
    
    nx, ny, nz = surface_normal_local
    tilt_x = np.arcsin(np.clip(ny, -1, 1))
    cos_rx = np.cos(tilt_x)
    if abs(cos_rx) > 1e-10:
        sin_ry = -nx / cos_rx
        tilt_y = np.arcsin(np.clip(sin_ry, -1, 1))
    else:
        tilt_y = 0.0
    
    print(f"\n表面倾斜角度（入射面局部坐标系）:")
    print(f"  tilt_x: {np.degrees(tilt_x):.2f}°")
    print(f"  tilt_y: {np.degrees(tilt_y):.2f}°")
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=surface_3.radius,
        thickness=surface_3.thickness,
        material='mirror',
        semi_aperture=surface_3.semi_aperture,
        conic=surface_3.conic,
        tilt_x=tilt_x,
        tilt_y=tilt_y,
    )
    
    # 4. 执行光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=0.55,
        chief_ray_direction=tuple(entrance_axis.direction.to_array()),
        entrance_position=tuple(entrance_axis.position.to_array()),
        exit_chief_direction=tuple(exit_axis.direction.to_array()),
    )
    
    output_rays = raytracer.trace(input_rays)
    
    out_x = np.asarray(output_rays.x)
    out_y = np.asarray(output_rays.y)
    out_opd = np.asarray(output_rays.opd)
    
    print(f"\n输出光线:")
    print(f"  X 范围: [{np.min(out_x):.4f}, {np.max(out_x):.4f}]")
    print(f"  Y 范围: [{np.min(out_y):.4f}, {np.max(out_y):.4f}]")
    print(f"  OPD 范围 (mm): [{np.min(out_opd):.6f}, {np.max(out_opd):.6f}]")
    
    # 5. 计算 OPD 变化
    input_opd = np.asarray(input_rays.opd)
    opd_change = out_opd - input_opd
    
    print(f"\nOPD 变化:")
    print(f"  范围 (mm): [{np.min(opd_change):.6f}, {np.max(opd_change):.6f}]")
    print(f"  范围 (waves): [{np.min(opd_change)/wavelength_mm:.6f}, {np.max(opd_change)/wavelength_mm:.6f}]")
    print(f"  RMS (waves): {np.std(opd_change)/wavelength_mm:.6f}")

    # 对于平面镜，OPD 变化应该为 0
    # 检查主光线
    center_idx = np.argmin(ray_x**2 + ray_y**2)
    print(f"\n主光线（索引 {center_idx}）:")
    print(f"  入射位置: ({ray_x[center_idx]:.4f}, {ray_y[center_idx]:.4f})")
    print(f"  出射位置: ({out_x[center_idx]:.4f}, {out_y[center_idx]:.4f})")
    print(f"  入射 OPD: {input_opd[center_idx]:.6f} mm")
    print(f"  出射 OPD: {out_opd[center_idx]:.6f} mm")
    print(f"  OPD 变化: {opd_change[center_idx]:.6f} mm")
    
    # 6. 重建波前
    print("\n" + "=" * 60)
    print("波前重建")
    print("=" * 60)
    
    reconstructor = RayToWavefrontReconstructor(
        grid_size=state_entrance.grid_sampling.grid_size,
        sampling_mm=state_entrance.grid_sampling.sampling_mm,
        wavelength_um=0.55,
    )
    
    opd_waves = out_opd / wavelength_mm
    valid_mask = np.ones(len(ray_x), dtype=bool)
    
    exit_complex = reconstructor.reconstruct(
        ray_x_in=ray_x,
        ray_y_in=ray_y,
        ray_x_out=out_x,
        ray_y_out=out_y,
        opd_waves=opd_waves,
        valid_mask=valid_mask,
    )
    
    exit_amplitude = np.abs(exit_complex)
    exit_phase = np.angle(exit_complex)
    
    print(f"\n重建结果:")
    print(f"  振幅范围: [{np.min(exit_amplitude):.6f}, {np.max(exit_amplitude):.6f}]")
    print(f"  相位范围: [{np.min(exit_phase):.6f}, {np.max(exit_phase):.6f}]")
    
    # 7. 计算相位误差
    grid_size = state_entrance.grid_sampling.grid_size
    sampling_mm = state_entrance.grid_sampling.sampling_mm
    x = np.arange(grid_size) - grid_size // 2
    y = np.arange(grid_size) - grid_size // 2
    xx, yy = np.meshgrid(x * sampling_mm, y * sampling_mm)
    r_sq = xx**2 + yy**2
    
    # 出射面 Pilot Beam 相位（与入射面相同，因为平面镜不改变曲率）
    R = state_entrance.pilot_beam_params.curvature_radius_mm
    k = 2 * np.pi / wavelength_mm
    
    if np.isinf(R):
        pilot_phase = np.zeros_like(r_sq)
    else:
        pilot_phase = k * r_sq / (2 * R)
    
    phase_diff = exit_phase - pilot_phase
    
    mask = exit_amplitude > 0.01
    if np.any(mask):
        phase_diff_masked = phase_diff[mask]
        phase_rms_waves = np.std(phase_diff_masked) / (2 * np.pi)
        phase_pv_waves = (np.max(phase_diff_masked) - np.min(phase_diff_masked)) / (2 * np.pi)
        
        print(f"\n相位误差（相对于 Pilot Beam）:")
        print(f"  RMS: {phase_rms_waves:.6f} waves")
        print(f"  PV: {phase_pv_waves:.6f} waves")
    
    # 8. 检查入射相位与 Pilot Beam 的差异
    print("\n" + "=" * 60)
    print("入射相位分析")
    print("=" * 60)
    
    # 入射相位
    entrance_phase = state_entrance.phase
    entrance_amplitude = state_entrance.amplitude
    
    # 入射 Pilot Beam 相位
    R_entrance = state_entrance.pilot_beam_params.curvature_radius_mm
    if np.isinf(R_entrance):
        pilot_phase_entrance = np.zeros_like(r_sq)
    else:
        pilot_phase_entrance = k * r_sq / (2 * R_entrance)
    
    phase_diff_entrance = entrance_phase - pilot_phase_entrance
    
    mask_entrance = entrance_amplitude > 0.01
    if np.any(mask_entrance):
        phase_diff_entrance_masked = phase_diff_entrance[mask_entrance]
        phase_rms_entrance = np.std(phase_diff_entrance_masked) / (2 * np.pi)
        
        print(f"入射相位与 Pilot Beam 的差异:")
        print(f"  RMS: {phase_rms_entrance:.6f} waves")
        print(f"  范围: [{np.min(phase_diff_entrance_masked):.6f}, {np.max(phase_diff_entrance_masked):.6f}] rad")
    
    # 9. 检查光线采样的 OPD 与 Pilot Beam 的关系
    print("\n" + "=" * 60)
    print("光线 OPD 与 Pilot Beam 的关系")
    print("=" * 60)
    
    # 计算光线位置处的 Pilot Beam OPD
    ray_r_sq = ray_x**2 + ray_y**2
    if np.isinf(R_entrance):
        pilot_opd_at_rays = np.zeros_like(ray_r_sq)
    else:
        pilot_opd_at_rays = ray_r_sq / (2 * R_entrance)  # mm
    
    pilot_opd_waves_at_rays = pilot_opd_at_rays / wavelength_mm
    
    # 比较
    opd_diff = ray_opd_waves - pilot_opd_waves_at_rays
    
    print(f"光线 OPD (waves): [{np.min(ray_opd_waves):.6f}, {np.max(ray_opd_waves):.6f}]")
    print(f"Pilot Beam OPD at rays (waves): [{np.min(pilot_opd_waves_at_rays):.6f}, {np.max(pilot_opd_waves_at_rays):.6f}]")
    print(f"差异 (waves): [{np.min(opd_diff):.6f}, {np.max(opd_diff):.6f}]")
    print(f"差异 RMS (waves): {np.std(opd_diff):.6f}")
    
    print("\n完成")


if __name__ == '__main__':
    main()
