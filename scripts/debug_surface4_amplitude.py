"""
调试表面 3 和表面 4 的误差来源

分析为什么从 Surface_3 开始出现误差
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

from optiland.rays import RealRays

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.material_detection import is_coordinate_break


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
    
    # 初始化
    propagator._current_state = propagator._initialize_propagation()
    propagator._surface_states = [propagator._current_state]
    
    # 传播到表面 3 入射面
    print("\n=== 传播到表面 3 入射面 ===")
    for i in range(3):
        surface = optical_system[i]
        if is_coordinate_break(surface):
            continue
        propagator._propagate_to_surface(i)
    
    # 获取表面 3 入射面状态
    state_s3_entrance = propagator._current_state
    print(f"表面 3 入射面状态:")
    print(f"  振幅范围: [{np.min(state_s3_entrance.amplitude):.6f}, {np.max(state_s3_entrance.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(state_s3_entrance.phase):.6f}, {np.max(state_s3_entrance.phase):.6f}]")
    print(f"  Pilot Beam 曲率半径: {state_s3_entrance.pilot_beam_params.curvature_radius_mm:.2f} mm")
    
    # 计算 Pilot Beam 相位
    grid_size = state_s3_entrance.grid_sampling.grid_size
    sampling_mm = state_s3_entrance.grid_sampling.sampling_mm
    x = np.arange(grid_size) - grid_size // 2
    y = np.arange(grid_size) - grid_size // 2
    xx, yy = np.meshgrid(x * sampling_mm, y * sampling_mm)
    r_sq = xx**2 + yy**2
    
    R = state_s3_entrance.pilot_beam_params.curvature_radius_mm
    wavelength_mm = 0.55 * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    if np.isinf(R):
        pilot_phase = np.zeros_like(r_sq)
    else:
        pilot_phase = k * r_sq / (2 * R)
    
    # 计算相位误差
    phase_diff = state_s3_entrance.phase - pilot_phase
    
    # 只在振幅非零区域计算
    mask = state_s3_entrance.amplitude > 0.01
    if np.any(mask):
        phase_diff_masked = phase_diff[mask]
        phase_rms_rad = np.std(phase_diff_masked)
        phase_rms_waves = phase_rms_rad / (2 * np.pi)
        print(f"  相位误差 RMS: {phase_rms_waves:.6f} waves")
        print(f"  相位误差 PV: {(np.max(phase_diff_masked) - np.min(phase_diff_masked)) / (2 * np.pi):.6f} waves")
    
    # 传播到表面 3 出射面
    print("\n=== 传播到表面 3 出射面 ===")
    propagator._propagate_to_surface(3)
    
    state_s3_exit = propagator._current_state
    print(f"表面 3 出射面状态:")
    print(f"  振幅范围: [{np.min(state_s3_exit.amplitude):.6f}, {np.max(state_s3_exit.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(state_s3_exit.phase):.6f}, {np.max(state_s3_exit.phase):.6f}]")
    print(f"  Pilot Beam 曲率半径: {state_s3_exit.pilot_beam_params.curvature_radius_mm:.2f} mm")
    
    # 计算出射面 Pilot Beam 相位
    R_exit = state_s3_exit.pilot_beam_params.curvature_radius_mm
    if np.isinf(R_exit):
        pilot_phase_exit = np.zeros_like(r_sq)
    else:
        pilot_phase_exit = k * r_sq / (2 * R_exit)
    
    # 计算相位误差
    phase_diff_exit = state_s3_exit.phase - pilot_phase_exit
    
    mask_exit = state_s3_exit.amplitude > 0.01
    if np.any(mask_exit):
        phase_diff_exit_masked = phase_diff_exit[mask_exit]
        phase_rms_exit_rad = np.std(phase_diff_exit_masked)
        phase_rms_exit_waves = phase_rms_exit_rad / (2 * np.pi)
        print(f"  相位误差 RMS: {phase_rms_exit_waves:.6f} waves")
        print(f"  相位误差 PV: {(np.max(phase_diff_exit_masked) - np.min(phase_diff_exit_masked)) / (2 * np.pi):.6f} waves")
    
    # 检查表面 3 的信息
    surface_3 = optical_system[3]
    print(f"\n表面 3 信息:")
    print(f"  类型: {surface_3.surface_type}")
    print(f"  是否反射镜: {surface_3.is_mirror}")
    print(f"  曲率半径: {surface_3.radius}")
    print(f"  厚度: {surface_3.thickness}")
    print(f"  顶点位置: {surface_3.vertex_position}")
    print(f"  表面法向量: {surface_3.surface_normal}")
    
    # 获取光轴状态
    entrance_axis = propagator.get_optical_axis_at_surface(3, 'entrance')
    exit_axis = propagator.get_optical_axis_at_surface(3, 'exit')
    
    print(f"\n入射光轴:")
    print(f"  位置: {entrance_axis.position.to_array()}")
    print(f"  方向: {entrance_axis.direction.to_array()}")
    
    print(f"\n出射光轴:")
    print(f"  位置: {exit_axis.position.to_array()}")
    print(f"  方向: {exit_axis.direction.to_array()}")
    
    # 手动执行光线追迹，检查 OPD
    print("\n=== 手动光线追迹分析 ===")
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition, compute_rotation_matrix
    
    sampler = WavefrontToRaysSampler(
        amplitude=state_s3_entrance.amplitude,
        phase=state_s3_entrance.phase,
        physical_size=state_s3_entrance.grid_sampling.physical_size_mm,
        wavelength=0.55,
        num_rays=150,
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    ray_x = np.asarray(input_rays.x)
    ray_y = np.asarray(input_rays.y)
    input_opd = np.asarray(input_rays.opd)
    
    print(f"输入光线:")
    print(f"  数量: {len(ray_x)}")
    print(f"  X 范围: [{np.min(ray_x):.4f}, {np.max(ray_x):.4f}]")
    print(f"  Y 范围: [{np.min(ray_y):.4f}, {np.max(ray_y):.4f}]")
    print(f"  OPD 范围: [{np.min(input_opd):.6f}, {np.max(input_opd):.6f}] mm")
    
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
    print(f"  OPD 范围: [{np.min(out_opd):.6f}, {np.max(out_opd):.6f}] mm")
    print(f"  OPD 变化: [{np.min(out_opd - input_opd):.6f}, {np.max(out_opd - input_opd):.6f}] mm")
    
    # 对于平面镜，OPD 变化应该为 0（相对于主光线）
    # 检查主光线的 OPD
    # 主光线是 x=0, y=0 的光线
    center_idx = np.argmin(ray_x**2 + ray_y**2)
    print(f"\n主光线（索引 {center_idx}）:")
    print(f"  位置: ({ray_x[center_idx]:.4f}, {ray_y[center_idx]:.4f})")
    print(f"  输入 OPD: {input_opd[center_idx]:.6f} mm")
    print(f"  输出 OPD: {out_opd[center_idx]:.6f} mm")
    print(f"  OPD 变化: {out_opd[center_idx] - input_opd[center_idx]:.6f} mm")
    
    # 计算相对于主光线的 OPD
    relative_opd = out_opd - out_opd[center_idx]
    print(f"\n相对于主光线的 OPD:")
    print(f"  范围: [{np.min(relative_opd):.6f}, {np.max(relative_opd):.6f}] mm")
    print(f"  RMS: {np.std(relative_opd):.6f} mm")
    print(f"  RMS (waves): {np.std(relative_opd) / wavelength_mm:.6f} waves")
    
    # 检查 Pilot Beam 参数变化
    print("\n=== Pilot Beam 参数分析 ===")
    print(f"入射面 Pilot Beam:")
    print(f"  曲率半径: {state_s3_entrance.pilot_beam_params.curvature_radius_mm:.2f} mm")
    print(f"  束腰半径: {state_s3_entrance.pilot_beam_params.waist_radius_mm:.4f} mm")
    print(f"  束腰位置: {state_s3_entrance.pilot_beam_params.waist_position_mm:.4f} mm")
    print(f"  光斑大小: {state_s3_entrance.pilot_beam_params.spot_size_mm:.4f} mm")
    
    print(f"\n出射面 Pilot Beam:")
    print(f"  曲率半径: {state_s3_exit.pilot_beam_params.curvature_radius_mm:.2f} mm")
    print(f"  束腰半径: {state_s3_exit.pilot_beam_params.waist_radius_mm:.4f} mm")
    print(f"  束腰位置: {state_s3_exit.pilot_beam_params.waist_position_mm:.4f} mm")
    print(f"  光斑大小: {state_s3_exit.pilot_beam_params.spot_size_mm:.4f} mm")
    
    # 对于平面镜，Pilot Beam 参数应该不变（只是坐标系旋转）
    # 检查曲率半径是否正确
    # 入射面和出射面的曲率半径应该相同（因为平面镜不改变波前曲率）
    print(f"\n曲率半径变化: {state_s3_exit.pilot_beam_params.curvature_radius_mm - state_s3_entrance.pilot_beam_params.curvature_radius_mm:.2f} mm")
    
    # 检查重建过程
    print("\n=== 重建过程分析 ===")
    from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
    
    reconstructor = RayToWavefrontReconstructor(
        grid_size=state_s3_entrance.grid_sampling.grid_size,
        sampling_mm=state_s3_entrance.grid_sampling.sampling_mm,
        wavelength_um=0.55,
    )
    
    # 计算 OPD（波长数）
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
    
    print(f"重建结果:")
    print(f"  振幅范围: [{np.min(exit_amplitude):.6f}, {np.max(exit_amplitude):.6f}]")
    print(f"  相位范围: [{np.min(exit_phase):.6f}, {np.max(exit_phase):.6f}]")
    
    # 计算重建相位与 Pilot Beam 相位的差异
    R_exit = state_s3_exit.pilot_beam_params.curvature_radius_mm
    if np.isinf(R_exit):
        pilot_phase_exit = np.zeros_like(r_sq)
    else:
        pilot_phase_exit = k * r_sq / (2 * R_exit)
    
    phase_diff_recon = exit_phase - pilot_phase_exit
    mask_recon = exit_amplitude > 0.01
    if np.any(mask_recon):
        phase_diff_recon_masked = phase_diff_recon[mask_recon]
        print(f"  相位误差 RMS: {np.std(phase_diff_recon_masked) / (2 * np.pi):.6f} waves")
        print(f"  相位误差 PV: {(np.max(phase_diff_recon_masked) - np.min(phase_diff_recon_masked)) / (2 * np.pi):.6f} waves")
    
    print("\n完成")
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
    
    # 初始化
    propagator._current_state = propagator._initialize_propagation()
    propagator._surface_states = [propagator._current_state]
    
    # 传播到表面 3（表面 4 之前）
    for i in range(4):
        surface = optical_system[i]
        if is_coordinate_break(surface):
            continue
        propagator._propagate_to_surface(i)
    
    print(f"\n表面 3 之后的状态:")
    state_before = propagator._current_state
    print(f"  振幅范围: [{np.min(state_before.amplitude):.6f}, {np.max(state_before.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(state_before.phase):.6f}, {np.max(state_before.phase):.6f}]")
    print(f"  振幅非零像素数: {np.sum(state_before.amplitude > 0.01)}")
    
if __name__ == '__main__':
    main()
