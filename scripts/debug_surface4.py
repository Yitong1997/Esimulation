"""
调试 Surface 4 (M1) 处振幅变为零的问题

详细追踪第一个反射镜处的传播过程。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    load_optical_system_from_zmx,
)
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
from hybrid_optical_propagation.free_space_propagator import FreeSpacePropagator, compute_propagation_distance
from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.material_detection import is_coordinate_break, detect_material_change
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
for i, surface in enumerate(optical_system):
    mirror_str = "反射镜" if surface.is_mirror else "透射面"
    print(f"  [{i}] 表面 {surface.index}: {surface.surface_type}, "
          f"R={surface.radius:.2f}mm, {mirror_str}, "
          f"位置={surface.vertex_position}")


# ============================================================================
# 创建光源和初始波前
# ============================================================================

print_section("创建光源和初始波前")

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

amplitude, phase, pilot_beam_params, proper_wfo = source.create_initial_wavefront()
grid_sampling = source.get_grid_sampling()

print(f"初始振幅范围: [{np.min(amplitude):.6f}, {np.max(amplitude):.6f}]")
print(f"初始相位范围: [{np.min(phase):.6f}, {np.max(phase):.6f}]")


# ============================================================================
# 创建传播器
# ============================================================================

print_section("创建传播器")

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=wavelength_um,
    grid_size=grid_size,
    num_rays=num_rays,
)

# 初始化传播状态
from sequential_system.coordinate_tracking import (
    OpticalAxisState,
    Position3D,
    RayDirection,
)
from hybrid_optical_propagation.data_models import PropagationState

initial_axis_state = OpticalAxisState(
    position=Position3D(0.0, 0.0, 0.0),
    direction=RayDirection(0.0, 0.0, 1.0),
    path_length=0.0,
)

initial_state = PropagationState(
    surface_index=-1,
    position='source',
    amplitude=amplitude,
    phase=phase,
    pilot_beam_params=pilot_beam_params,
    proper_wfo=proper_wfo,
    optical_axis_state=initial_axis_state,
    grid_sampling=grid_sampling,
)


# ============================================================================
# 逐步传播到 Surface 4
# ============================================================================

print_section("逐步传播到 Surface 4")

current_state = initial_state
free_space_propagator = FreeSpacePropagator(wavelength_um)
hybrid_element_propagator = HybridElementPropagator(
    wavelength_um=wavelength_um,
    num_rays=num_rays,
    method="local_raytracing",
)

for i, surface in enumerate(optical_system[:4]):  # 只处理前 4 个表面
    print(f"\n--- 处理表面 [{i}] (Surface {surface.index}) ---")
    print(f"  类型: {surface.surface_type}")
    print(f"  是否反射镜: {surface.is_mirror}")
    print(f"  位置: {surface.vertex_position}")
    
    # 检查是否是坐标断点
    if is_coordinate_break(surface):
        print("  [跳过] 坐标断点")
        continue
    
    # 获取光轴状态
    entrance_axis = propagator.get_optical_axis_at_surface(i, 'entrance')
    exit_axis = propagator.get_optical_axis_at_surface(i, 'exit')
    
    print(f"  入射光轴位置: {entrance_axis.position.to_array()}")
    print(f"  入射光轴方向: {entrance_axis.direction.to_array()}")
    print(f"  出射光轴方向: {exit_axis.direction.to_array()}")
    
    # 1. 自由空间传播到入射面
    current_pos = current_state.optical_axis_state.position.to_array()
    target_pos = entrance_axis.position.to_array()
    current_dir = current_state.optical_axis_state.direction.to_array()
    
    distance = compute_propagation_distance(current_pos, target_pos, current_dir)
    print(f"  传播距离: {distance:.2f} mm")
    
    if abs(distance) > 1e-10:
        print("  执行自由空间传播...")
        current_state = free_space_propagator.propagate(
            current_state,
            entrance_axis,
            i,
            'entrance',
        )
        print(f"  入射面振幅范围: [{np.min(current_state.amplitude):.6f}, {np.max(current_state.amplitude):.6f}]")
        print(f"  入射面相位范围: [{np.min(current_state.phase):.6f}, {np.max(current_state.phase):.6f}]")
    
    # 2. 检查是否需要混合元件传播
    prev_surface = optical_system[i-1] if i > 0 else None
    needs_hybrid = detect_material_change(surface, prev_surface)
    
    print(f"  需要混合元件传播: {needs_hybrid}")
    
    if needs_hybrid:
        print("  执行混合元件传播...")
        
        # 详细调试混合元件传播
        print(f"\n  === 混合元件传播详细调试 ===")
        
        # 采样光线
        print(f"  采样光线...")
        sampler = WavefrontToRaysSampler(
            amplitude=current_state.amplitude,
            phase=current_state.phase,
            physical_size=current_state.grid_sampling.physical_size_mm,
            wavelength=wavelength_um,
            num_rays=num_rays,
            distribution="hexapolar",
        )
        
        input_rays = sampler.get_output_rays()
        ray_x = np.asarray(input_rays.x)
        ray_y = np.asarray(input_rays.y)
        
        print(f"  采样光线数量: {len(ray_x)}")
        print(f"  光线 x 范围: [{np.min(ray_x):.4f}, {np.max(ray_x):.4f}] mm")
        print(f"  光线 y 范围: [{np.min(ray_y):.4f}, {np.max(ray_y):.4f}] mm")
        
        # 检查光线方向
        L = np.asarray(input_rays.L)
        M = np.asarray(input_rays.M)
        N = np.asarray(input_rays.N)
        print(f"  光线 L 范围: [{np.min(L):.6f}, {np.max(L):.6f}]")
        print(f"  光线 M 范围: [{np.min(M):.6f}, {np.max(M):.6f}]")
        print(f"  光线 N 范围: [{np.min(N):.6f}, {np.max(N):.6f}]")
        
        # 创建表面定义
        surface_def = SurfaceDefinition(
            surface_type='mirror' if surface.is_mirror else 'refract',
            radius=surface.radius,
            thickness=surface.thickness,
            material='mirror' if surface.is_mirror else surface.material,
            semi_aperture=surface.semi_aperture,
            conic=surface.conic,
        )
        
        print(f"\n  表面定义:")
        print(f"    类型: {surface_def.surface_type}")
        print(f"    曲率半径: {surface_def.radius}")
        print(f"    厚度: {surface_def.thickness}")
        print(f"    半口径: {surface_def.semi_aperture}")
        
        # 创建光线追迹器
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=wavelength_um,
            chief_ray_direction=tuple(entrance_axis.direction.to_array()),
            entrance_position=tuple(entrance_axis.position.to_array()),
        )
        
        print(f"\n  光线追迹器:")
        print(f"    主光线方向: {raytracer.chief_ray_direction}")
        print(f"    入射位置: {raytracer.entrance_position}")
        
        # 执行光线追迹
        print(f"\n  执行光线追迹...")
        output_rays = raytracer.trace(input_rays)
        
        ray_x_out = np.asarray(output_rays.x)
        ray_y_out = np.asarray(output_rays.y)
        ray_z_out = np.asarray(output_rays.z)
        
        print(f"  出射光线 x 范围: [{np.min(ray_x_out):.4f}, {np.max(ray_x_out):.4f}] mm")
        print(f"  出射光线 y 范围: [{np.min(ray_y_out):.4f}, {np.max(ray_y_out):.4f}] mm")
        print(f"  出射光线 z 范围: [{np.min(ray_z_out):.4f}, {np.max(ray_z_out):.4f}] mm")
        
        # 检查出射光线方向
        L_out = np.asarray(output_rays.L)
        M_out = np.asarray(output_rays.M)
        N_out = np.asarray(output_rays.N)
        print(f"  出射光线 L 范围: [{np.min(L_out):.6f}, {np.max(L_out):.6f}]")
        print(f"  出射光线 M 范围: [{np.min(M_out):.6f}, {np.max(M_out):.6f}]")
        print(f"  出射光线 N 范围: [{np.min(N_out):.6f}, {np.max(N_out):.6f}]")
        
        # 检查 OPD
        opd_out = np.asarray(output_rays.opd)
        print(f"  出射光线 OPD 范围: [{np.min(opd_out):.6f}, {np.max(opd_out):.6f}] mm")
        
        # 检查网格范围
        grid_half_size = current_state.grid_sampling.physical_size_mm / 2
        print(f"\n  网格范围: [{-grid_half_size:.4f}, {grid_half_size:.4f}] mm")
        
        in_range_x = (ray_x_out >= -grid_half_size) & (ray_x_out <= grid_half_size)
        in_range_y = (ray_y_out >= -grid_half_size) & (ray_y_out <= grid_half_size)
        in_range = in_range_x & in_range_y
        
        print(f"  在范围内的光线数量: {np.sum(in_range)} / {len(ray_x_out)}")
        
        if np.sum(in_range) == 0:
            print("\n  [ERROR] 所有出射光线都在网格范围外！")
            print(f"  出射光线中心: ({np.mean(ray_x_out):.4f}, {np.mean(ray_y_out):.4f})")
            print(f"  期望的出射光轴位置: {exit_axis.position.to_array()}")
        
        # 执行完整的混合元件传播
        new_state = hybrid_element_propagator.propagate(
            current_state,
            surface,
            entrance_axis,
            exit_axis,
            i,
        )
        
        current_state = new_state
        print(f"\n  出射面振幅范围: [{np.min(current_state.amplitude):.6f}, {np.max(current_state.amplitude):.6f}]")
        print(f"  出射面相位范围: [{np.min(current_state.phase):.6f}, {np.max(current_state.phase):.6f}]")
    else:
        # 只更新光轴状态
        current_state = PropagationState(
            surface_index=i,
            position='exit',
            amplitude=current_state.amplitude.copy(),
            phase=current_state.phase.copy(),
            pilot_beam_params=current_state.pilot_beam_params,
            proper_wfo=current_state.proper_wfo,
            optical_axis_state=exit_axis,
            grid_sampling=current_state.grid_sampling,
        )


print_section("调试完成")
