"""
调试 Pilot Beam 参数追踪

检查：
1. 自由空间传播前后 Pilot Beam 参数的变化
2. 平面镜处 Pilot Beam 参数是否正确保持不变
3. 波前 RMS 误差与采样精度的关系
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
from hybrid_optical_propagation.data_models import PilotBeamParams
from hybrid_optical_propagation.material_detection import is_coordinate_break


def test_pilot_beam_free_space():
    """测试自由空间传播时 Pilot Beam 参数的变化"""
    print("=" * 60)
    print("测试 1: 自由空间传播时 Pilot Beam 参数变化")
    print("=" * 60)
    
    # 创建初始 Pilot Beam
    wavelength_um = 0.55
    w0_mm = 5.0
    z0_mm = 0.0  # 束腰在当前位置
    
    pilot = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    print(f"\n初始 Pilot Beam:")
    print(f"  波长: {pilot.wavelength_um} μm")
    print(f"  束腰半径: {pilot.waist_radius_mm} mm")
    print(f"  束腰位置: {pilot.waist_position_mm} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm} mm")
    print(f"  光斑大小: {pilot.spot_size_mm} mm")
    print(f"  瑞利长度: {pilot.rayleigh_length_mm} mm")
    print(f"  q 参数: {pilot.q_parameter}")
    
    # 传播 10mm
    distance = 10.0
    pilot_after = pilot.propagate(distance)
    
    print(f"\n传播 {distance} mm 后:")
    print(f"  束腰半径: {pilot_after.waist_radius_mm} mm")
    print(f"  束腰位置: {pilot_after.waist_position_mm} mm")
    print(f"  曲率半径: {pilot_after.curvature_radius_mm} mm")
    print(f"  光斑大小: {pilot_after.spot_size_mm} mm")
    print(f"  q 参数: {pilot_after.q_parameter}")
    
    # 验证：束腰位置应该变化 -distance
    expected_waist_pos = z0_mm - distance
    print(f"\n验证:")
    print(f"  期望束腰位置: {expected_waist_pos} mm")
    print(f"  实际束腰位置: {pilot_after.waist_position_mm} mm")
    print(f"  差异: {abs(pilot_after.waist_position_mm - expected_waist_pos)} mm")
    
    # 传播 40mm（到表面 3 的位置）
    distance = 40.0
    pilot_40 = pilot.propagate(distance)
    
    print(f"\n传播 {distance} mm 后:")
    print(f"  束腰位置: {pilot_40.waist_position_mm} mm")
    print(f"  曲率半径: {pilot_40.curvature_radius_mm} mm")


def test_pilot_beam_flat_mirror():
    """测试平面镜处 Pilot Beam 参数是否保持不变"""
    print("\n" + "=" * 60)
    print("测试 2: 平面镜处 Pilot Beam 参数")
    print("=" * 60)
    
    wavelength_um = 0.55
    w0_mm = 5.0
    z0_mm = -40.0  # 束腰在 40mm 之前
    
    pilot = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    print(f"\n平面镜入射前 Pilot Beam:")
    print(f"  束腰位置: {pilot.waist_position_mm} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm} mm")
    print(f"  q 参数: {pilot.q_parameter}")
    
    # 平面镜反射（曲率半径无穷大）
    pilot_after = pilot.apply_mirror(np.inf)
    
    print(f"\n平面镜反射后 Pilot Beam:")
    print(f"  束腰位置: {pilot_after.waist_position_mm} mm")
    print(f"  曲率半径: {pilot_after.curvature_radius_mm} mm")
    print(f"  q 参数: {pilot_after.q_parameter}")
    
    # 验证：参数应该完全相同
    print(f"\n验证（平面镜应该不改变参数）:")
    print(f"  束腰位置变化: {pilot_after.waist_position_mm - pilot.waist_position_mm} mm")
    print(f"  曲率半径变化: {pilot_after.curvature_radius_mm - pilot.curvature_radius_mm} mm")
    print(f"  是同一对象: {pilot_after is pilot}")


def test_pilot_beam_curved_mirror():
    """测试曲面镜处 Pilot Beam 参数变化"""
    print("\n" + "=" * 60)
    print("测试 3: 曲面镜处 Pilot Beam 参数")
    print("=" * 60)
    
    wavelength_um = 0.55
    w0_mm = 5.0
    z0_mm = -100.0  # 束腰在 100mm 之前
    
    pilot = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    print(f"\n曲面镜入射前 Pilot Beam:")
    print(f"  束腰位置: {pilot.waist_position_mm} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm} mm")
    
    # 凹面镜反射（R = 200mm，焦距 f = 100mm）
    mirror_radius = 200.0
    pilot_after = pilot.apply_mirror(mirror_radius)
    
    print(f"\n凹面镜（R={mirror_radius}mm）反射后 Pilot Beam:")
    print(f"  束腰位置: {pilot_after.waist_position_mm} mm")
    print(f"  曲率半径: {pilot_after.curvature_radius_mm} mm")
    
    # 理论计算：球面镜等效于焦距 f = R/2 的薄透镜
    # 1/q_out = 1/q_in - 2/R
    print(f"\n理论验证:")
    print(f"  镜面焦距: {mirror_radius/2} mm")


def test_full_propagation_pilot_beam():
    """测试完整传播过程中 Pilot Beam 参数"""
    print("\n" + "=" * 60)
    print("测试 4: 完整传播过程中 Pilot Beam 参数追踪")
    print("=" * 60)
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    print(f"\n加载光学系统: {zmx_file}")
    optical_system = load_optical_system_from_zmx(zmx_file)
    
    # 打印光学系统信息
    print(f"\n光学系统表面:")
    for i, surface in enumerate(optical_system):
        if is_coordinate_break(surface):
            print(f"  [{i}] 坐标断点")
        else:
            print(f"  [{i}] {surface.surface_type}, R={surface.radius}, "
                  f"mirror={surface.is_mirror}, pos={surface.vertex_position}")
    
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
    
    # 手动追踪 Pilot Beam
    print(f"\n手动追踪 Pilot Beam 参数:")
    
    # 初始状态
    pilot = PilotBeamParams.from_gaussian_source(0.55, 5.0, 0.0)
    print(f"\n[初始] z=0:")
    print(f"  束腰位置: {pilot.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm:.2f} mm")
    
    # 传播到表面 0（z=10mm）
    pilot = pilot.propagate(10.0)
    print(f"\n[表面 0 入射] z=10mm:")
    print(f"  束腰位置: {pilot.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm:.2f} mm")
    
    # 表面 0 是平面（无光学作用）
    # 传播到表面 3（z=40mm，距离 30mm）
    pilot = pilot.propagate(30.0)
    print(f"\n[表面 3 入射] z=40mm:")
    print(f"  束腰位置: {pilot.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm:.2f} mm")
    
    # 表面 3 是 45° 平面镜
    surface_3 = optical_system[3]
    print(f"\n表面 3 信息: R={surface_3.radius}, mirror={surface_3.is_mirror}")
    
    pilot_after_mirror = pilot.apply_mirror(surface_3.radius)
    print(f"\n[表面 3 出射] 平面镜反射后:")
    print(f"  束腰位置: {pilot_after_mirror.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {pilot_after_mirror.curvature_radius_mm:.2f} mm")
    print(f"  参数是否改变: {pilot_after_mirror is not pilot}")
    
    # 执行实际传播并比较
    print("\n" + "-" * 40)
    print("实际传播结果:")
    
    result = propagator.propagate()
    
    for state in result.surface_states:
        if state.position == 'source':
            print(f"\n[{state.position}]:")
        else:
            print(f"\n[表面 {state.surface_index} {state.position}]:")
        print(f"  束腰位置: {state.pilot_beam_params.waist_position_mm:.4f} mm")
        print(f"  曲率半径: {state.pilot_beam_params.curvature_radius_mm:.2f} mm")


def test_sampling_precision():
    """测试采样精度对波前 RMS 误差的影响"""
    print("\n" + "=" * 60)
    print("测试 5: 采样精度对波前 RMS 误差的影响")
    print("=" * 60)
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    optical_system = load_optical_system_from_zmx(zmx_file)
    
    # 测试不同的光线数量
    ray_counts = [50, 100, 150, 200, 300]
    grid_sizes = [128, 256, 512]
    
    print(f"\n测试不同光线数量（grid_size=256）:")
    print("-" * 50)
    
    for num_rays in ray_counts:
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
            num_rays=num_rays,
        )
        
        result = propagator.propagate()
        
        # 找到表面 3 出射面状态
        for state in result.surface_states:
            if state.surface_index == 3 and state.position == 'exit':
                # 计算相位误差
                grid_size = state.grid_sampling.grid_size
                sampling_mm = state.grid_sampling.sampling_mm
                x = np.arange(grid_size) - grid_size // 2
                y = np.arange(grid_size) - grid_size // 2
                xx, yy = np.meshgrid(x * sampling_mm, y * sampling_mm)
                r_sq = xx**2 + yy**2
                
                R = state.pilot_beam_params.curvature_radius_mm
                wavelength_mm = 0.55 * 1e-3
                k = 2 * np.pi / wavelength_mm
                
                if np.isinf(R):
                    pilot_phase = np.zeros_like(r_sq)
                else:
                    pilot_phase = k * r_sq / (2 * R)
                
                phase_diff = state.phase - pilot_phase
                mask = state.amplitude > 0.01
                
                if np.any(mask):
                    phase_rms_waves = np.std(phase_diff[mask]) / (2 * np.pi)
                    print(f"  num_rays={num_rays:3d}: RMS = {phase_rms_waves:.6f} waves")
                break
    
    print(f"\n测试不同网格大小（num_rays=150）:")
    print("-" * 50)
    
    for grid_size in grid_sizes:
        source = SourceDefinition(
            wavelength_um=0.55,
            w0_mm=5.0,
            z0_mm=0.0,
            grid_size=grid_size,
            physical_size_mm=40.0,
        )
        
        propagator = HybridOpticalPropagator(
            optical_system=optical_system,
            source=source,
            wavelength_um=0.55,
            grid_size=grid_size,
            num_rays=150,
        )
        
        result = propagator.propagate()
        
        # 找到表面 3 出射面状态
        for state in result.surface_states:
            if state.surface_index == 3 and state.position == 'exit':
                # 计算相位误差
                gs = state.grid_sampling.grid_size
                sampling_mm = state.grid_sampling.sampling_mm
                x = np.arange(gs) - gs // 2
                y = np.arange(gs) - gs // 2
                xx, yy = np.meshgrid(x * sampling_mm, y * sampling_mm)
                r_sq = xx**2 + yy**2
                
                R = state.pilot_beam_params.curvature_radius_mm
                wavelength_mm = 0.55 * 1e-3
                k = 2 * np.pi / wavelength_mm
                
                if np.isinf(R):
                    pilot_phase = np.zeros_like(r_sq)
                else:
                    pilot_phase = k * r_sq / (2 * R)
                
                phase_diff = state.phase - pilot_phase
                mask = state.amplitude > 0.01
                
                if np.any(mask):
                    phase_rms_waves = np.std(phase_diff[mask]) / (2 * np.pi)
                    print(f"  grid_size={grid_size:3d}: RMS = {phase_rms_waves:.6f} waves")
                break


def test_hybrid_element_propagator_pilot_beam():
    """测试 HybridElementPropagator 中 Pilot Beam 的更新"""
    print("\n" + "=" * 60)
    print("测试 6: HybridElementPropagator 中 Pilot Beam 更新")
    print("=" * 60)
    
    from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
    from hybrid_optical_propagation.data_models import PilotBeamParams
    
    propagator = HybridElementPropagator(
        wavelength_um=0.55,
        num_rays=150,
        method="local_raytracing",
    )
    
    # 测试 _update_pilot_beam 方法
    pilot = PilotBeamParams.from_gaussian_source(0.55, 5.0, -40.0)
    
    print(f"\n入射 Pilot Beam:")
    print(f"  束腰位置: {pilot.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {pilot.curvature_radius_mm:.2f} mm")
    
    # 创建一个模拟的平面镜表面
    class MockSurface:
        def __init__(self, radius, is_mirror, material=''):
            self.radius = radius
            self.is_mirror = is_mirror
            self.material = material
    
    # 平面镜
    flat_mirror = MockSurface(np.inf, True)
    new_pilot = propagator._update_pilot_beam(pilot, flat_mirror)
    
    print(f"\n平面镜后 Pilot Beam:")
    print(f"  束腰位置: {new_pilot.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {new_pilot.curvature_radius_mm:.2f} mm")
    print(f"  是同一对象: {new_pilot is pilot}")
    
    # 凹面镜（R=200mm）
    curved_mirror = MockSurface(200.0, True)
    new_pilot_curved = propagator._update_pilot_beam(pilot, curved_mirror)
    
    print(f"\n凹面镜（R=200mm）后 Pilot Beam:")
    print(f"  束腰位置: {new_pilot_curved.waist_position_mm:.4f} mm")
    print(f"  曲率半径: {new_pilot_curved.curvature_radius_mm:.2f} mm")


def main():
    """主函数"""
    test_pilot_beam_free_space()
    test_pilot_beam_flat_mirror()
    test_pilot_beam_curved_mirror()
    test_full_propagation_pilot_beam()
    test_hybrid_element_propagator_pilot_beam()
    test_sampling_precision()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
