"""
简化的混合模式测试

测试单个凸抛物面镜（无倾斜）的混合传播
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
)


def test_single_parabolic_mirror():
    """测试单个凸抛物面镜（无倾斜）"""
    print("=" * 70)
    print("测试单个凸抛物面镜（无倾斜）")
    print("=" * 70)
    
    # 参数
    wavelength = 10.64  # μm
    w0 = 10.0  # mm
    f = -50.0  # mm，凸抛物面镜
    
    # 光源
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=0.0)
    
    # 测试纯 PROPER 模式
    print("\n1. 纯 PROPER 模式:")
    print("-" * 50)
    
    system_proper = SequentialOpticalSystem(
        source=source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=False,
    )
    
    system_proper.add_sampling_plane(distance=0.0, name="Input")
    system_proper.add_surface(ParabolicMirror(
        parent_focal_length=f,
        thickness=50.0,
        semi_aperture=20.0,
        off_axis_distance=0.0,  # 无离轴
        tilt_x=0.0,  # 无倾斜
        name="OAP",
    ))
    system_proper.add_sampling_plane(distance=50.0, name="Output")
    
    results_proper = system_proper.run()
    
    for result in results_proper:
        abcd = system_proper.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"  {result.name}: w={result.beam_radius:.3f} mm, ABCD={abcd.w:.3f} mm, "
              f"误差={error:.2f}%, WFE={result.wavefront_rms:.4f}")
    
    # 测试混合模式
    print("\n2. 混合模式:")
    print("-" * 50)
    
    system_hybrid = SequentialOpticalSystem(
        source=source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
        hybrid_num_rays=100,
    )
    
    system_hybrid.add_sampling_plane(distance=0.0, name="Input")
    system_hybrid.add_surface(ParabolicMirror(
        parent_focal_length=f,
        thickness=50.0,
        semi_aperture=20.0,
        off_axis_distance=0.0,
        tilt_x=0.0,
        name="OAP",
    ))
    system_hybrid.add_sampling_plane(distance=50.0, name="Output")
    
    results_hybrid = system_hybrid.run()
    
    for result in results_hybrid:
        abcd = system_hybrid.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"  {result.name}: w={result.beam_radius:.3f} mm, ABCD={abcd.w:.3f} mm, "
              f"误差={error:.2f}%, WFE={result.wavefront_rms:.4f}")
    
    # 对比
    print("\n3. 对比:")
    print("-" * 50)
    
    for r_proper, r_hybrid in zip(results_proper, results_hybrid):
        diff = r_hybrid.beam_radius - r_proper.beam_radius
        print(f"  {r_proper.name}: PROPER={r_proper.beam_radius:.3f}, "
              f"Hybrid={r_hybrid.beam_radius:.3f}, 差异={diff:.3f} mm")


def test_single_parabolic_mirror_45deg():
    """测试单个凸抛物面镜（45° 倾斜）"""
    print("\n" + "=" * 70)
    print("测试单个凸抛物面镜（45° 倾斜）")
    print("=" * 70)
    
    # 参数
    wavelength = 10.64  # μm
    w0 = 10.0  # mm
    f = -50.0  # mm，凸抛物面镜
    theta = np.radians(45.0)
    
    # 光源
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=0.0)
    
    # 测试纯 PROPER 模式
    print("\n1. 纯 PROPER 模式:")
    print("-" * 50)
    
    system_proper = SequentialOpticalSystem(
        source=source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=False,
    )
    
    system_proper.add_sampling_plane(distance=0.0, name="Input")
    system_proper.add_surface(ParabolicMirror(
        parent_focal_length=f,
        thickness=50.0,
        semi_aperture=20.0,
        off_axis_distance=2*abs(f),  # 离轴距离
        tilt_x=theta,  # 45° 倾斜
        name="OAP",
    ))
    system_proper.add_sampling_plane(distance=50.0, name="Output")
    
    results_proper = system_proper.run()
    
    for result in results_proper:
        abcd = system_proper.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"  {result.name}: w={result.beam_radius:.3f} mm, ABCD={abcd.w:.3f} mm, "
              f"误差={error:.2f}%, WFE={result.wavefront_rms:.4f}")
    
    # 测试混合模式
    print("\n2. 混合模式:")
    print("-" * 50)
    
    system_hybrid = SequentialOpticalSystem(
        source=source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
        hybrid_num_rays=100,
    )
    
    system_hybrid.add_sampling_plane(distance=0.0, name="Input")
    system_hybrid.add_surface(ParabolicMirror(
        parent_focal_length=f,
        thickness=50.0,
        semi_aperture=20.0,
        off_axis_distance=2*abs(f),
        tilt_x=theta,
        name="OAP",
    ))
    system_hybrid.add_sampling_plane(distance=50.0, name="Output")
    
    results_hybrid = system_hybrid.run()
    
    for result in results_hybrid:
        abcd = system_hybrid.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"  {result.name}: w={result.beam_radius:.3f} mm, ABCD={abcd.w:.3f} mm, "
              f"误差={error:.2f}%, WFE={result.wavefront_rms:.4f}")
    
    # 对比
    print("\n3. 对比:")
    print("-" * 50)
    
    for r_proper, r_hybrid in zip(results_proper, results_hybrid):
        diff = r_hybrid.beam_radius - r_proper.beam_radius
        print(f"  {r_proper.name}: PROPER={r_proper.beam_radius:.3f}, "
              f"Hybrid={r_hybrid.beam_radius:.3f}, 差异={diff:.3f} mm")


def main():
    test_single_parabolic_mirror()
    test_single_parabolic_mirror_45deg()


if __name__ == "__main__":
    main()
