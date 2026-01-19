"""
诊断 Galilean OAP 扩束镜问题

目标：分离自由空间传输和元件追迹，定位错误位置

测试方案：
1. 纯 PROPER 模式（use_hybrid_propagation=False）- 基准
2. 混合模式（use_hybrid_propagation=True）- 当前问题
3. 单独测试每个元件的光线追迹 OPD
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


def create_system(use_hybrid: bool) -> SequentialOpticalSystem:
    """创建 Galilean OAP 扩束镜系统"""
    # 光源参数
    wavelength = 10.64      # μm
    w0_input = 10.0         # mm
    
    # 扩束镜焦距设计
    f1 = -50.0              # mm, OAP1 焦距（凸面）
    f2 = 150.0              # mm, OAP2 焦距（凹面）
    
    # 离轴参数
    d_off_oap1 = 2 * abs(f1)  # 100 mm
    d_off_oap2 = 2 * f2       # 300 mm
    
    # 倾斜角度
    theta = np.radians(45.0)
    
    # 几何参数
    d_oap1_to_fold = 50.0
    d_fold_to_oap2 = 50.0
    d_oap2_to_output = 100.0
    
    # 光源
    source = GaussianBeamSource(wavelength=wavelength, w0=w0_input, z0=0.0)
    
    # 创建系统
    system = SequentialOpticalSystem(
        source=source,
        grid_size=512,
        beam_ratio=0.25,
        use_hybrid_propagation=use_hybrid,
        hybrid_num_rays=100,
    )
    
    # 添加采样面和元件
    system.add_sampling_plane(distance=0.0, name="Input")
    
    system.add_surface(ParabolicMirror(
        parent_focal_length=f1,
        thickness=d_oap1_to_fold,
        semi_aperture=20.0,
        off_axis_distance=d_off_oap1,
        tilt_x=theta,
        name="OAP1",
    ))
    
    system.add_sampling_plane(distance=d_oap1_to_fold, name="After OAP1")
    
    system.add_surface(FlatMirror(
        thickness=d_fold_to_oap2,
        semi_aperture=30.0,
        tilt_x=theta,
        name="Fold",
    ))
    
    system.add_sampling_plane(distance=d_oap1_to_fold + d_fold_to_oap2, name="After Fold")
    
    system.add_surface(ParabolicMirror(
        parent_focal_length=f2,
        thickness=d_oap2_to_output,
        semi_aperture=50.0,
        off_axis_distance=d_off_oap2,
        tilt_x=theta,
        name="OAP2",
    ))
    
    system.add_sampling_plane(distance=200.0, name="Output")
    
    return system


def test_pure_proper():
    """测试 1: 纯 PROPER 模式（基准）"""
    print("\n" + "=" * 60)
    print("测试 1: 纯 PROPER 模式（use_hybrid_propagation=False）")
    print("=" * 60)
    
    system = create_system(use_hybrid=False)
    results = system.run()
    
    print(f"\n{'采样面':<15} {'PROPER w (mm)':<15} {'ABCD w (mm)':<15} {'误差 (%)':<10} {'WFE RMS':<10}")
    print("-" * 65)
    
    for result in results:
        abcd = system.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"{result.name:<15} {result.beam_radius:<15.3f} {abcd.w:<15.3f} {error:<10.2f} {result.wavefront_rms:<10.4f}")
    
    return results


def test_hybrid_mode():
    """测试 2: 混合传播模式"""
    print("\n" + "=" * 60)
    print("测试 2: 混合传播模式（use_hybrid_propagation=True）")
    print("=" * 60)
    
    system = create_system(use_hybrid=True)
    results = system.run()
    
    print(f"\n{'采样面':<15} {'PROPER w (mm)':<15} {'ABCD w (mm)':<15} {'误差 (%)':<10} {'WFE RMS':<10}")
    print("-" * 65)
    
    for result in results:
        abcd = system.get_abcd_result(result.distance)
        error = abs(result.beam_radius - abcd.w) / abcd.w * 100 if abcd.w > 0.001 else 0
        print(f"{result.name:<15} {result.beam_radius:<15.3f} {abcd.w:<15.3f} {error:<10.2f} {result.wavefront_rms:<10.4f}")
    
    return results


def test_single_oap_raytracing():
    """测试 3: 单独测试 OAP 光线追迹"""
    print("\n" + "=" * 60)
    print("测试 3: 单独测试 OAP1 光线追迹 OPD")
    print("=" * 60)
    
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
    # OAP1 参数
    f1 = -50.0  # mm
    vertex_radius = 2 * f1  # -100 mm
    
    # 创建表面定义
    surface_def = SurfaceDefinition(
        surface_type='conic',
        radius=vertex_radius,
        conic=-1.0,  # 抛物面
        is_mirror=True,
        semi_aperture=20.0,
        tilt_x=np.radians(45.0),
    )
    
    # 创建入射光线（平行光，沿 z 轴）
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    wavelength = 10.64  # μm
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n入射光线 x 坐标: {x}")
    print(f"出射光线 x 坐标: {np.asarray(rays_out.x)}")
    print(f"出射光线 y 坐标: {np.asarray(rays_out.y)}")
    print(f"OPD (波长数): {opd_waves}")
    print(f"有效光线: {valid_mask}")
    
    # 分析 OPD
    valid_opd = opd_waves[valid_mask]
    if len(valid_opd) > 0:
        print(f"\nOPD 统计:")
        print(f"  最小值: {np.min(valid_opd):.6f} waves")
        print(f"  最大值: {np.max(valid_opd):.6f} waves")
        print(f"  PV: {np.max(valid_opd) - np.min(valid_opd):.6f} waves")
        print(f"  RMS: {np.std(valid_opd):.6f} waves")
    
    return opd_waves, valid_mask


def test_single_oap_no_tilt():
    """测试 4: 无倾斜的 OAP 光线追迹（验证基本功能）"""
    print("\n" + "=" * 60)
    print("测试 4: 无倾斜的 OAP1 光线追迹（验证基本功能）")
    print("=" * 60)
    
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
    # OAP1 参数（无倾斜）
    f1 = -50.0  # mm
    vertex_radius = 2 * f1  # -100 mm
    
    # 创建表面定义（无倾斜）
    surface_def = SurfaceDefinition(
        surface_type='conic',
        radius=vertex_radius,
        conic=-1.0,  # 抛物面
        is_mirror=True,
        semi_aperture=20.0,
        tilt_x=0.0,  # 无倾斜
    )
    
    # 创建入射光线
    n_rays = 11
    x = np.linspace(-10, 10, n_rays)
    y = np.zeros(n_rays)
    z = np.zeros(n_rays)
    L = np.zeros(n_rays)
    M = np.zeros(n_rays)
    N = np.ones(n_rays)
    intensity = np.ones(n_rays)
    wavelength = 10.64  # μm
    
    rays_in = RealRays(x, y, z, L, M, N, intensity, wavelength)
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"\n入射光线 x 坐标: {x}")
    print(f"出射光线 x 坐标: {np.asarray(rays_out.x)}")
    print(f"OPD (波长数): {opd_waves}")
    
    # 分析 OPD
    valid_opd = opd_waves[valid_mask]
    if len(valid_opd) > 0:
        print(f"\nOPD 统计:")
        print(f"  最小值: {np.min(valid_opd):.6f} waves")
        print(f"  最大值: {np.max(valid_opd):.6f} waves")
        print(f"  PV: {np.max(valid_opd) - np.min(valid_opd):.6f} waves")
        print(f"  RMS: {np.std(valid_opd):.6f} waves")
        
        # 理论上，凸抛物面镜对平行光应该产生发散球面波
        # OPD 应该是二次函数形式
        print(f"\n理论分析:")
        print(f"  凸抛物面镜 f = {f1} mm")
        print(f"  对于平行光入射，出射应为发散球面波")
        print(f"  OPD 应该是 r²/(2f) 的形式")
    
    return opd_waves, valid_mask


def test_compare_proper_vs_raytracing():
    """测试 5: 对比 PROPER prop_lens 和光线追迹的 OPD"""
    print("\n" + "=" * 60)
    print("测试 5: 对比 PROPER prop_lens 和光线追迹的 OPD")
    print("=" * 60)
    
    import proper
    
    # 参数
    wavelength_um = 10.64
    wavelength_m = wavelength_um * 1e-6
    w0 = 10.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3
    grid_size = 64
    beam_ratio = 0.25
    f1 = -50.0  # mm
    
    # 创建 PROPER 波前
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 获取初始相位
    phase_before = proper.prop_get_phase(wfo).copy()
    
    # 应用透镜
    proper.prop_lens(wfo, f1 * 1e-3)
    
    # 获取透镜后相位
    phase_after = proper.prop_get_phase(wfo).copy()
    
    # 计算相位变化
    phase_change = phase_after - phase_before
    
    # 转换为 OPD（波长数）
    opd_proper_waves = phase_change / (2 * np.pi)
    
    # 获取采样
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    n = grid_size
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    
    # 理论 OPD（透镜相位）
    # phase = -k * r² / (2f)
    # OPD = phase / (2π) = -r² / (2f * λ)
    wavelength_mm = wavelength_um * 1e-3
    opd_theory_waves = -R**2 / (2 * f1 * wavelength_mm)
    
    # 中心切片对比
    center = n // 2
    
    print(f"\n采样: {sampling_mm:.4f} mm/pixel")
    print(f"物理尺寸: {2*half_size:.1f} mm")
    
    print(f"\n中心行 OPD 对比 (波长数):")
    print(f"{'x (mm)':<10} {'PROPER':<15} {'理论':<15} {'差异':<15}")
    print("-" * 55)
    
    for i in range(0, n, n//8):
        x = coords[i]
        proper_val = opd_proper_waves[center, i]
        theory_val = opd_theory_waves[center, i]
        diff = proper_val - theory_val
        print(f"{x:<10.2f} {proper_val:<15.6f} {theory_val:<15.6f} {diff:<15.6f}")
    
    return opd_proper_waves, opd_theory_waves


def main():
    """主函数"""
    print("=" * 60)
    print("Galilean OAP 扩束镜问题诊断")
    print("=" * 60)
    
    # 测试 1: 纯 PROPER 模式
    results_proper = test_pure_proper()
    
    # 测试 2: 混合模式
    results_hybrid = test_hybrid_mode()
    
    # 测试 3: 单独测试 OAP 光线追迹（有倾斜）
    test_single_oap_raytracing()
    
    # 测试 4: 无倾斜的 OAP 光线追迹
    test_single_oap_no_tilt()
    
    # 测试 5: 对比 PROPER 和光线追迹
    test_compare_proper_vs_raytracing()
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    print("\n对比纯 PROPER 和混合模式的结果:")
    print(f"{'采样面':<15} {'PROPER w':<12} {'Hybrid w':<12} {'差异':<12}")
    print("-" * 50)
    
    for r_proper, r_hybrid in zip(results_proper, results_hybrid):
        diff = r_hybrid.beam_radius - r_proper.beam_radius
        print(f"{r_proper.name:<15} {r_proper.beam_radius:<12.3f} {r_hybrid.beam_radius:<12.3f} {diff:<12.3f}")
    
    print("\n关键观察:")
    print("1. 如果纯 PROPER 模式正确，混合模式错误 -> 问题在元件追迹")
    print("2. 如果两种模式都错误 -> 问题可能在系统配置或自由空间传输")
    print("3. 检查 OAP 光线追迹的 OPD 是否符合预期")


if __name__ == "__main__":
    main()
