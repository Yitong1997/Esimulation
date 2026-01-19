"""
调试混合传播模式的问题

分析为什么 PROPER 输出的光束半径是 263.6 mm 而不是预期的 30 mm
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_pure_proper_mode():
    """测试纯 PROPER 模式（不使用混合传播）"""
    print_section("测试纯 PROPER 模式")
    
    # 光源参数
    wavelength = 10.64      # μm
    w0_input = 10.0         # mm
    
    # 扩束镜焦距设计
    f1 = -50.0              # mm, OAP1 焦距（负值 = 凸面，发散）
    f2 = 150.0              # mm, OAP2 焦距（正值 = 凹面，准直）
    
    # 几何参数
    d_oap1_to_fold = 50.0
    d_fold_to_oap2 = 50.0
    d_oap2_to_output = 100.0
    total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output
    
    # 离轴参数
    d_off_oap1 = 2 * abs(f1)
    d_off_oap2 = 2 * f2
    theta = np.pi / 4
    
    # 创建系统 - 禁用混合传播模式
    source = GaussianBeamSource(wavelength=wavelength, w0=w0_input, z0=0.0)
    system = SequentialOpticalSystem(
        source=source,
        grid_size=512,
        beam_ratio=0.25,
        use_hybrid_propagation=False,  # 禁用混合传播
    )
    
    # 添加元件
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
    system.add_sampling_plane(distance=total_path, name="Output")
    
    # 运行仿真
    results = system.run()
    
    print("\n纯 PROPER 模式结果:")
    print(f"{'采样面':<12} {'PROPER w (mm)':<15} {'ABCD w (mm)':<15}")
    print("-" * 45)
    for result in results:
        abcd_result = system.get_abcd_result(result.distance)
        print(f"{result.name:<12} {result.beam_radius:<15.3f} {abcd_result.w:<15.3f}")


def test_hybrid_mode():
    """测试混合传播模式"""
    print_section("测试混合传播模式")
    
    # 光源参数
    wavelength = 10.64      # μm
    w0_input = 10.0         # mm
    
    # 扩束镜焦距设计
    f1 = -50.0              # mm
    f2 = 150.0              # mm
    
    # 几何参数
    d_oap1_to_fold = 50.0
    d_fold_to_oap2 = 50.0
    d_oap2_to_output = 100.0
    total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output
    
    # 离轴参数
    d_off_oap1 = 2 * abs(f1)
    d_off_oap2 = 2 * f2
    theta = np.pi / 4
    
    # 创建系统 - 启用混合传播模式
    source = GaussianBeamSource(wavelength=wavelength, w0=w0_input, z0=0.0)
    system = SequentialOpticalSystem(
        source=source,
        grid_size=512,
        beam_ratio=0.25,
        use_hybrid_propagation=True,  # 启用混合传播
        hybrid_num_rays=100,
    )
    
    # 添加元件
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
    system.add_sampling_plane(distance=total_path, name="Output")
    
    # 运行仿真
    results = system.run()
    
    print("\n混合传播模式结果:")
    print(f"{'采样面':<12} {'PROPER w (mm)':<15} {'ABCD w (mm)':<15}")
    print("-" * 45)
    for result in results:
        abcd_result = system.get_abcd_result(result.distance)
        print(f"{result.name:<12} {result.beam_radius:<15.3f} {abcd_result.w:<15.3f}")


def test_simple_lens_proper():
    """测试简单透镜的 PROPER 行为"""
    print_section("测试简单透镜的 PROPER 行为")
    
    # 参数
    wavelength_m = 10.64e-6  # m
    w0 = 10.0e-3  # m
    beam_diameter = 4 * w0  # m
    grid_size = 512
    beam_ratio = 0.25
    
    f1 = -50.0e-3  # m，凸面镜
    f2 = 150.0e-3  # m，凹面镜
    
    # 初始化波前
    wfo = proper.prop_begin(beam_diameter, wavelength_m, grid_size, beam_ratio)
    
    # 应用初始高斯光束
    n = proper.prop_get_gridsize(wfo)
    sampling = proper.prop_get_sampling(wfo) * 1e3  # mm
    half_size = sampling * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    w_0_mm = w0 * 1e3
    amplitude = np.exp(-R_sq / w_0_mm**2)
    gaussian_field = amplitude * np.ones_like(amplitude, dtype=complex)
    gaussian_field_fft = proper.prop_shift_center(gaussian_field)
    wfo.wfarr = wfo.wfarr * gaussian_field_fft
    
    print(f"\n初始状态:")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  w0 = {wfo.w0 * 1e3:.3f} mm")
    print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.3f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
    
    # 测量初始光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    beam_radius_0 = np.sqrt(2 * (x_var + y_var))
    print(f"  测量光束半径 = {beam_radius_0:.3f} mm")
    
    # 应用第一个透镜（凸面镜，f=-50mm）
    proper.prop_lens(wfo, f1)
    
    print(f"\n应用 f1={f1*1e3:.0f}mm 透镜后:")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  w0 = {wfo.w0 * 1e3:.3f} mm")
    print(f"  z_Rayleigh = {wfo.z_Rayleigh * 1e3:.3f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
    
    # 传播 50mm
    proper.prop_propagate(wfo, 50e-3)
    
    print(f"\n传播 50mm 后:")
    print(f"  z = {wfo.z * 1e3:.3f} mm")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    
    # 测量光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    beam_radius_1 = np.sqrt(2 * (x_var + y_var))
    print(f"  测量光束半径 = {beam_radius_1:.3f} mm")
    
    # 应用第二个透镜（凹面镜，f=150mm）
    proper.prop_lens(wfo, f2)
    
    print(f"\n应用 f2={f2*1e3:.0f}mm 透镜后:")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  w0 = {wfo.w0 * 1e3:.3f} mm")
    
    # 传播 100mm
    proper.prop_propagate(wfo, 100e-3)
    
    print(f"\n传播 100mm 后:")
    print(f"  z = {wfo.z * 1e3:.3f} mm")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    
    # 测量光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    beam_radius_2 = np.sqrt(2 * (x_var + y_var))
    print(f"  测量光束半径 = {beam_radius_2:.3f} mm")
    
    # ABCD 计算
    print(f"\nABCD 理论计算:")
    # 伽利略扩束镜：M = -f2/f1 = 3
    M = -f2 / f1
    print(f"  放大倍率 M = {M:.1f}")
    print(f"  预期输出光束半径 = {w0 * 1e3 * M:.1f} mm")


def test_proper_amplitude_vs_gaussian_params():
    """测试 PROPER 振幅分布与高斯参数的关系"""
    print_section("测试 PROPER 振幅分布与高斯参数的关系")
    
    # 参数
    wavelength_m = 10.64e-6  # m
    w0 = 10.0e-3  # m
    beam_diameter = 4 * w0  # m
    grid_size = 512
    beam_ratio = 0.25
    
    # 初始化波前
    wfo = proper.prop_begin(beam_diameter, wavelength_m, grid_size, beam_ratio)
    
    # 应用初始高斯光束
    n = proper.prop_get_gridsize(wfo)
    sampling = proper.prop_get_sampling(wfo) * 1e3  # mm
    half_size = sampling * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    w_0_mm = w0 * 1e3
    amplitude = np.exp(-R_sq / w_0_mm**2)
    gaussian_field = amplitude * np.ones_like(amplitude, dtype=complex)
    gaussian_field_fft = proper.prop_shift_center(gaussian_field)
    wfo.wfarr = wfo.wfarr * gaussian_field_fft
    
    print(f"\n初始状态:")
    print(f"  PROPER 内部 w0 = {wfo.w0 * 1e3:.3f} mm")
    
    # 测量振幅分布的光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    measured_w = np.sqrt(2 * (x_var + y_var))
    print(f"  从振幅测量的光束半径 = {measured_w:.3f} mm")
    
    # 应用透镜
    f = -50.0e-3  # m
    proper.prop_lens(wfo, f)
    
    print(f"\n应用 f={f*1e3:.0f}mm 透镜后:")
    print(f"  PROPER 内部 w0 = {wfo.w0 * 1e3:.3f} mm")
    print(f"  PROPER 内部 z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    
    # 测量振幅分布的光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    measured_w = np.sqrt(2 * (x_var + y_var))
    print(f"  从振幅测量的光束半径 = {measured_w:.3f} mm")
    print(f"  注意：prop_lens 只改变相位，不改变振幅分布！")
    
    # 传播一段距离
    proper.prop_propagate(wfo, 50e-3)
    
    print(f"\n传播 50mm 后:")
    print(f"  PROPER 内部 w0 = {wfo.w0 * 1e3:.3f} mm")
    print(f"  PROPER 内部 z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  PROPER 内部 z = {wfo.z * 1e3:.3f} mm")
    
    # 计算理论光束半径
    z_from_waist = wfo.z - wfo.z_w0
    w_theory = wfo.w0 * np.sqrt(1 + (z_from_waist / wfo.z_Rayleigh)**2)
    print(f"  理论光束半径 (从 PROPER 参数) = {w_theory * 1e3:.3f} mm")
    
    # 测量振幅分布的光束半径
    amp = proper.prop_get_amplitude(wfo)
    intensity = amp**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    measured_w = np.sqrt(2 * (x_var + y_var))
    print(f"  从振幅测量的光束半径 = {measured_w:.3f} mm")


if __name__ == "__main__":
    test_pure_proper_mode()
    test_hybrid_mode()
    test_simple_lens_proper()
    test_proper_amplitude_vs_gaussian_params()
