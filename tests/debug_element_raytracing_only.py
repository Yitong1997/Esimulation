"""
单独测试元件处的光线追迹流程（不使用 PROPER）

测试流程：
1. 创建一个简单的高斯振幅场（模拟入射波前）
2. 采样为光线
3. 通过元件进行光线追迹
4. 重建复振幅
5. 验证重建结果

这个测试不涉及 PROPER，只测试 wavefront_sampler + element_raytracer + amplitude_reconstruction
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from hybrid_propagation.amplitude_reconstruction import AmplitudeReconstructor


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_gaussian_wavefront(grid_size, physical_size, w0, wavelength_um):
    """创建高斯波前（平面波，无曲率）
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸 (mm)
        w0: 束腰半径 (mm)
        wavelength_um: 波长 (μm)
    
    返回:
        complex_amplitude: 复振幅场
        X, Y: 坐标网格
    """
    half_size = physical_size / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # 高斯振幅，相位为零（平面波）
    amplitude = np.exp(-R_sq / w0**2)
    phase = np.zeros_like(amplitude)
    complex_amplitude = amplitude * np.exp(1j * phase)
    
    return complex_amplitude, X, Y


def measure_beam_radius(amplitude, X, Y):
    """从振幅分布测量光束半径"""
    intensity = np.abs(amplitude)**2
    total = np.sum(intensity)
    
    if total < 1e-15:
        return 0.0
    
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    
    return np.sqrt(2 * (x_var + y_var))


def test_flat_mirror_45deg():
    """测试 45° 平面镜的光线追迹"""
    print_section("测试 45° 平面镜")
    
    # 参数
    grid_size = 256
    physical_size = 80.0  # mm
    w0 = 10.0  # mm
    wavelength_um = 10.64  # μm
    num_rays = 500
    
    # 创建入射波前
    wavefront_in, X, Y = create_gaussian_wavefront(
        grid_size, physical_size, w0, wavelength_um
    )
    
    input_beam_radius = measure_beam_radius(wavefront_in, X, Y)
    print(f"输入光束半径: {input_beam_radius:.3f} mm")
    
    # 创建波前采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront_in,
        wavelength=wavelength_um,
        physical_size=physical_size,
        num_rays=num_rays,
    )
    
    rays_in = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    input_opd = sampler.get_ray_opd()
    
    print(f"采样光线数: {len(ray_x)}")
    print(f"光线位置范围: x=[{np.min(ray_x):.2f}, {np.max(ray_x):.2f}], y=[{np.min(ray_y):.2f}, {np.max(ray_y):.2f}]")
    
    # 创建 45° 平面镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=40.0,
        tilt_x=np.pi/4,  # 45°
    )
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"有效光线数: {np.sum(valid_mask)}")
    print(f"元件 OPD 范围: [{np.nanmin(element_opd):.4f}, {np.nanmax(element_opd):.4f}] waves")
    
    # 出射光线位置
    exit_x = np.asarray(rays_out.x)
    exit_y = np.asarray(rays_out.y)
    exit_intensity = np.asarray(rays_out.i)
    
    print(f"出射光线位置范围: x=[{np.nanmin(exit_x):.2f}, {np.nanmax(exit_x):.2f}], y=[{np.nanmin(exit_y):.2f}, {np.nanmax(exit_y):.2f}]")
    
    # 总 OPD
    total_opd = input_opd + element_opd
    
    # 重建复振幅
    reconstructor = AmplitudeReconstructor(
        grid_size=grid_size,
        physical_size=physical_size,
        wavelength=wavelength_um,
    )
    
    zero_reference = np.zeros((grid_size, grid_size))
    wavefront_out = reconstructor.reconstruct(
        ray_x=exit_x,
        ray_y=exit_y,
        ray_intensity=exit_intensity,
        ray_opd_waves=total_opd,
        reference_phase=zero_reference,
        valid_mask=valid_mask,
    )
    
    output_beam_radius = measure_beam_radius(wavefront_out, X, Y)
    print(f"输出光束半径: {output_beam_radius:.3f} mm")
    print(f"光束半径变化: {(output_beam_radius - input_beam_radius) / input_beam_radius * 100:.1f}%")
    
    # 平面镜不应改变光束半径
    print(f"\n预期: 平面镜不改变光束半径")
    print(f"结果: {'通过' if abs(output_beam_radius - input_beam_radius) / input_beam_radius < 0.1 else '失败'}")
    
    return wavefront_in, wavefront_out, X, Y


def test_parabolic_mirror_45deg():
    """测试 45° 抛物面镜的光线追迹"""
    print_section("测试 45° 抛物面镜 (f=-50mm)")
    
    # 参数
    grid_size = 256
    physical_size = 80.0  # mm
    w0 = 10.0  # mm
    wavelength_um = 10.64  # μm
    num_rays = 500
    f = -50.0  # mm，凸面镜
    
    # 创建入射波前
    wavefront_in, X, Y = create_gaussian_wavefront(
        grid_size, physical_size, w0, wavelength_um
    )
    
    input_beam_radius = measure_beam_radius(wavefront_in, X, Y)
    print(f"输入光束半径: {input_beam_radius:.3f} mm")
    
    # 创建波前采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront_in,
        wavelength=wavelength_um,
        physical_size=physical_size,
        num_rays=num_rays,
    )
    
    rays_in = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    input_opd = sampler.get_ray_opd()
    
    print(f"采样光线数: {len(ray_x)}")
    
    # 创建 45° 抛物面镜
    # 抛物面：conic = -1, radius = 2*f
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=2*f,  # -100mm
        thickness=0.0,
        material='mirror',
        semi_aperture=40.0,
        conic=-1.0,  # 抛物面
        tilt_x=np.pi/4,  # 45°
    )
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    element_opd = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"有效光线数: {np.sum(valid_mask)}")
    print(f"元件 OPD 范围: [{np.nanmin(element_opd):.4f}, {np.nanmax(element_opd):.4f}] waves")
    
    # 出射光线位置和方向
    exit_x = np.asarray(rays_out.x)
    exit_y = np.asarray(rays_out.y)
    exit_L = np.asarray(rays_out.L)
    exit_M = np.asarray(rays_out.M)
    exit_N = np.asarray(rays_out.N)
    exit_intensity = np.asarray(rays_out.i)
    
    print(f"出射光线位置范围: x=[{np.nanmin(exit_x):.2f}, {np.nanmax(exit_x):.2f}]")
    print(f"出射光线方向 (中心光线): L={exit_L[0]:.4f}, M={exit_M[0]:.4f}, N={exit_N[0]:.4f}")
    
    # 总 OPD
    total_opd = input_opd + element_opd
    print(f"总 OPD 范围: [{np.nanmin(total_opd):.4f}, {np.nanmax(total_opd):.4f}] waves")
    
    # 重建复振幅
    reconstructor = AmplitudeReconstructor(
        grid_size=grid_size,
        physical_size=physical_size,
        wavelength=wavelength_um,
    )
    
    zero_reference = np.zeros((grid_size, grid_size))
    wavefront_out = reconstructor.reconstruct(
        ray_x=exit_x,
        ray_y=exit_y,
        ray_intensity=exit_intensity,
        ray_opd_waves=total_opd,
        reference_phase=zero_reference,
        valid_mask=valid_mask,
    )
    
    output_beam_radius = measure_beam_radius(wavefront_out, X, Y)
    print(f"输出光束半径: {output_beam_radius:.3f} mm")
    
    # 抛物面镜在元件处不改变光束半径（只改变波前曲率）
    # 光束半径的变化需要传播一段距离后才能看到
    print(f"\n预期: 抛物面镜在元件处不改变光束半径（只改变波前曲率）")
    print(f"结果: {'通过' if abs(output_beam_radius - input_beam_radius) / input_beam_radius < 0.2 else '失败'}")
    
    return wavefront_in, wavefront_out, X, Y, element_opd, valid_mask


def test_reconstruction_quality():
    """测试重建质量：不同光线数量的影响"""
    print_section("测试重建质量：不同光线数量")
    
    # 参数
    grid_size = 256
    physical_size = 80.0  # mm
    w0 = 10.0  # mm
    wavelength_um = 10.64  # μm
    
    # 创建入射波前
    wavefront_in, X, Y = create_gaussian_wavefront(
        grid_size, physical_size, w0, wavelength_um
    )
    
    input_beam_radius = measure_beam_radius(wavefront_in, X, Y)
    print(f"输入光束半径: {input_beam_radius:.3f} mm")
    
    # 测试不同光线数量
    ray_counts = [50, 100, 200, 500, 1000, 2000]
    
    print(f"\n{'光线数':<10} {'输出半径':<12} {'误差':<10}")
    print("-" * 35)
    
    for num_rays in ray_counts:
        # 采样
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=wavefront_in,
            wavelength=wavelength_um,
            physical_size=physical_size,
            num_rays=num_rays,
        )
        
        rays = sampler.get_output_rays()
        ray_x, ray_y = sampler.get_ray_positions()
        ray_opd = sampler.get_ray_opd()
        
        # 重建（不经过元件，直接重建）
        reconstructor = AmplitudeReconstructor(
            grid_size=grid_size,
            physical_size=physical_size,
            wavelength=wavelength_um,
        )
        
        valid_mask = np.ones(len(ray_x), dtype=bool)
        zero_reference = np.zeros((grid_size, grid_size))
        
        wavefront_out = reconstructor.reconstruct(
            ray_x=ray_x,
            ray_y=ray_y,
            ray_intensity=np.asarray(rays.i),
            ray_opd_waves=ray_opd,
            reference_phase=zero_reference,
            valid_mask=valid_mask,
        )
        
        output_beam_radius = measure_beam_radius(wavefront_out, X, Y)
        error = abs(output_beam_radius - input_beam_radius) / input_beam_radius * 100
        
        print(f"{num_rays:<10} {output_beam_radius:<12.3f} {error:<10.1f}%")


def test_opd_accuracy():
    """测试 OPD 计算精度"""
    print_section("测试 OPD 计算精度")
    
    # 参数
    wavelength_um = 10.64  # μm
    f = -50.0  # mm，凸面镜
    
    # 创建简单的输入光线（平行光）
    from optiland.rays import RealRays
    
    # 在不同半径处的光线
    radii = [0.0, 5.0, 10.0, 15.0, 20.0]
    
    print(f"\n抛物面镜 (f={f}mm) 的 OPD 计算:")
    print(f"{'半径 (mm)':<12} {'实测 OPD':<15} {'理论 OPD':<15} {'误差':<10}")
    print("-" * 55)
    
    for r in radii:
        # 创建单条光线
        input_rays = RealRays(
            x=np.array([r]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([wavelength_um]),
        )
        input_rays.opd = np.array([0.0])
        
        # 创建抛物面镜（无倾斜）
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=2*f,
            thickness=0.0,
            material='mirror',
            semi_aperture=30.0,
            conic=-1.0,
            tilt_x=0.0,  # 无倾斜
        )
        
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        rays_out = raytracer.trace(input_rays)
        opd_waves = raytracer.get_relative_opd_waves()
        
        # 理论 OPD（抛物面镜）
        # 对于抛物面 z = r²/(4f)，反射后 OPD = 2z = r²/(2f)
        # 单位：mm -> waves
        wavelength_mm = wavelength_um * 1e-3
        z_sag = r**2 / (4 * abs(f))
        opd_theory_mm = 2 * z_sag  # 反射加倍
        opd_theory_waves = opd_theory_mm / wavelength_mm
        
        measured_opd = opd_waves[0] if len(opd_waves) > 0 else np.nan
        error = abs(measured_opd - opd_theory_waves) if not np.isnan(measured_opd) else np.nan
        
        print(f"{r:<12.1f} {measured_opd:<15.4f} {opd_theory_waves:<15.4f} {error:<10.4f}")


def visualize_results():
    """可视化测试结果"""
    print_section("生成可视化图像")
    
    # 运行平面镜测试
    wf_in_flat, wf_out_flat, X, Y = test_flat_mirror_45deg()
    
    # 运行抛物面镜测试
    wf_in_para, wf_out_para, _, _, opd_para, valid_para = test_parabolic_mirror_45deg()
    
    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 平面镜
    im1 = axes[0, 0].imshow(np.abs(wf_in_flat), extent=[-40, 40, -40, 40], cmap='hot')
    axes[0, 0].set_title('Input Amplitude (Flat Mirror)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(np.abs(wf_out_flat), extent=[-40, 40, -40, 40], cmap='hot')
    axes[0, 1].set_title('Output Amplitude (Flat Mirror)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    diff_flat = np.abs(wf_out_flat) - np.abs(wf_in_flat)
    im3 = axes[0, 2].imshow(diff_flat, extent=[-40, 40, -40, 40], cmap='RdBu_r')
    axes[0, 2].set_title('Amplitude Difference (Flat Mirror)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 抛物面镜
    im4 = axes[1, 0].imshow(np.abs(wf_in_para), extent=[-40, 40, -40, 40], cmap='hot')
    axes[1, 0].set_title('Input Amplitude (Parabolic Mirror)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(np.abs(wf_out_para), extent=[-40, 40, -40, 40], cmap='hot')
    axes[1, 1].set_title('Output Amplitude (Parabolic Mirror)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # OPD 分布
    opd_valid = np.where(valid_para, opd_para, np.nan)
    im6 = axes[1, 2].scatter(
        np.arange(len(opd_valid))[valid_para], 
        opd_valid[valid_para], 
        c=opd_valid[valid_para], 
        cmap='viridis', 
        s=5
    )
    axes[1, 2].set_title('Element OPD (Parabolic Mirror)')
    axes[1, 2].set_xlabel('Ray Index')
    axes[1, 2].set_ylabel('OPD (waves)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('tests/output/element_raytracing_test.png', dpi=150)
    plt.close()
    print("✓ 保存: tests/output/element_raytracing_test.png")


if __name__ == "__main__":
    test_flat_mirror_45deg()
    test_parabolic_mirror_45deg()
    test_reconstruction_quality()
    test_opd_accuracy()
    
    # 可视化
    import os
    os.makedirs('tests/output', exist_ok=True)
    visualize_results()
