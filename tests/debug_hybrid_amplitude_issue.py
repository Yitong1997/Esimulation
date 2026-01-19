"""
调试混合传播模式的振幅重建问题

核心问题：
- PROPER 的传播机制依赖于振幅分布在 prop_lens 后保持不变
- 混合模式重建的振幅分布与原始分布不同，导致后续传播出错
- 从 100 条光线重建 512x512 的振幅场，信息严重不足

解决方案：
- 混合模式应该只修改相位，不修改振幅
- 或者使用更多的光线来重建振幅
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from hybrid_propagation.amplitude_reconstruction import AmplitudeReconstructor


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_amplitude_reconstruction_quality():
    """测试振幅重建的质量"""
    print_section("测试振幅重建的质量")
    
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
    
    # 获取原始振幅
    original_amplitude = proper.prop_get_amplitude(wfo)
    
    print(f"原始振幅统计:")
    print(f"  形状: {original_amplitude.shape}")
    print(f"  最大值: {np.max(original_amplitude):.6f}")
    print(f"  总能量: {np.sum(original_amplitude**2):.6f}")
    
    # 计算原始光束半径
    intensity = original_amplitude**2
    total = np.sum(intensity)
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    original_beam_radius = np.sqrt(2 * (x_var + y_var))
    print(f"  光束半径: {original_beam_radius:.3f} mm")
    
    # 模拟混合模式的采样和重建过程
    print(f"\n模拟混合模式采样和重建:")
    
    # 获取复振幅
    stored_amplitude = proper.prop_shift_center(wfo.wfarr.copy())
    wavelength_um = wavelength_m * 1e6
    physical_size = sampling * n
    
    # 测试不同的光线数量
    for num_rays in [100, 500, 1000, 5000]:
        # 创建波前采样器
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=stored_amplitude,
            wavelength=wavelength_um,
            physical_size=physical_size,
            num_rays=num_rays,
        )
        
        rays = sampler.get_output_rays()
        ray_x, ray_y = sampler.get_ray_positions()
        ray_opd = sampler.get_ray_opd()
        
        # 创建重建器
        reconstructor = AmplitudeReconstructor(
            grid_size=n,
            physical_size=physical_size,
            wavelength=wavelength_um,
        )
        
        # 重建复振幅
        valid_mask = np.ones(len(ray_x), dtype=bool)
        zero_reference = np.zeros((n, n))
        reconstructed = reconstructor.reconstruct(
            ray_x=ray_x,
            ray_y=ray_y,
            ray_intensity=np.asarray(rays.i),
            ray_opd_waves=ray_opd,
            reference_phase=zero_reference,
            valid_mask=valid_mask,
        )
        
        reconstructed_amplitude = np.abs(reconstructed)
        
        # 计算重建光束半径
        intensity_recon = reconstructed_amplitude**2
        total_recon = np.sum(intensity_recon)
        if total_recon > 0:
            x_var_recon = np.sum(X**2 * intensity_recon) / total_recon
            y_var_recon = np.sum(Y**2 * intensity_recon) / total_recon
            recon_beam_radius = np.sqrt(2 * (x_var_recon + y_var_recon))
        else:
            recon_beam_radius = 0
        
        print(f"\n  {num_rays} 条光线:")
        print(f"    重建振幅最大值: {np.max(reconstructed_amplitude):.6f}")
        print(f"    重建总能量: {np.sum(reconstructed_amplitude**2):.6f}")
        print(f"    重建光束半径: {recon_beam_radius:.3f} mm")
        print(f"    光束半径误差: {abs(recon_beam_radius - original_beam_radius) / original_beam_radius * 100:.1f}%")


def test_phase_only_modification():
    """测试只修改相位的方案"""
    print_section("测试只修改相位的方案")
    
    # 参数
    wavelength_m = 10.64e-6  # m
    w0 = 10.0e-3  # m
    beam_diameter = 4 * w0  # m
    grid_size = 512
    beam_ratio = 0.25
    
    f1 = -50.0e-3  # m，凸面镜
    
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
    
    # 保存原始振幅
    original_amplitude = proper.prop_get_amplitude(wfo).copy()
    
    print(f"初始状态:")
    print(f"  z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
    print(f"  w0 = {wfo.w0 * 1e3:.3f} mm")
    
    # 方案 1：使用 prop_lens（标准 PROPER 方式）
    wfo1 = proper.prop_begin(beam_diameter, wavelength_m, grid_size, beam_ratio)
    wfo1.wfarr = wfo1.wfarr * gaussian_field_fft
    proper.prop_lens(wfo1, f1)
    proper.prop_propagate(wfo1, 50e-3)
    
    amp1 = proper.prop_get_amplitude(wfo1)
    intensity1 = amp1**2
    total1 = np.sum(intensity1)
    x_var1 = np.sum(X**2 * intensity1) / total1
    y_var1 = np.sum(Y**2 * intensity1) / total1
    beam_radius1 = np.sqrt(2 * (x_var1 + y_var1))
    
    print(f"\n方案 1 (prop_lens):")
    print(f"  传播 50mm 后光束半径: {beam_radius1:.3f} mm")
    print(f"  z_w0 = {wfo1.z_w0 * 1e3:.3f} mm")
    
    # 方案 2：只修改相位，保持振幅不变
    wfo2 = proper.prop_begin(beam_diameter, wavelength_m, grid_size, beam_ratio)
    wfo2.wfarr = wfo2.wfarr * gaussian_field_fft
    
    # 计算透镜相位（与 prop_lens 相同）
    sampling_m = proper.prop_get_sampling(wfo2)
    coords_m = np.linspace(-sampling_m * n / 2, sampling_m * n / 2, n)
    X_m, Y_m = np.meshgrid(coords_m, coords_m)
    R_sq_m = X_m**2 + Y_m**2
    
    k = 2 * np.pi / wavelength_m
    lens_phase = -k * R_sq_m / (2 * f1)
    
    # 应用透镜相位（保持振幅不变）
    stored = proper.prop_shift_center(wfo2.wfarr.copy())
    stored_amplitude = np.abs(stored)
    stored_phase = np.angle(stored)
    new_phase = stored_phase + lens_phase
    new_field = stored_amplitude * np.exp(1j * new_phase)
    wfo2.wfarr = proper.prop_shift_center(new_field)
    
    # 手动更新高斯参数（复制 prop_lens 的逻辑）
    # 这是关键！
    rayleigh_factor = proper.rayleigh_factor
    wfo2.z_Rayleigh = np.pi * wfo2.w0**2 / wfo2.lamda
    w_at_surface = wfo2.w0 * np.sqrt(1.0 + ((wfo2.z - wfo2.z_w0) / wfo2.z_Rayleigh)**2)
    
    if (wfo2.z - wfo2.z_w0) != 0.0:
        gR_beam_old = (wfo2.z - wfo2.z_w0) + wfo2.z_Rayleigh**2 / (wfo2.z - wfo2.z_w0)
        if gR_beam_old != f1:
            gR_beam = 1.0 / (1.0 / gR_beam_old - 1.0 / f1)
            gR_beam_inf = 0
        else:
            gR_beam_inf = 1
    else:
        gR_beam = -f1
        gR_beam_inf = 0
    
    if not gR_beam_inf:
        wfo2.z_w0 = -gR_beam / (1.0 + (wfo2.lamda * gR_beam / (np.pi * w_at_surface**2))**2) + wfo2.z
        wfo2.w0 = w_at_surface / np.sqrt(1.0 + (np.pi * w_at_surface**2 / (wfo2.lamda * gR_beam))**2)
    else:
        wfo2.z_w0 = wfo2.z
        wfo2.w0 = w_at_surface
    
    wfo2.z_Rayleigh = np.pi * wfo2.w0**2 / wfo2.lamda
    
    if np.abs(wfo2.z_w0 - wfo2.z) < rayleigh_factor * wfo2.z_Rayleigh:
        beam_type_new = "INSIDE_"
    else:
        beam_type_new = "OUTSIDE"
    
    wfo2.propagator_type = wfo2.beam_type_old + "_to_" + beam_type_new
    
    if beam_type_new == "INSIDE_":
        wfo2.reference_surface = "PLANAR"
    else:
        wfo2.reference_surface = "SPHERI"
    
    wfo2.beam_type_old = beam_type_new
    wfo2.current_fratio = np.abs(wfo2.z_w0 - wfo2.z) / (2.0 * w_at_surface)
    
    # 传播
    proper.prop_propagate(wfo2, 50e-3)
    
    amp2 = proper.prop_get_amplitude(wfo2)
    intensity2 = amp2**2
    total2 = np.sum(intensity2)
    x_var2 = np.sum(X**2 * intensity2) / total2
    y_var2 = np.sum(Y**2 * intensity2) / total2
    beam_radius2 = np.sqrt(2 * (x_var2 + y_var2))
    
    print(f"\n方案 2 (只修改相位 + 更新高斯参数):")
    print(f"  传播 50mm 后光束半径: {beam_radius2:.3f} mm")
    print(f"  z_w0 = {wfo2.z_w0 * 1e3:.3f} mm")
    
    # 方案 3：只修改相位，不更新高斯参数
    wfo3 = proper.prop_begin(beam_diameter, wavelength_m, grid_size, beam_ratio)
    wfo3.wfarr = wfo3.wfarr * gaussian_field_fft
    
    # 应用透镜相位
    stored = proper.prop_shift_center(wfo3.wfarr.copy())
    stored_amplitude = np.abs(stored)
    stored_phase = np.angle(stored)
    new_phase = stored_phase + lens_phase
    new_field = stored_amplitude * np.exp(1j * new_phase)
    wfo3.wfarr = proper.prop_shift_center(new_field)
    
    # 不更新高斯参数
    proper.prop_propagate(wfo3, 50e-3)
    
    amp3 = proper.prop_get_amplitude(wfo3)
    intensity3 = amp3**2
    total3 = np.sum(intensity3)
    x_var3 = np.sum(X**2 * intensity3) / total3
    y_var3 = np.sum(Y**2 * intensity3) / total3
    beam_radius3 = np.sqrt(2 * (x_var3 + y_var3))
    
    print(f"\n方案 3 (只修改相位，不更新高斯参数):")
    print(f"  传播 50mm 后光束半径: {beam_radius3:.3f} mm")
    print(f"  z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
    
    print(f"\n结论:")
    print(f"  方案 1 和方案 2 应该给出相同的结果")
    print(f"  方案 3 会给出错误的结果，因为高斯参数没有更新")


def test_hybrid_mode_fix():
    """测试修复后的混合模式"""
    print_section("测试修复后的混合模式")
    
    print("""
修复方案：
1. 混合模式应该只计算元件引入的 OPD（相位变化）
2. 将 OPD 转换为相位，应用到原始波前
3. 更新 PROPER 的高斯光束参数
4. 不要重建振幅分布！

这样做的好处：
- 保持 PROPER 的振幅分布不变
- 只添加元件引入的相位变化
- 高斯光束参数正确更新
- 后续传播正常工作
""")


if __name__ == "__main__":
    test_amplitude_reconstruction_quality()
    test_phase_only_modification()
    test_hybrid_mode_fix()
