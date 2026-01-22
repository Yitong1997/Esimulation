"""测试 WavefrontToRaysSampler 的 OPD 计算修正

验证：
1. 光线方向正确（由相位梯度决定）
2. 光线 OPD 正确（直接从输入相位计算）
3. 符号约定正确（正相位 → 正 OPD）
4. 新接口（振幅和相位分离）正常工作
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler


def test_spherical_wavefront():
    """测试球面波前的 OPD 计算
    
    创建一个球面波前（曲率半径 R），验证：
    - 光线方向指向曲率中心
    - 光线 OPD = r² / (2R)（球面波前的 OPD 公式）
    """
    print("=" * 60)
    print("测试 1：球面波前（振幅和相位分离接口）")
    print("=" * 60)
    
    # 参数
    grid_size = 128
    physical_size = 10.0  # mm
    wavelength = 0.6328  # μm (HeNe)
    R_curvature = 100000.0  # mm，确保相位范围远小于 π
    
    # 创建坐标网格
    half_size = physical_size / 2.0
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    # 创建球面波前相位（非折叠实数）
    wavelength_mm = wavelength * 1e-3
    k = 2 * np.pi / wavelength_mm
    phase = k * r_sq / (2 * R_curvature)
    
    # 创建均匀振幅
    amplitude = np.ones_like(phase)
    
    max_phase = np.max(phase)
    print(f"参数：")
    print(f"  网格大小: {grid_size} x {grid_size}")
    print(f"  物理尺寸: {physical_size} mm")
    print(f"  波长: {wavelength} μm")
    print(f"  曲率半径: {R_curvature} mm")
    print(f"  相位范围: [0, {max_phase:.4f}] rad = [0, {max_phase/np.pi:.4f}π]")
    
    # 创建采样器（使用新接口）
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=100,
    )
    
    # 获取光线数据
    rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    L, M, N = sampler.get_ray_directions()
    opd_mm = np.asarray(rays.opd)
    
    print(f"\n光线数据：")
    print(f"  光线数量: {len(ray_x)}")
    
    # 验证光线方向
    r_sq_rays = ray_x**2 + ray_y**2
    d = np.sqrt(r_sq_rays + R_curvature**2)
    expected_L = -ray_x / d
    expected_M = -ray_y / d
    expected_N = R_curvature / d
    
    L_error = np.abs(L - expected_L)
    M_error = np.abs(M - expected_M)
    N_error = np.abs(N - expected_N)
    
    print(f"\n光线方向验证：")
    print(f"  L 误差 max: {np.max(L_error):.6e}")
    print(f"  M 误差 max: {np.max(M_error):.6e}")
    print(f"  N 误差 max: {np.max(N_error):.6e}")
    
    direction_ok = np.max(L_error) < 2e-4 and np.max(M_error) < 2e-4 and np.max(N_error) < 1e-4
    print(f"  方向正确: {'✓' if direction_ok else '✗'}")
    
    # 验证 OPD
    expected_opd_mm = r_sq_rays / (2 * R_curvature)
    opd_error_mm = np.abs(opd_mm - expected_opd_mm)
    opd_error_waves = opd_error_mm / wavelength_mm
    
    print(f"\nOPD 验证：")
    print(f"  OPD 范围: [{np.min(opd_mm):.6f}, {np.max(opd_mm):.6f}] mm")
    print(f"  期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm")
    print(f"  OPD 误差 max: {np.max(opd_error_mm):.6e} mm = {np.max(opd_error_waves):.6e} waves")
    
    opd_ok = np.max(opd_error_waves) < 0.01
    print(f"  OPD 正确: {'✓' if opd_ok else '✗'}")
    
    return direction_ok and opd_ok


def test_flat_wavefront():
    """测试平面波前的 OPD 计算"""
    print("\n" + "=" * 60)
    print("测试 2：平面波前")
    print("=" * 60)
    
    grid_size = 64
    physical_size = 10.0
    wavelength = 0.6328
    
    # 平面波前：零相位
    amplitude = np.ones((grid_size, grid_size))
    phase = np.zeros((grid_size, grid_size))
    
    print(f"参数：")
    print(f"  网格大小: {grid_size} x {grid_size}")
    print(f"  物理尺寸: {physical_size} mm")
    print(f"  波长: {wavelength} μm")
    
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=50,
    )
    
    rays = sampler.get_output_rays()
    L, M, N = sampler.get_ray_directions()
    opd_mm = np.asarray(rays.opd)
    
    print(f"\n光线数据：")
    print(f"  光线数量: {len(L)}")
    
    # 验证光线方向（应该沿 Z 轴）
    L_error = np.abs(L - 0)
    M_error = np.abs(M - 0)
    N_error = np.abs(N - 1)
    
    print(f"\n光线方向验证：")
    print(f"  L max: {np.max(np.abs(L)):.6e}")
    print(f"  M max: {np.max(np.abs(M)):.6e}")
    print(f"  N 范围: [{np.min(N):.6f}, {np.max(N):.6f}]")
    
    direction_ok = np.max(L_error) < 1e-6 and np.max(M_error) < 1e-6 and np.max(N_error) < 1e-6
    print(f"  方向正确: {'✓' if direction_ok else '✗'}")
    
    # 验证 OPD（应该为 0）
    print(f"\nOPD 验证：")
    print(f"  OPD 范围: [{np.min(opd_mm):.6e}, {np.max(opd_mm):.6e}] mm")
    
    wavelength_mm = wavelength * 1e-3
    opd_error_waves = np.abs(opd_mm) / wavelength_mm
    print(f"  OPD 误差 max: {np.max(opd_error_waves):.6e} waves")
    
    opd_ok = np.max(opd_error_waves) < 0.001
    print(f"  OPD 正确: {'✓' if opd_ok else '✗'}")
    
    return direction_ok and opd_ok


def test_sign_convention():
    """测试 OPD 符号约定：正相位 → 正 OPD"""
    print("\n" + "=" * 60)
    print("测试 3：OPD 符号约定")
    print("=" * 60)
    
    grid_size = 64
    physical_size = 10.0
    wavelength = 0.6328
    
    coords = np.linspace(-physical_size/2, physical_size/2, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r = np.sqrt(X**2 + Y**2)
    
    # 高斯形状的正相位
    phase = 2.0 * np.exp(-r**2 / 4.0)
    amplitude = np.ones_like(phase)
    
    print(f"参数：")
    print(f"  相位范围: [{np.min(phase):.4f}, {np.max(phase):.4f}] rad")
    print(f"  中心相位: {phase[grid_size//2, grid_size//2]:.4f} rad")
    
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=50,
    )
    
    rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    opd_mm = np.asarray(rays.opd)
    
    r_rays = np.sqrt(ray_x**2 + ray_y**2)
    center_idx = np.argmin(r_rays)
    
    center_opd = opd_mm[center_idx]
    
    print(f"\n符号验证：")
    print(f"  中心光线位置: ({ray_x[center_idx]:.4f}, {ray_y[center_idx]:.4f}) mm")
    print(f"  中心光线 OPD: {center_opd:.6f} mm")
    
    wavelength_mm = wavelength * 1e-3
    expected_center_opd = phase[grid_size//2, grid_size//2] * wavelength_mm / (2 * np.pi)
    print(f"  期望 OPD: {expected_center_opd:.6f} mm")
    
    sign_ok = center_opd > 0
    print(f"  符号正确: {'✓' if sign_ok else '✗'}")
    
    return sign_ok


def test_large_phase():
    """测试大相位范围（超过 2π）
    
    这是新接口的关键优势：相位以非折叠实数存储，可以超过 [-π, π]
    """
    print("\n" + "=" * 60)
    print("测试 4：大相位范围（超过 2π）")
    print("=" * 60)
    
    grid_size = 128
    physical_size = 10.0
    wavelength = 0.6328
    
    # 使用较小的曲率半径，使相位超过 2π
    R_curvature = 10000.0  # mm
    
    half_size = physical_size / 2.0
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    wavelength_mm = wavelength * 1e-3
    k = 2 * np.pi / wavelength_mm
    phase = k * r_sq / (2 * R_curvature)
    amplitude = np.ones_like(phase)
    
    max_phase = np.max(phase)
    print(f"参数：")
    print(f"  曲率半径: {R_curvature} mm")
    print(f"  相位范围: [0, {max_phase:.4f}] rad = [0, {max_phase/np.pi:.4f}π]")
    print(f"  相位超过 2π: {'是' if max_phase > 2*np.pi else '否'}")
    
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=100,
    )
    
    rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    opd_mm = np.asarray(rays.opd)
    
    # 验证 OPD
    r_sq_rays = ray_x**2 + ray_y**2
    expected_opd_mm = r_sq_rays / (2 * R_curvature)
    opd_error_mm = np.abs(opd_mm - expected_opd_mm)
    opd_error_waves = opd_error_mm / wavelength_mm
    
    print(f"\nOPD 验证：")
    print(f"  OPD 范围: [{np.min(opd_mm):.6f}, {np.max(opd_mm):.6f}] mm")
    print(f"  期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm")
    print(f"  OPD 误差 max: {np.max(opd_error_mm):.6e} mm = {np.max(opd_error_waves):.6e} waves")
    
    opd_ok = np.max(opd_error_waves) < 0.01
    print(f"  OPD 正确: {'✓' if opd_ok else '✗'}")
    
    return opd_ok


if __name__ == "__main__":
    print("WavefrontToRaysSampler OPD 计算测试")
    print("=" * 60)
    
    results = []
    
    results.append(("球面波前", test_spherical_wavefront()))
    results.append(("平面波前", test_flat_wavefront()))
    results.append(("符号约定", test_sign_convention()))
    results.append(("大相位范围", test_large_phase()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查代码。")
