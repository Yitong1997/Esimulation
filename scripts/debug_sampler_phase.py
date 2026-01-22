"""调试 WavefrontToRaysSampler 的相位提取"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler

# 参数
grid_size = 64
physical_size = 10.0  # mm
wavelength = 0.6328  # μm
R_curvature = 10000.0  # mm

# 创建坐标网格
half_size = physical_size / 2.0
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_sq = X**2 + Y**2

# 创建球面波前相位
wavelength_mm = wavelength * 1e-3
k = 2 * np.pi / wavelength_mm
phase = k * r_sq / (2 * R_curvature)

print(f"输入相位网格：")
print(f"  形状: {phase.shape}")
print(f"  范围: [{np.min(phase):.6f}, {np.max(phase):.6f}] rad")
print(f"  中心值 phase[{grid_size//2}, {grid_size//2}]: {phase[grid_size//2, grid_size//2]:.6f} rad")
print(f"  角落值 phase[0, 0]: {phase[0, 0]:.6f} rad")

# 创建复振幅
wavefront = np.exp(1j * phase)

# 创建采样器
sampler = WavefrontToRaysSampler(
    wavefront_amplitude=wavefront,
    physical_size=physical_size,
    wavelength=wavelength,
    num_rays=50,
)

print(f"\n采样器内部相位网格：")
print(f"  形状: {sampler.phase_grid.shape}")
print(f"  范围: [{np.min(sampler.phase_grid):.6f}, {np.max(sampler.phase_grid):.6f}] rad")

# 检查相位提取是否正确
phase_diff = sampler.phase_grid - phase
print(f"  与输入相位的差异: max={np.max(np.abs(phase_diff)):.6e}")

# 获取光线数据
rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
opd_mm = np.asarray(rays.opd)

print(f"\n光线数据：")
print(f"  光线数量: {len(ray_x)}")

# 手动计算期望的 OPD
r_sq_rays = ray_x**2 + ray_y**2
expected_opd_mm = r_sq_rays / (2 * R_curvature)

# 手动从相位网格插值
from scipy.interpolate import RegularGridInterpolator

interpolator = RegularGridInterpolator(
    (sampler.y_coords, sampler.x_coords),
    sampler.phase_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

points = np.column_stack([ray_y, ray_x])
phase_at_rays = interpolator(points)
manual_opd_mm = phase_at_rays * wavelength_mm / (2 * np.pi)

print(f"\nOPD 比较：")
print(f"  采样器 OPD 范围: [{np.min(opd_mm):.6f}, {np.max(opd_mm):.6f}] mm")
print(f"  手动插值 OPD 范围: [{np.min(manual_opd_mm):.6f}, {np.max(manual_opd_mm):.6f}] mm")
print(f"  期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm")

# 检查差异
diff_sampler = opd_mm - expected_opd_mm
diff_manual = manual_opd_mm - expected_opd_mm

print(f"\n差异分析：")
print(f"  采样器 OPD 与期望的差异: max={np.max(np.abs(diff_sampler)):.6e} mm")
print(f"  手动插值 OPD 与期望的差异: max={np.max(np.abs(diff_manual)):.6e} mm")

# 检查几条具体的光线
print(f"\n具体光线检查：")
for i in [0, len(ray_x)//2, -1]:
    x, y = ray_x[i], ray_y[i]
    r2 = x**2 + y**2
    expected = r2 / (2 * R_curvature)
    actual = opd_mm[i]
    manual = manual_opd_mm[i]
    
    # 从相位网格直接查找最近的值
    ix = np.argmin(np.abs(sampler.x_coords - x))
    iy = np.argmin(np.abs(sampler.y_coords - y))
    phase_nearest = sampler.phase_grid[iy, ix]
    opd_nearest = phase_nearest * wavelength_mm / (2 * np.pi)
    
    print(f"  光线 {i}: 位置=({x:.4f}, {y:.4f})")
    print(f"    r² = {r2:.6f}")
    print(f"    期望 OPD = {expected:.6f} mm")
    print(f"    采样器 OPD = {actual:.6f} mm")
    print(f"    手动插值 OPD = {manual:.6f} mm")
    print(f"    最近网格点相位 = {phase_nearest:.6f} rad")
    print(f"    最近网格点 OPD = {opd_nearest:.6f} mm")
