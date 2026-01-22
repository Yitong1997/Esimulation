"""
调试脚本：检查相位折叠对 OPD 的影响

目标：
1. 使用较小的相位值（避免折叠）来验证 OPD 计算
2. 确认 optiland 的 OPD 单位问题
"""

import sys
sys.path.insert(0, 'src')

import numpy as np


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 使用较小的相位值测试
# ============================================================================

print_section("使用较小的相位值测试")

from optiland.optic import Optic
from optiland.phase import GridPhaseProfile
from optiland.rays import RealRays

# 参数
wavelength_um = 0.55
grid_size = 64
physical_size_mm = 20.0

# 创建坐标网格
half_size = physical_size_mm / 2.0
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
R_sq = X**2 + Y**2

# 使用非常大的曲率半径，使相位很小
R_curvature_mm = 10000.0  # 10 m
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

# 球面波前相位：φ = k * r² / (2R)
phase_rad = k * R_sq / (2 * R_curvature_mm)

print(f"波长: {wavelength_um} μm")
print(f"曲率半径: {R_curvature_mm} mm")
print(f"相位范围: [{np.min(phase_rad):.6f}, {np.max(phase_rad):.6f}] rad")
print(f"相位范围: [{np.min(phase_rad)/(2*np.pi):.6f}, {np.max(phase_rad)/(2*np.pi):.6f}] waves")

# 创建光学系统
optic = Optic()
optic.set_aperture(aperture_type='EPD', value=physical_size_mm)
optic.set_field_type(field_type='angle')
optic.add_field(y=0, x=0)
optic.add_wavelength(value=wavelength_um, is_primary=True)

# 创建相位分布
phase_profile = GridPhaseProfile(
    x_coords=coords,
    y_coords=coords,
    phase_grid=phase_rad,
)

# 添加表面
optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
optic.add_surface(
    index=1,
    surface_type='standard',
    radius=np.inf,
    thickness=0.0,
    material='air',
    is_stop=True,
    phase_profile=phase_profile,
)
optic.add_surface(index=2, radius=np.inf, thickness=0.0, material='air')

# 追迹光线
print("\n追迹主光线 (Px=0, Py=0):")
chief_ray = optic.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=wavelength_um)
print(f"  OPD: {chief_ray.opd[0]:.6f} mm")

# 追迹多条光线
print("\n追迹不同位置的光线:")
for px in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ray = optic.trace_generic(Hx=0, Hy=0, Px=px, Py=0, wavelength=wavelength_um)
    x = ray.x[0]
    r_sq = x**2
    expected_phase = k * r_sq / (2 * R_curvature_mm)
    expected_opd_mm = r_sq / (2 * R_curvature_mm)
    relative_opd = ray.opd[0] - chief_ray.opd[0]
    
    print(f"  Px={px:.2f}: x={x:.4f} mm, phase={expected_phase:.6f} rad, "
          f"expected_opd={expected_opd_mm:.6f} mm, actual_relative_opd={relative_opd:.6f} mm")


# ============================================================================
# 检查 optiland 的 OPD 计算公式
# ============================================================================

print_section("检查 optiland 的 OPD 计算公式")

# 在 x=5mm 处
x_test = 5.0
r_sq_test = x_test**2
phase_test = k * r_sq_test / (2 * R_curvature_mm)

print(f"测试点: x = {x_test} mm")
print(f"  r² = {r_sq_test} mm²")
print(f"  相位 = {phase_test:.6f} rad")

# optiland 的计算
k0 = 2 * np.pi / wavelength_um  # rad/μm
opd_shift_optiland = -phase_test / k0  # 单位是 μm，但被当作 mm 使用
print(f"\noptiland 计算:")
print(f"  k0 = 2π / {wavelength_um} μm = {k0:.4f} rad/μm")
print(f"  opd_shift = -phase / k0 = -{phase_test:.6f} / {k0:.4f} = {opd_shift_optiland:.6f}")
print(f"  （optiland 把这个值当作 mm，但实际单位是 μm）")

# 期望的计算
expected_opd_mm = r_sq_test / (2 * R_curvature_mm)
print(f"\n期望计算:")
print(f"  OPD = r² / (2R) = {r_sq_test} / (2 × {R_curvature_mm}) = {expected_opd_mm:.6f} mm")

# 比较
print(f"\n比较:")
print(f"  optiland opd_shift = {opd_shift_optiland:.6f} (被当作 mm)")
print(f"  期望 OPD = {expected_opd_mm:.6f} mm")
print(f"  比值 = {opd_shift_optiland / expected_opd_mm:.4f}")
print(f"  （应该是 -1000，因为 optiland 把 μm 当作 mm）")


# ============================================================================
# 验证 WavefrontToRaysSampler 的行为
# ============================================================================

print_section("验证 WavefrontToRaysSampler 的行为")

from wavefront_to_rays import WavefrontToRaysSampler

# 创建复振幅
amplitude = np.ones((grid_size, grid_size))
simulation_amplitude = amplitude * np.exp(1j * phase_rad)

# 使用 WavefrontToRaysSampler
sampler = WavefrontToRaysSampler(
    wavefront_amplitude=simulation_amplitude,
    physical_size=physical_size_mm,
    wavelength=wavelength_um,
    num_rays=50,
    distribution="hexapolar",
)

# 获取光线数据
ray_x, ray_y = sampler.get_ray_positions()
output_rays = sampler.get_output_rays()
ray_opd_mm = np.asarray(output_rays.opd)

# 获取主光线 OPD
chief_ray_sampler = sampler.optic.trace_generic(
    Hx=0, Hy=0, Px=0, Py=0, wavelength=wavelength_um
)
chief_opd_mm = float(np.asarray(chief_ray_sampler.opd).item())

# 计算相对 OPD
relative_opd_mm = ray_opd_mm - chief_opd_mm

# 计算期望 OPD
ray_r_sq = ray_x**2 + ray_y**2
expected_opd_mm = ray_r_sq / (2 * R_curvature_mm)

print(f"WavefrontToRaysSampler 结果:")
print(f"  光线数量: {len(ray_x)}")
print(f"  主光线 OPD: {chief_opd_mm:.6f} mm")
print(f"  相对 OPD 范围: [{np.min(relative_opd_mm):.6f}, {np.max(relative_opd_mm):.6f}] mm")
print(f"  期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm")

# 比较
opd_diff = relative_opd_mm - expected_opd_mm
opd_diff_waves = opd_diff / wavelength_mm
print(f"\nOPD 差异:")
print(f"  范围: [{np.min(opd_diff):.6f}, {np.max(opd_diff):.6f}] mm")
print(f"  范围: [{np.min(opd_diff_waves):.4f}, {np.max(opd_diff_waves):.4f}] waves")
print(f"  RMS: {np.std(opd_diff_waves):.4f} waves")

# 检查 WavefrontToRaysSampler 中的相位缩放
print(f"\nWavefrontToRaysSampler 中的相位缩放:")
print(f"  原始相位范围: [{np.min(phase_rad):.6f}, {np.max(phase_rad):.6f}] rad")
print(f"  缩放后相位范围: [{np.min(phase_rad/1000):.6f}, {np.max(phase_rad/1000):.6f}] rad")

# 计算缩放后的 OPD shift
scaled_phase = phase_rad / 1000.0
scaled_opd_shift = -scaled_phase / k0  # μm，但被当作 mm
print(f"\n缩放后的 opd_shift:")
print(f"  范围: [{np.min(scaled_opd_shift):.9f}, {np.max(scaled_opd_shift):.9f}] (被当作 mm)")
print(f"  实际单位是 μm / 1000 = nm")


# ============================================================================
# 总结
# ============================================================================

print_section("总结")

print(f"""
关键发现：

1. optiland 的 PhaseInteractionModel 中 OPD 计算：
   - opd_shift = -phase_val / k0
   - k0 = 2π / wavelength_um（单位：rad/μm）
   - opd_shift 单位是 μm，但被当作 mm 加到 rays.opd 上
   - 这导致 OPD 被放大了 1000 倍！

2. WavefrontToRaysSampler 的修正：
   - 将相位缩小 1000 倍来修正光线方向
   - 这也导致 OPD shift 被缩小 1000 倍
   - 最终 OPD shift 的单位变成了 nm（纳米）
   - 所以相位面对 OPD 的贡献几乎为零

3. 实际观察：
   - 相对 OPD 范围: [{np.min(relative_opd_mm):.6f}, {np.max(relative_opd_mm):.6f}] mm
   - 期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm
   - OPD 差异 RMS: {np.std(opd_diff_waves):.4f} waves

结论：
- optiland 的相位面 OPD 计算存在单位问题
- WavefrontToRaysSampler 的 1000 倍修正使问题更严重
- 相位面对 OPD 的贡献被严重低估（约 1000000 倍）
""")
