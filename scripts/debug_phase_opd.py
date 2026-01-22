"""
调试脚本：检查薄相位元件的 OPD 计算

目标：
1. 理解 WavefrontToRaysSampler 中相位如何转换为 OPD
2. 检查 optiland 的 PhaseInteractionModel 中 OPD 计算的单位问题
3. 验证出射光线 OPD 是否等于输入相位对应的光程

关键公式：
- optiland 中：opd_shift = -phase_val / k0
- k0 = 2π / wavelength（wavelength 单位是 μm）
- 所以 opd_shift 单位是 μm，但 optiland 的 OPD 单位应该是 mm
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
# 第一部分：创建简单的测试波前
# ============================================================================

print_section("第一部分：创建简单的测试波前")

# 参数
wavelength_um = 0.55
grid_size = 256
physical_size_mm = 20.0

# 创建坐标网格
half_size = physical_size_mm / 2.0
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
R_sq = X**2 + Y**2

# 创建一个简单的球面波前相位
# 假设曲率半径 R_curvature = 100 mm
R_curvature_mm = 100.0
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

# 球面波前相位：φ = k * r² / (2R)
phase_rad = k * R_sq / (2 * R_curvature_mm)

# 对应的 OPD（mm）：OPD = r² / (2R)
expected_opd_mm = R_sq / (2 * R_curvature_mm)

print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")
print(f"曲率半径: {R_curvature_mm} mm")
print(f"网格大小: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")
print(f"\n相位范围: [{np.min(phase_rad):.4f}, {np.max(phase_rad):.4f}] rad")
print(f"期望 OPD 范围: [{np.min(expected_opd_mm):.6f}, {np.max(expected_opd_mm):.6f}] mm")

# 创建复振幅
amplitude = np.ones((grid_size, grid_size))
simulation_amplitude = amplitude * np.exp(1j * phase_rad)


# ============================================================================
# 第二部分：使用 WavefrontToRaysSampler 采样光线
# ============================================================================

print_section("第二部分：使用 WavefrontToRaysSampler 采样光线")

from wavefront_to_rays import WavefrontToRaysSampler

sampler = WavefrontToRaysSampler(
    wavefront_amplitude=simulation_amplitude,
    physical_size=physical_size_mm,
    wavelength=wavelength_um,
    num_rays=100,
    distribution="hexapolar",
)

# 获取光线数据
ray_x, ray_y = sampler.get_ray_positions()
output_rays = sampler.get_output_rays()
ray_opd_mm = np.asarray(output_rays.opd)

print(f"采样光线数量: {len(ray_x)}")
print(f"光线位置范围 X: [{np.min(ray_x):.4f}, {np.max(ray_x):.4f}] mm")
print(f"光线位置范围 Y: [{np.min(ray_y):.4f}, {np.max(ray_y):.4f}] mm")
print(f"光线 OPD 范围: [{np.min(ray_opd_mm):.6f}, {np.max(ray_opd_mm):.6f}] mm")


# ============================================================================
# 第三部分：计算期望的 OPD 并比较
# ============================================================================

print_section("第三部分：计算期望的 OPD 并比较")

# 在光线位置处计算期望的 OPD
ray_r_sq = ray_x**2 + ray_y**2
expected_ray_opd_mm = ray_r_sq / (2 * R_curvature_mm)

print(f"期望光线 OPD 范围: [{np.min(expected_ray_opd_mm):.6f}, {np.max(expected_ray_opd_mm):.6f}] mm")

# 计算主光线 OPD（作为参考）
chief_ray = sampler.optic.trace_generic(
    Hx=0, Hy=0, Px=0, Py=0, wavelength=wavelength_um
)
chief_opd_mm = float(np.asarray(chief_ray.opd).item())
print(f"\n主光线 OPD: {chief_opd_mm:.6f} mm")

# 计算相对 OPD
relative_ray_opd_mm = ray_opd_mm - chief_opd_mm
print(f"相对光线 OPD 范围: [{np.min(relative_ray_opd_mm):.6f}, {np.max(relative_ray_opd_mm):.6f}] mm")

# 比较相对 OPD 与期望 OPD
opd_diff_mm = relative_ray_opd_mm - expected_ray_opd_mm
opd_diff_waves = opd_diff_mm / wavelength_mm

print(f"\nOPD 差异范围: [{np.min(opd_diff_mm):.6f}, {np.max(opd_diff_mm):.6f}] mm")
print(f"OPD 差异范围: [{np.min(opd_diff_waves):.6f}, {np.max(opd_diff_waves):.6f}] waves")
print(f"OPD 差异 RMS: {np.std(opd_diff_waves):.6f} waves")


# ============================================================================
# 第四部分：分析 optiland 的 OPD 计算
# ============================================================================

print_section("第四部分：分析 optiland 的 OPD 计算")

# optiland 的 PhaseInteractionModel 中：
# opd_shift = -phase_val / k0
# k0 = 2π / wavelength_um
# 所以 opd_shift = -phase_val * wavelength_um / (2π)

# 但是 WavefrontToRaysSampler 中将相位缩小了 1000 倍：
# corrected_phase = phase_rad / 1000.0

# 所以实际的 opd_shift = -corrected_phase * wavelength_um / (2π)
#                      = -phase_rad / 1000.0 * wavelength_um / (2π)
#                      = -phase_rad * wavelength_um / (2000π)

# 而期望的 OPD（mm）= phase_rad / k = phase_rad * wavelength_mm / (2π)
#                   = phase_rad * wavelength_um * 1e-3 / (2π)

# 比较：
# 实际 opd_shift = -phase_rad * wavelength_um / (2000π)
# 期望 OPD = phase_rad * wavelength_um * 1e-3 / (2π)
#          = phase_rad * wavelength_um / (2000π)

# 所以实际 opd_shift = -期望 OPD
# 符号相反！

print("optiland PhaseInteractionModel 中的 OPD 计算：")
print("  opd_shift = -phase_val / k0")
print("  k0 = 2π / wavelength_um")
print("  所以 opd_shift = -phase_val × wavelength_um / (2π)")
print("")
print("WavefrontToRaysSampler 中将相位缩小了 1000 倍：")
print("  corrected_phase = phase_rad / 1000.0")
print("")
print("所以实际的 opd_shift = -corrected_phase × wavelength_um / (2π)")
print("                    = -phase_rad × wavelength_um / (2000π)")
print("")
print("而期望的 OPD（mm）= phase_rad × wavelength_mm / (2π)")
print("                 = phase_rad × wavelength_um × 1e-3 / (2π)")
print("                 = phase_rad × wavelength_um / (2000π)")
print("")
print("比较：")
print("  实际 opd_shift = -期望 OPD")
print("  符号相反！")

# 验证
# 在光线位置处计算相位
ray_phase_rad = k * ray_r_sq / (2 * R_curvature_mm)
corrected_ray_phase = ray_phase_rad / 1000.0
k0_um = 2 * np.pi / wavelength_um
opd_shift_from_phase = -corrected_ray_phase / k0_um

print(f"\n验证计算：")
print(f"  光线位置处相位范围: [{np.min(ray_phase_rad):.4f}, {np.max(ray_phase_rad):.4f}] rad")
print(f"  修正后相位范围: [{np.min(corrected_ray_phase):.6f}, {np.max(corrected_ray_phase):.6f}] rad")
print(f"  opd_shift 范围: [{np.min(opd_shift_from_phase):.6f}, {np.max(opd_shift_from_phase):.6f}] μm")
print(f"  期望 OPD 范围: [{np.min(expected_ray_opd_mm):.6f}, {np.max(expected_ray_opd_mm):.6f}] mm")


# ============================================================================
# 第五部分：检查物面到相位面的传播 OPD
# ============================================================================

print_section("第五部分：检查物面到相位面的传播 OPD")

# optiland 的光学系统结构：
# index=0: 物面（无穷远，thickness=inf）
# index=1: 相位面（平面，thickness=0）
# index=2: 像面

# 从物面到相位面的传播会产生 OPD
# 对于平面波入射（从无穷远），所有光线的传播距离相同
# 所以传播 OPD 应该是常数

# 检查 optiland 的表面配置
print("optiland 光学系统配置：")
for i, surface in enumerate(sampler.optic.surface_group.surfaces):
    print(f"  Surface {i}:")
    print(f"    radius: {surface.geometry.radius}")
    if hasattr(surface.geometry, 'cs') and hasattr(surface.geometry.cs, 'thickness'):
        print(f"    thickness: {surface.geometry.cs.thickness}")
    if hasattr(surface, 'material_pre'):
        print(f"    material_pre: {surface.material_pre}")
    if hasattr(surface, 'material_post'):
        print(f"    material_post: {surface.material_post}")


# ============================================================================
# 第六部分：绘制诊断图
# ============================================================================

print_section("第六部分：绘制诊断图")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 输入相位分布
im1 = axes[0, 0].imshow(phase_rad, extent=[-half_size, half_size, -half_size, half_size], cmap='RdBu_r')
axes[0, 0].set_title('Input Phase (rad)')
axes[0, 0].set_xlabel('X (mm)')
axes[0, 0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0, 0])

# 2. 期望 OPD 分布
im2 = axes[0, 1].imshow(expected_opd_mm, extent=[-half_size, half_size, -half_size, half_size], cmap='viridis')
axes[0, 1].set_title('Expected OPD (mm)')
axes[0, 1].set_xlabel('X (mm)')
axes[0, 1].set_ylabel('Y (mm)')
plt.colorbar(im2, ax=axes[0, 1])

# 3. 光线 OPD vs 期望 OPD
axes[0, 2].scatter(expected_ray_opd_mm, relative_ray_opd_mm, s=5, alpha=0.5)
axes[0, 2].plot([0, np.max(expected_ray_opd_mm)], [0, np.max(expected_ray_opd_mm)], 'r--', label='y=x')
axes[0, 2].set_xlabel('Expected OPD (mm)')
axes[0, 2].set_ylabel('Actual Relative OPD (mm)')
axes[0, 2].set_title('OPD Comparison')
axes[0, 2].legend()

# 4. OPD 差异 vs 半径
ray_r = np.sqrt(ray_r_sq)
axes[1, 0].scatter(ray_r, opd_diff_waves, s=5, alpha=0.5)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Radius (mm)')
axes[1, 0].set_ylabel('OPD Difference (waves)')
axes[1, 0].set_title(f'OPD Difference vs Radius\nRMS={np.std(opd_diff_waves):.4f} waves')

# 5. OPD 差异分布
axes[1, 1].hist(opd_diff_waves, bins=50, color='steelblue', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('OPD Difference (waves)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('OPD Difference Distribution')

# 6. 相位 vs OPD 关系
axes[1, 2].scatter(ray_phase_rad, relative_ray_opd_mm * 1000, s=5, alpha=0.5, label='Actual')
axes[1, 2].scatter(ray_phase_rad, expected_ray_opd_mm * 1000, s=5, alpha=0.5, label='Expected')
axes[1, 2].set_xlabel('Phase (rad)')
axes[1, 2].set_ylabel('OPD (μm)')
axes[1, 2].set_title('Phase vs OPD')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('debug_phase_opd.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: debug_phase_opd.png")


# ============================================================================
# 第七部分：总结
# ============================================================================

print_section("第七部分：总结")

print(f"""
分析结果：

1. 输入相位范围: [{np.min(phase_rad):.4f}, {np.max(phase_rad):.4f}] rad
2. 期望 OPD 范围: [{np.min(expected_ray_opd_mm):.6f}, {np.max(expected_ray_opd_mm):.6f}] mm
3. 实际相对 OPD 范围: [{np.min(relative_ray_opd_mm):.6f}, {np.max(relative_ray_opd_mm):.6f}] mm
4. OPD 差异 RMS: {np.std(opd_diff_waves):.6f} waves

关键发现：
- optiland 的 PhaseInteractionModel 中 OPD 计算使用 opd_shift = -phase_val / k0
- k0 = 2π / wavelength_um，所以 opd_shift 单位是 μm
- WavefrontToRaysSampler 将相位缩小 1000 倍来修正光线方向
- 这导致 OPD 也被缩小了 1000 倍，从 μm 变成了 mm（正好抵消）
- 但符号是负的！

问题：
- 相位 φ 对应的 OPD 应该是 φ/k = φ × λ / (2π)
- 但 optiland 计算的是 -φ/k0 = -φ × λ / (2π)
- 符号相反！
""")
