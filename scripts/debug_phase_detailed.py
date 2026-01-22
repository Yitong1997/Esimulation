"""
详细调试脚本：逐步检查 optiland 的相位面 OPD 计算

目标：
1. 检查相位面是否正确应用了相位
2. 检查 OPD 计算的每一步
3. 找出为什么相对 OPD 几乎为零
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
grid_size = 64  # 使用较小的网格便于调试
physical_size_mm = 20.0

# 创建坐标网格
half_size = physical_size_mm / 2.0
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
R_sq = X**2 + Y**2

# 创建一个简单的球面波前相位
R_curvature_mm = 100.0
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

# 球面波前相位：φ = k * r² / (2R)
phase_rad = k * R_sq / (2 * R_curvature_mm)

print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")
print(f"曲率半径: {R_curvature_mm} mm")
print(f"相位范围: [{np.min(phase_rad):.4f}, {np.max(phase_rad):.4f}] rad")
print(f"相位范围: [{np.min(phase_rad)/(2*np.pi):.4f}, {np.max(phase_rad)/(2*np.pi):.4f}] waves")

# 创建复振幅
amplitude = np.ones((grid_size, grid_size))
simulation_amplitude = amplitude * np.exp(1j * phase_rad)


# ============================================================================
# 第二部分：手动创建 optiland 光学系统并追迹
# ============================================================================

print_section("第二部分：手动创建 optiland 光学系统并追迹")

from optiland.optic import Optic
from optiland.phase import GridPhaseProfile
from optiland.rays import RealRays

# 创建光学系统
optic = Optic()
optic.set_aperture(aperture_type='EPD', value=physical_size_mm)
optic.set_field_type(field_type='angle')
optic.add_field(y=0, x=0)
optic.add_wavelength(value=wavelength_um, is_primary=True)

# 创建相位分布（不缩小 1000 倍，看看原始行为）
phase_profile_original = GridPhaseProfile(
    x_coords=coords,
    y_coords=coords,
    phase_grid=phase_rad,  # 原始相位
)

# 添加物面
optic.add_surface(index=0, radius=np.inf, thickness=np.inf)

# 添加相位面
optic.add_surface(
    index=1,
    surface_type='standard',
    radius=np.inf,
    thickness=0.0,
    material='air',
    is_stop=True,
    phase_profile=phase_profile_original,
)

# 添加像面
optic.add_surface(index=2, radius=np.inf, thickness=0.0, material='air')

# 追迹单条光线（主光线）
print("\n追迹主光线 (Px=0, Py=0):")
chief_ray = optic.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=wavelength_um)
print(f"  位置: ({chief_ray.x[0]:.6f}, {chief_ray.y[0]:.6f}, {chief_ray.z[0]:.6f}) mm")
print(f"  方向: ({chief_ray.L[0]:.6f}, {chief_ray.M[0]:.6f}, {chief_ray.N[0]:.6f})")
print(f"  OPD: {chief_ray.opd[0]:.6f} mm")

# 追迹边缘光线
print("\n追迹边缘光线 (Px=1, Py=0):")
edge_ray = optic.trace_generic(Hx=0, Hy=0, Px=1, Py=0, wavelength=wavelength_um)
print(f"  位置: ({edge_ray.x[0]:.6f}, {edge_ray.y[0]:.6f}, {edge_ray.z[0]:.6f}) mm")
print(f"  方向: ({edge_ray.L[0]:.6f}, {edge_ray.M[0]:.6f}, {edge_ray.N[0]:.6f})")
print(f"  OPD: {edge_ray.opd[0]:.6f} mm")

# 计算相对 OPD
relative_opd = edge_ray.opd[0] - chief_ray.opd[0]
print(f"\n相对 OPD (边缘 - 主光线): {relative_opd:.6f} mm")

# 计算期望的 OPD
edge_r_sq = edge_ray.x[0]**2 + edge_ray.y[0]**2
expected_opd = edge_r_sq / (2 * R_curvature_mm)
print(f"期望 OPD: {expected_opd:.6f} mm")

# 计算边缘处的相位
edge_phase = k * edge_r_sq / (2 * R_curvature_mm)
print(f"边缘处相位: {edge_phase:.4f} rad = {edge_phase/(2*np.pi):.4f} waves")


# ============================================================================
# 第三部分：检查 PhaseInteractionModel 的 OPD 计算
# ============================================================================

print_section("第三部分：检查 PhaseInteractionModel 的 OPD 计算")

# 获取相位面
phase_surface = optic.surface_group.surfaces[1]
print(f"相位面类型: {type(phase_surface)}")
print(f"交互模型类型: {type(phase_surface.interaction_model)}")

# 检查相位值
x_test = np.array([0.0, 5.0, 10.0])
y_test = np.array([0.0, 0.0, 0.0])
phase_values = phase_profile_original.get_phase(x_test, y_test)
print(f"\n测试点相位值:")
for i in range(len(x_test)):
    r_sq = x_test[i]**2 + y_test[i]**2
    expected_phase = k * r_sq / (2 * R_curvature_mm)
    print(f"  ({x_test[i]:.1f}, {y_test[i]:.1f}): 实际={phase_values[i]:.4f} rad, 期望={expected_phase:.4f} rad")

# 计算 optiland 的 OPD shift
k0 = 2 * np.pi / wavelength_um  # rad/μm
opd_shift = -phase_values / k0  # μm
print(f"\noptiland OPD shift (= -phase / k0):")
for i in range(len(x_test)):
    print(f"  ({x_test[i]:.1f}, {y_test[i]:.1f}): {opd_shift[i]:.6f} μm")

# 期望的 OPD（mm）
expected_opd_mm = phase_values / k  # mm
print(f"\n期望 OPD (= phase / k):")
for i in range(len(x_test)):
    print(f"  ({x_test[i]:.1f}, {y_test[i]:.1f}): {expected_opd_mm[i]:.6f} mm")


# ============================================================================
# 第四部分：检查传播 OPD
# ============================================================================

print_section("第四部分：检查传播 OPD")

# 从物面到相位面的传播 OPD
# 对于平面波入射（从无穷远），传播距离是无穷大
# 但 optiland 使用的是 abs(t * n)，其中 t 是到表面的距离

# 让我们手动追迹一条光线，看看每一步的 OPD 变化
print("手动追迹光线，检查每一步的 OPD 变化：")

# 创建一条测试光线
test_ray = RealRays(
    x=np.array([5.0]),  # 在 x=5mm 处
    y=np.array([0.0]),
    z=np.array([0.0]),
    L=np.array([0.0]),
    M=np.array([0.0]),
    N=np.array([1.0]),
    intensity=np.array([1.0]),
    wavelength=np.array([wavelength_um]),
)
test_ray.opd = np.array([0.0])  # 初始 OPD 为 0

print(f"\n初始状态:")
print(f"  位置: ({test_ray.x[0]:.6f}, {test_ray.y[0]:.6f}, {test_ray.z[0]:.6f})")
print(f"  OPD: {test_ray.opd[0]:.6f} mm")

# 通过相位面
# 相位面的交互模型会修改光线方向和 OPD
phase_surface = optic.surface_group.surfaces[1]

# 先定位到表面
phase_surface.geometry.localize(test_ray)
print(f"\n定位到相位面后:")
print(f"  位置: ({test_ray.x[0]:.6f}, {test_ray.y[0]:.6f}, {test_ray.z[0]:.6f})")

# 计算到表面的距离
t = phase_surface.geometry.distance(test_ray)
print(f"  到表面距离 t: {t[0]:.6f} mm")

# 传播到表面
n_pre = phase_surface.material_pre.n(test_ray.w)
print(f"  材料折射率 n: {n_pre}")

# OPD 增量（传播部分）
opd_propagation = np.abs(t * n_pre)
print(f"  传播 OPD 增量: {opd_propagation[0]:.6f} mm")

# 应用交互模型
test_ray_copy = RealRays(
    x=test_ray.x.copy(),
    y=test_ray.y.copy(),
    z=test_ray.z.copy(),
    L=test_ray.L.copy(),
    M=test_ray.M.copy(),
    N=test_ray.N.copy(),
    intensity=test_ray.i.copy(),
    wavelength=test_ray.w.copy(),
)
test_ray_copy.opd = test_ray.opd.copy()

# 传播光线
phase_surface.material_pre.propagation_model.propagate(test_ray_copy, t)
test_ray_copy.opd = test_ray_copy.opd + np.abs(t * n_pre)

print(f"\n传播后:")
print(f"  位置: ({test_ray_copy.x[0]:.6f}, {test_ray_copy.y[0]:.6f}, {test_ray_copy.z[0]:.6f})")
print(f"  OPD: {test_ray_copy.opd[0]:.6f} mm")

# 应用相位交互
test_ray_after = phase_surface.interaction_model.interact_real_rays(test_ray_copy)

print(f"\n相位交互后:")
print(f"  位置: ({test_ray_after.x[0]:.6f}, {test_ray_after.y[0]:.6f}, {test_ray_after.z[0]:.6f})")
print(f"  方向: ({test_ray_after.L[0]:.6f}, {test_ray_after.M[0]:.6f}, {test_ray_after.N[0]:.6f})")
print(f"  OPD: {test_ray_after.opd[0]:.6f} mm")

# 计算相位交互引入的 OPD 变化
opd_phase_change = test_ray_after.opd[0] - test_ray_copy.opd[0]
print(f"\n相位交互引入的 OPD 变化: {opd_phase_change:.6f} mm")

# 期望的 OPD 变化
r_sq = 5.0**2 + 0.0**2
expected_phase = k * r_sq / (2 * R_curvature_mm)
expected_opd_change = expected_phase / k  # = r_sq / (2 * R_curvature_mm)
print(f"期望的 OPD 变化: {expected_opd_change:.6f} mm")

# optiland 计算的 OPD 变化
# opd_shift = -phase_val / k0
# k0 = 2π / wavelength_um
k0 = 2 * np.pi / wavelength_um
optiland_opd_shift = -expected_phase / k0
print(f"optiland 计算的 OPD shift: {optiland_opd_shift:.6f} μm = {optiland_opd_shift * 1e-3:.6f} mm")


# ============================================================================
# 第五部分：分析单位问题
# ============================================================================

print_section("第五部分：分析单位问题")

print("""
optiland 的 PhaseInteractionModel 中：
  opd_shift = -phase_val / k0
  k0 = 2π / wavelength_um

问题分析：
1. k0 的单位是 rad/μm
2. phase_val 的单位是 rad
3. 所以 opd_shift = -phase_val / k0 的单位是 μm

但是 optiland 的 rays.opd 单位是 mm！

这意味着：
- optiland 将 opd_shift（单位 μm）直接加到 rays.opd（单位 mm）上
- 这导致 OPD 被缩小了 1000 倍！

验证：
""")

# 验证
phase_at_5mm = k * 25.0 / (2 * R_curvature_mm)  # r=5mm 处的相位
print(f"r=5mm 处的相位: {phase_at_5mm:.4f} rad")

opd_shift_um = -phase_at_5mm / k0
print(f"optiland 计算的 opd_shift: {opd_shift_um:.6f} μm")

expected_opd_mm = 25.0 / (2 * R_curvature_mm)
print(f"期望的 OPD: {expected_opd_mm:.6f} mm")

print(f"\n比较：")
print(f"  opd_shift (μm) = {opd_shift_um:.6f}")
print(f"  expected_opd (mm) = {expected_opd_mm:.6f}")
print(f"  opd_shift (μm) / expected_opd (mm) = {opd_shift_um / expected_opd_mm:.6f}")
print(f"  （应该是 -1000，因为 1 mm = 1000 μm，且符号相反）")


# ============================================================================
# 第六部分：绘制诊断图
# ============================================================================

print_section("第六部分：绘制诊断图")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 相位分布
im1 = axes[0].imshow(phase_rad, extent=[-half_size, half_size, -half_size, half_size], cmap='RdBu_r')
axes[0].set_title('Phase (rad)')
axes[0].set_xlabel('X (mm)')
axes[0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0])

# 2. 期望 OPD 分布
expected_opd_grid = R_sq / (2 * R_curvature_mm)
im2 = axes[1].imshow(expected_opd_grid, extent=[-half_size, half_size, -half_size, half_size], cmap='viridis')
axes[1].set_title('Expected OPD (mm)')
axes[1].set_xlabel('X (mm)')
axes[1].set_ylabel('Y (mm)')
plt.colorbar(im2, ax=axes[1])

# 3. optiland 计算的 OPD shift
optiland_opd_shift_grid = -phase_rad / k0  # μm
im3 = axes[2].imshow(optiland_opd_shift_grid, extent=[-half_size, half_size, -half_size, half_size], cmap='viridis')
axes[2].set_title('optiland OPD shift (μm)')
axes[2].set_xlabel('X (mm)')
axes[2].set_ylabel('Y (mm)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('debug_phase_detailed.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: debug_phase_detailed.png")


# ============================================================================
# 第七部分：总结
# ============================================================================

print_section("第七部分：总结")

print(f"""
关键发现：

1. optiland 的 PhaseInteractionModel 中 OPD 计算存在单位问题：
   - opd_shift = -phase_val / k0
   - k0 = 2π / wavelength_um（单位：rad/μm）
   - 所以 opd_shift 单位是 μm
   - 但 rays.opd 单位是 mm
   - 这导致 OPD 被缩小了 1000 倍

2. 符号问题：
   - optiland 使用 opd_shift = -phase_val / k0（负号）
   - 期望的 OPD = phase_val / k = phase_val × λ / (2π)（正号）
   - 符号相反

3. WavefrontToRaysSampler 的修正：
   - 将相位缩小 1000 倍：corrected_phase = phase_rad / 1000.0
   - 这修正了光线方向的单位问题
   - 但 OPD 计算仍然有问题：
     - opd_shift = -corrected_phase / k0 = -phase_rad / (1000 × k0)
     - 单位变成了 μm / 1000 = nm（纳米）！
     - 而不是 mm

4. 实际观察到的行为：
   - 相对 OPD 几乎为零（[-0.000231, 0.000360] mm）
   - 这说明相位面的 OPD 贡献被严重缩小了

结论：
- optiland 的 PhaseInteractionModel 存在单位不一致问题
- WavefrontToRaysSampler 的 1000 倍修正只修正了光线方向，没有正确修正 OPD
- 需要进一步检查 optiland 的源代码来确认这个问题
""")
