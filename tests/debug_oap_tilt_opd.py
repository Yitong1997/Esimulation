"""
调试 OAP 倾斜 OPD 计算

分析折叠倾斜引入的理想 OPD 计算是否正确
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 系统参数
# ============================================================

print_section("系统参数")

wavelength_um = 10.64      # μm
w0_input = 10.0            # mm
f1 = -50.0                 # mm, OAP1 焦距（凸面）

# 离轴参数
d_off_oap1 = 2 * abs(f1)   # 100 mm
theta = np.radians(45.0)   # 45° 倾斜

# 波长转换
wavelength_mm = wavelength_um * 1e-3

print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")
print(f"OAP1 焦距: {f1} mm (凸面)")
print(f"OAP1 曲率半径: {2*f1} mm")
print(f"OAP1 离轴距离: {d_off_oap1} mm")
print(f"OAP1 倾斜角: {np.degrees(theta):.1f}°")


# ============================================================
# 创建 OAP1 的 SurfaceDefinition
# ============================================================

print_section("OAP1 SurfaceDefinition")

oap1 = ParabolicMirror(
    parent_focal_length=f1,
    thickness=50.0,
    semi_aperture=20.0,
    off_axis_distance=d_off_oap1,
    tilt_x=theta,
    name="OAP1",
)

surface_def = oap1.get_surface_definition()

print(f"表面类型: {surface_def.surface_type}")
print(f"曲率半径: {surface_def.radius} mm")
print(f"圆锥常数: {surface_def.conic}")
print(f"半口径: {surface_def.semi_aperture} mm")
print(f"倾斜 X: {np.degrees(surface_def.tilt_x):.1f}°")
print(f"倾斜 Y: {np.degrees(surface_def.tilt_y):.1f}°")


# ============================================================
# 测试光线追迹
# ============================================================

print_section("光线追迹测试")

# 创建测试光线（小范围）
n_rays_1d = 5
half_size = 10.0  # mm
ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
ray_x = ray_X.flatten()
ray_y = ray_Y.flatten()
n_rays = len(ray_x)

print(f"测试光线数量: {n_rays}")
print(f"采样范围: ±{half_size} mm")

# 创建平行光入射光线
rays_in = RealRays(
    x=ray_x,
    y=ray_y,
    z=np.zeros(n_rays),
    L=np.zeros(n_rays),
    M=np.zeros(n_rays),
    N=np.ones(n_rays),
    intensity=np.ones(n_rays),
    wavelength=np.full(n_rays, wavelength_um),
)

# 光线追迹
raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=wavelength_um,
)

rays_out = raytracer.trace(rays_in)
opd_waves = raytracer.get_relative_opd_waves()
valid_mask = raytracer.get_valid_ray_mask()

print(f"\n有效光线数量: {np.sum(valid_mask)}")
print(f"实际 OPD 范围: {np.min(opd_waves[valid_mask]):.2f} ~ {np.max(opd_waves[valid_mask]):.2f} waves")


# ============================================================
# 分析理想 OPD 组成
# ============================================================

print_section("理想 OPD 分析")

ray_r_sq = ray_x**2 + ray_y**2
focal_length_mm = oap1.focal_length
tilt_x = oap1.tilt_x
tilt_y = oap1.tilt_y

# 1. 理想聚焦 OPD
ideal_focus_opd_waves = ray_r_sq / (2 * focal_length_mm * wavelength_mm)
print(f"\n1. 理想聚焦 OPD:")
print(f"   公式: r² / (2f) / λ")
print(f"   f = {focal_length_mm} mm")
print(f"   范围: {np.min(ideal_focus_opd_waves):.2f} ~ {np.max(ideal_focus_opd_waves):.2f} waves")

# 2. 理想倾斜 OPD（反射镜，OPD 加倍）
tilt_opd_mm = 2.0 * (ray_x * np.sin(tilt_y) + ray_y * np.sin(tilt_x))
ideal_tilt_opd_waves = tilt_opd_mm / wavelength_mm
print(f"\n2. 理想倾斜 OPD:")
print(f"   公式: 2 * (x*sin(tilt_y) + y*sin(tilt_x)) / λ")
print(f"   tilt_x = {np.degrees(tilt_x):.1f}°, tilt_y = {np.degrees(tilt_y):.1f}°")
print(f"   sin(tilt_x) = {np.sin(tilt_x):.4f}")
print(f"   范围: {np.min(ideal_tilt_opd_waves):.2f} ~ {np.max(ideal_tilt_opd_waves):.2f} waves")

# 3. 总理想 OPD
ideal_opd_waves = ideal_focus_opd_waves + ideal_tilt_opd_waves
print(f"\n3. 总理想 OPD:")
print(f"   范围: {np.min(ideal_opd_waves):.2f} ~ {np.max(ideal_opd_waves):.2f} waves")

# 4. 像差
aberration_waves = opd_waves - ideal_opd_waves
print(f"\n4. 像差 (实际 - 理想):")
print(f"   范围: {np.min(aberration_waves[valid_mask]):.2f} ~ {np.max(aberration_waves[valid_mask]):.2f} waves")
print(f"   均值: {np.mean(aberration_waves[valid_mask]):.2f} waves")
print(f"   标准差: {np.std(aberration_waves[valid_mask]):.2f} waves")


# ============================================================
# 详细分析特定光线
# ============================================================

print_section("详细光线分析")

# 中心光线
center_idx = n_rays // 2
print(f"\n中心光线 (idx={center_idx}):")
print(f"  位置: ({ray_x[center_idx]:.2f}, {ray_y[center_idx]:.2f}) mm")
print(f"  实际 OPD: {opd_waves[center_idx]:.4f} waves")
print(f"  理想聚焦 OPD: {ideal_focus_opd_waves[center_idx]:.4f} waves")
print(f"  理想倾斜 OPD: {ideal_tilt_opd_waves[center_idx]:.4f} waves")
print(f"  总理想 OPD: {ideal_opd_waves[center_idx]:.4f} waves")
print(f"  像差: {aberration_waves[center_idx]:.4f} waves")

# Y 方向边缘光线
y_plus_idx = np.argmax(ray_y)
y_minus_idx = np.argmin(ray_y)

print(f"\nY+ 边缘光线 (idx={y_plus_idx}):")
print(f"  位置: ({ray_x[y_plus_idx]:.2f}, {ray_y[y_plus_idx]:.2f}) mm")
print(f"  实际 OPD: {opd_waves[y_plus_idx]:.4f} waves")
print(f"  理想聚焦 OPD: {ideal_focus_opd_waves[y_plus_idx]:.4f} waves")
print(f"  理想倾斜 OPD: {ideal_tilt_opd_waves[y_plus_idx]:.4f} waves")
print(f"  总理想 OPD: {ideal_opd_waves[y_plus_idx]:.4f} waves")
print(f"  像差: {aberration_waves[y_plus_idx]:.4f} waves")

print(f"\nY- 边缘光线 (idx={y_minus_idx}):")
print(f"  位置: ({ray_x[y_minus_idx]:.2f}, {ray_y[y_minus_idx]:.2f}) mm")
print(f"  实际 OPD: {opd_waves[y_minus_idx]:.4f} waves")
print(f"  理想聚焦 OPD: {ideal_focus_opd_waves[y_minus_idx]:.4f} waves")
print(f"  理想倾斜 OPD: {ideal_tilt_opd_waves[y_minus_idx]:.4f} waves")
print(f"  总理想 OPD: {ideal_opd_waves[y_minus_idx]:.4f} waves")
print(f"  像差: {aberration_waves[y_minus_idx]:.4f} waves")


# ============================================================
# 检查倾斜 OPD 公式
# ============================================================

print_section("倾斜 OPD 公式验证")

# 对于 45° 倾斜的反射镜，Y 方向的 OPD 变化应该是：
# ΔW = 2 * y * sin(45°) = 2 * y * 0.707 = 1.414 * y
# 对于 y = 10 mm：ΔW = 14.14 mm
# 转换为波长数：14.14 / 0.01064 = 1329 waves

y_test = 10.0  # mm
expected_tilt_opd_mm = 2.0 * y_test * np.sin(theta)
expected_tilt_opd_waves = expected_tilt_opd_mm / wavelength_mm

print(f"测试 y = {y_test} mm 处的倾斜 OPD:")
print(f"  公式: 2 * y * sin(45°) / λ")
print(f"  = 2 * {y_test} * {np.sin(theta):.4f} / {wavelength_mm}")
print(f"  = {expected_tilt_opd_mm:.4f} mm / {wavelength_mm}")
print(f"  = {expected_tilt_opd_waves:.2f} waves")

# 实际测量
y_10_idx = np.argmin(np.abs(ray_y - 10.0) + np.abs(ray_x))
print(f"\n实际测量 (idx={y_10_idx}, y={ray_y[y_10_idx]:.2f} mm):")
print(f"  实际 OPD: {opd_waves[y_10_idx]:.2f} waves")
print(f"  理想聚焦 OPD: {ideal_focus_opd_waves[y_10_idx]:.2f} waves")
print(f"  实际 - 聚焦 = {opd_waves[y_10_idx] - ideal_focus_opd_waves[y_10_idx]:.2f} waves")
print(f"  期望倾斜 OPD: {expected_tilt_opd_waves:.2f} waves")


# ============================================================
# 问题诊断
# ============================================================

print_section("问题诊断")

# 检查 Y 方向的 OPD 变化是否与倾斜一致
y_values = np.unique(ray_y)
print("\nY 方向 OPD 变化分析:")
print(f"{'Y (mm)':<10} {'实际 OPD':<15} {'理想聚焦':<15} {'差值':<15} {'期望倾斜':<15}")
print("-" * 70)

for y_val in y_values:
    mask = (ray_y == y_val) & (ray_x == 0)
    if np.any(mask):
        idx = np.where(mask)[0][0]
        actual = opd_waves[idx]
        focus = ideal_focus_opd_waves[idx]
        diff = actual - focus
        expected_tilt = 2.0 * y_val * np.sin(theta) / wavelength_mm
        print(f"{y_val:<10.2f} {actual:<15.2f} {focus:<15.2f} {diff:<15.2f} {expected_tilt:<15.2f}")

print("\n结论:")
print("如果 '差值' 列与 '期望倾斜' 列接近，说明倾斜 OPD 公式正确")
print("如果差异很大，说明 ElementRaytracer 的 OPD 计算可能有问题")
