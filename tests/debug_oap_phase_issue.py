"""
调试伽利略 OAP 扩束镜相位精度问题

分析 OAP1（凸面抛物面镜）的 OPD 计算
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
f2 = 150.0                 # mm, OAP2 焦距（凹面）

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
print(f"OPD 范围: {np.min(opd_waves[valid_mask]):.2f} ~ {np.max(opd_waves[valid_mask]):.2f} waves")
print(f"OPD 均值: {np.mean(opd_waves[valid_mask]):.2f} waves")
print(f"OPD 标准差: {np.std(opd_waves[valid_mask]):.2f} waves")


# ============================================================
# 分析理想 OPD
# ============================================================

print_section("理想 OPD 分析")

ray_r_sq = ray_x**2 + ray_y**2

# 理想聚焦 OPD（波长数）
# OPD_ideal = r² / (2f) / λ
focal_length_mm = oap1.focal_length
ideal_opd_waves = ray_r_sq / (2 * focal_length_mm * wavelength_mm)

print(f"焦距: {focal_length_mm} mm")
print(f"波长: {wavelength_mm} mm")
print(f"理想 OPD 范围: {np.min(ideal_opd_waves):.2f} ~ {np.max(ideal_opd_waves):.2f} waves")

# 计算像差（不需要取反！）
# 实际 OPD 和理想 OPD 使用相同的符号约定
aberration_waves = opd_waves - ideal_opd_waves

print(f"\n像差范围: {np.min(aberration_waves[valid_mask]):.2f} ~ {np.max(aberration_waves[valid_mask]):.2f} waves")
print(f"像差均值: {np.mean(aberration_waves[valid_mask]):.2f} waves")
print(f"像差标准差: {np.std(aberration_waves[valid_mask]):.2f} waves")


# ============================================================
# 详细分析中心和边缘光线
# ============================================================

print_section("详细光线分析")

# 找到中心光线和边缘光线
center_idx = n_rays // 2
edge_indices = [0, n_rays_1d - 1, n_rays - n_rays_1d, n_rays - 1]

print(f"\n中心光线 (idx={center_idx}):")
print(f"  位置: ({ray_x[center_idx]:.2f}, {ray_y[center_idx]:.2f}) mm")
print(f"  实际 OPD: {opd_waves[center_idx]:.4f} waves")
print(f"  理想 OPD: {ideal_opd_waves[center_idx]:.4f} waves")
print(f"  像差: {aberration_waves[center_idx]:.4f} waves")

for i, idx in enumerate(edge_indices):
    print(f"\n边缘光线 {i+1} (idx={idx}):")
    print(f"  位置: ({ray_x[idx]:.2f}, {ray_y[idx]:.2f}) mm")
    print(f"  r² = {ray_r_sq[idx]:.2f} mm²")
    print(f"  实际 OPD: {opd_waves[idx]:.4f} waves")
    print(f"  理想 OPD: {ideal_opd_waves[idx]:.4f} waves")
    print(f"  像差: {aberration_waves[idx]:.4f} waves")


# ============================================================
# 检查公式
# ============================================================

print_section("公式验证")

# 对于凸面镜（f < 0），理想 OPD 应该是负的
# OPD_ideal = r² / (2f) / λ
# 当 f = -50 mm, r = 10 mm, λ = 0.01064 mm 时：
# OPD_ideal = 100 / (2 * -50) / 0.01064 = 100 / -100 / 0.01064 = -93.98 waves

r_test = 10.0  # mm
opd_test = r_test**2 / (2 * focal_length_mm * wavelength_mm)
print(f"测试计算:")
print(f"  r = {r_test} mm")
print(f"  f = {focal_length_mm} mm")
print(f"  λ = {wavelength_mm} mm")
print(f"  OPD_ideal = r² / (2f) / λ = {r_test**2} / (2 * {focal_length_mm}) / {wavelength_mm}")
print(f"           = {r_test**2} / {2 * focal_length_mm} / {wavelength_mm}")
print(f"           = {r_test**2 / (2 * focal_length_mm)} / {wavelength_mm}")
print(f"           = {opd_test:.2f} waves")

print(f"\n注意：对于凸面镜（f < 0），理想 OPD 是负的，表示边缘光程短于中心")


# ============================================================
# 检查 ElementRaytracer 的 OPD 计算
# ============================================================

print_section("ElementRaytracer OPD 详细分析")

# 获取原始 OPD（mm）
rays_opd_mm = np.asarray(rays_out.opd)
chief_opd_mm = raytracer._chief_ray_opd

print(f"主光线 OPD: {chief_opd_mm:.6f} mm")
print(f"光线 OPD 范围: {np.min(rays_opd_mm):.6f} ~ {np.max(rays_opd_mm):.6f} mm")

# 相对 OPD（mm）
relative_opd_mm = rays_opd_mm - chief_opd_mm
print(f"相对 OPD 范围: {np.min(relative_opd_mm):.6f} ~ {np.max(relative_opd_mm):.6f} mm")

# 转换为波长数
relative_opd_waves = relative_opd_mm / wavelength_mm
print(f"相对 OPD (waves): {np.min(relative_opd_waves):.2f} ~ {np.max(relative_opd_waves):.2f} waves")

# 与 get_relative_opd_waves() 对比
print(f"\nget_relative_opd_waves() 结果: {np.min(opd_waves):.2f} ~ {np.max(opd_waves):.2f} waves")


# ============================================================
# 问题诊断
# ============================================================

print_section("问题诊断")

print("""
可能的问题：

1. 理想 OPD 公式对于凸面镜（f < 0）的符号处理
   - 凸面镜发散光束，边缘光程应该比中心短
   - 理想 OPD = r² / (2f) / λ，当 f < 0 时，OPD < 0
   
2. ElementRaytracer 的 OPD 符号约定
   - 需要确认 optiland 的 OPD 符号约定
   
3. 像差计算公式
   - aberration = actual_OPD - ideal_OPD
   - 需要确保符号一致
   
4. 相位转换
   - phase = -2π * aberration
   - 正 OPD（光程长）对应负相位（波前滞后）
""")

# 检查实际 OPD 的符号
print(f"\n实际 OPD 符号分析:")
print(f"  中心光线 OPD: {opd_waves[center_idx]:.4f} waves")
print(f"  边缘光线 OPD: {opd_waves[edge_indices[0]]:.4f} waves")
print(f"  差值（边缘-中心）: {opd_waves[edge_indices[0]] - opd_waves[center_idx]:.4f} waves")

print(f"\n理想 OPD 符号分析:")
print(f"  中心光线 OPD: {ideal_opd_waves[center_idx]:.4f} waves")
print(f"  边缘光线 OPD: {ideal_opd_waves[edge_indices[0]]:.4f} waves")
print(f"  差值（边缘-中心）: {ideal_opd_waves[edge_indices[0]] - ideal_opd_waves[center_idx]:.4f} waves")

