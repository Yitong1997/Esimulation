"""
调试 OAP 像差计算

分析抛物面镜的像差是否正常
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from sequential_system import (
    ParabolicMirror,
)
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 测试 1: 无倾斜的抛物面镜
# ============================================================

print_section("测试 1: 无倾斜的抛物面镜")

wavelength_um = 10.64
wavelength_mm = wavelength_um * 1e-3
f1 = -50.0  # mm, 凸面抛物面镜

# 创建无倾斜的抛物面镜
oap_no_tilt = ParabolicMirror(
    parent_focal_length=f1,
    thickness=50.0,
    semi_aperture=20.0,
    off_axis_distance=100.0,
    tilt_x=0.0,  # 无倾斜
    name="OAP_no_tilt",
)

surface_def = oap_no_tilt.get_surface_definition()
print(f"曲率半径: {surface_def.radius} mm")
print(f"圆锥常数: {surface_def.conic}")
print(f"倾斜: ({np.degrees(surface_def.tilt_x):.1f}°, {np.degrees(surface_def.tilt_y):.1f}°)")

# 创建测试光线
n_rays_1d = 5
half_size = 10.0
ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
ray_x = ray_X.flatten()
ray_y = ray_Y.flatten()
n_rays = len(ray_x)

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

raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=wavelength_um,
)

rays_out = raytracer.trace(rays_in)
opd_waves = raytracer.get_relative_opd_waves()
valid_mask = raytracer.get_valid_ray_mask()

print(f"\n实际 OPD 范围: {np.min(opd_waves[valid_mask]):.4f} ~ {np.max(opd_waves[valid_mask]):.4f} waves")

# 计算理想 OPD
ray_r_sq = ray_x**2 + ray_y**2
focal_length_mm = oap_no_tilt.focal_length
ideal_opd_waves = ray_r_sq / (2 * focal_length_mm * wavelength_mm)

print(f"理想 OPD 范围: {np.min(ideal_opd_waves):.4f} ~ {np.max(ideal_opd_waves):.4f} waves")

# 像差
aberration_waves = opd_waves - ideal_opd_waves
print(f"像差范围: {np.min(aberration_waves[valid_mask]):.4f} ~ {np.max(aberration_waves[valid_mask]):.4f} waves")
print(f"像差 RMS: {np.std(aberration_waves[valid_mask]):.4f} waves")


# ============================================================
# 测试 2: 有倾斜的抛物面镜
# ============================================================

print_section("测试 2: 有 45° 倾斜的抛物面镜")

oap_with_tilt = ParabolicMirror(
    parent_focal_length=f1,
    thickness=50.0,
    semi_aperture=20.0,
    off_axis_distance=100.0,
    tilt_x=np.pi/4,  # 45° 倾斜
    name="OAP_with_tilt",
)

surface_def_tilt = oap_with_tilt.get_surface_definition()
print(f"曲率半径: {surface_def_tilt.radius} mm")
print(f"圆锥常数: {surface_def_tilt.conic}")
print(f"倾斜: ({np.degrees(surface_def_tilt.tilt_x):.1f}°, {np.degrees(surface_def_tilt.tilt_y):.1f}°)")

raytracer_tilt = ElementRaytracer(
    surfaces=[surface_def_tilt],
    wavelength=wavelength_um,
)

rays_out_tilt = raytracer_tilt.trace(rays_in)
opd_waves_tilt = raytracer_tilt.get_relative_opd_waves()
valid_mask_tilt = raytracer_tilt.get_valid_ray_mask()

print(f"\n实际 OPD 范围: {np.min(opd_waves_tilt[valid_mask_tilt]):.4f} ~ {np.max(opd_waves_tilt[valid_mask_tilt]):.4f} waves")

# 像差（使用相同的理想 OPD）
aberration_waves_tilt = opd_waves_tilt - ideal_opd_waves
print(f"像差范围: {np.min(aberration_waves_tilt[valid_mask_tilt]):.4f} ~ {np.max(aberration_waves_tilt[valid_mask_tilt]):.4f} waves")
print(f"像差 RMS: {np.std(aberration_waves_tilt[valid_mask_tilt]):.4f} waves")


# ============================================================
# 测试 3: 球面镜（无圆锥常数）
# ============================================================

print_section("测试 3: 球面镜（conic=0）对比")

# 创建球面镜的 SurfaceDefinition
surface_def_sphere = SurfaceDefinition(
    surface_type='mirror',
    radius=2 * f1,  # 与抛物面镜相同的曲率半径
    thickness=0.0,
    material='mirror',
    semi_aperture=20.0,
    conic=0.0,  # 球面
    tilt_x=0.0,
    tilt_y=0.0,
)

print(f"曲率半径: {surface_def_sphere.radius} mm")
print(f"圆锥常数: {surface_def_sphere.conic}")

raytracer_sphere = ElementRaytracer(
    surfaces=[surface_def_sphere],
    wavelength=wavelength_um,
)

rays_out_sphere = raytracer_sphere.trace(rays_in)
opd_waves_sphere = raytracer_sphere.get_relative_opd_waves()
valid_mask_sphere = raytracer_sphere.get_valid_ray_mask()

print(f"\n实际 OPD 范围: {np.min(opd_waves_sphere[valid_mask_sphere]):.4f} ~ {np.max(opd_waves_sphere[valid_mask_sphere]):.4f} waves")

# 像差
aberration_waves_sphere = opd_waves_sphere - ideal_opd_waves
print(f"像差范围: {np.min(aberration_waves_sphere[valid_mask_sphere]):.4f} ~ {np.max(aberration_waves_sphere[valid_mask_sphere]):.4f} waves")
print(f"像差 RMS: {np.std(aberration_waves_sphere[valid_mask_sphere]):.4f} waves")


# ============================================================
# 分析
# ============================================================

print_section("分析")

print("""
抛物面镜（conic=-1）对于轴上平行光入射应该是无像差的。
如果像差很大，可能的原因：

1. 离轴抛物面镜（OAP）的特性
   - OAP 只对特定入射角的平行光无像差
   - 对于正入射（沿光轴），OAP 会有像差

2. 理想 OPD 公式的适用性
   - r² / (2f) / λ 是薄透镜/球面镜的近似公式
   - 对于抛物面镜，可能需要不同的公式

3. 坐标系问题
   - 光线追迹的坐标系与理想 OPD 公式的坐标系可能不一致
""")

# 检查抛物面镜的理论像差
print("\n理论分析:")
print(f"抛物面镜焦距: f = {f1} mm")
print(f"曲率半径: R = 2f = {2*f1} mm")
print(f"离轴距离: d = {oap_no_tilt.off_axis_distance} mm")

# 对于离轴抛物面镜，入射角应该等于离轴角
# 离轴角 = arctan(d / (2f)) 对于 90° OAP
# 但这里 d = 2|f|，所以离轴角 = 45°
off_axis_angle = np.arctan(oap_no_tilt.off_axis_distance / (2 * abs(f1)))
print(f"离轴角: {np.degrees(off_axis_angle):.1f}°")

print("\n对于 90° OAP（离轴角 45°），入射光应该以 45° 角入射才能无像差。")
print("当前测试使用正入射（沿 Z 轴），所以会有像差。")
