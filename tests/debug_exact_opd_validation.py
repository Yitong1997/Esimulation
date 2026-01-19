"""
调试：验证精确 OPD 公式与 ElementRaytracer 的一致性
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_exact_mirror_opd(r_sq, focal_length_mm):
    """计算反射镜的精确 OPD（与 system.py 中的方法相同）"""
    f = focal_length_mm
    
    # 表面矢高
    sag = r_sq / (4 * f)
    
    # 归一化因子的平方
    n_mag_sq = 1 + r_sq / (4 * f**2)
    
    # 反射方向 z 分量
    rz = 1 - 2 / n_mag_sq
    
    # 入射光程
    incident_path = sag
    
    # 反射光程
    reflected_path = np.abs(sag / rz)
    
    # 总光程
    total_path = incident_path + reflected_path
    
    return total_path


# ============================================================
# 测试参数
# ============================================================

wavelength_um = 10.64
wavelength_mm = wavelength_um * 1e-3

print_section("测试参数")
print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")


# ============================================================
# 测试 1: 凹面抛物面镜（f = 50 mm）
# ============================================================

print_section("测试 1: 凹面抛物面镜（f = 50 mm）")

f = 50.0  # mm
R = 2 * f  # mm
r_max = 10.0  # mm

print(f"焦距: {f} mm")
print(f"曲率半径: {R} mm")
print(f"最大半径: {r_max} mm")

# 创建测试光线
n_rays_1d = 21
ray_coords = np.linspace(-r_max, r_max, n_rays_1d)
ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
ray_x = ray_X.flatten()
ray_y = ray_Y.flatten()
n_rays = len(ray_x)
r_sq = ray_x**2 + ray_y**2

# ElementRaytracer
surface = SurfaceDefinition(
    surface_type='mirror',
    radius=R,
    thickness=0.0,
    material='mirror',
    semi_aperture=r_max * 1.1,
    conic=-1.0,  # 抛物面
    tilt_x=0.0,
    tilt_y=0.0,
)

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
    surfaces=[surface],
    wavelength=wavelength_um,
)

rays_out = raytracer.trace(rays_in)
opd_waves_raytracer = raytracer.get_relative_opd_waves()
valid_mask = raytracer.get_valid_ray_mask()

# 精确公式
opd_mm_exact = calculate_exact_mirror_opd(r_sq, f)
# 相对于中心的 OPD
center_idx = n_rays // 2
opd_mm_exact_relative = opd_mm_exact - opd_mm_exact[center_idx]
opd_waves_exact = opd_mm_exact_relative / wavelength_mm

# 比较
diff_waves = opd_waves_raytracer - opd_waves_exact
diff_waves_valid = diff_waves[valid_mask]

print(f"\n结果比较:")
print(f"  ElementRaytracer OPD 范围: {np.min(opd_waves_raytracer[valid_mask]):.4f} ~ {np.max(opd_waves_raytracer[valid_mask]):.4f} waves")
print(f"  精确公式 OPD 范围:         {np.min(opd_waves_exact[valid_mask]):.4f} ~ {np.max(opd_waves_exact[valid_mask]):.4f} waves")
print(f"  差异范围:                  {np.min(diff_waves_valid):.6f} ~ {np.max(diff_waves_valid):.6f} waves")
print(f"  差异 RMS:                  {np.std(diff_waves_valid):.6f} waves")
print(f"  差异 PV:                   {np.max(diff_waves_valid) - np.min(diff_waves_valid):.6f} waves")


# ============================================================
# 测试 2: 凸面抛物面镜（f = -50 mm）
# ============================================================

print_section("测试 2: 凸面抛物面镜（f = -50 mm）")

f = -50.0  # mm
R = 2 * f  # mm

print(f"焦距: {f} mm")
print(f"曲率半径: {R} mm")
print(f"最大半径: {r_max} mm")

# ElementRaytracer
surface_convex = SurfaceDefinition(
    surface_type='mirror',
    radius=R,
    thickness=0.0,
    material='mirror',
    semi_aperture=r_max * 1.1,
    conic=-1.0,  # 抛物面
    tilt_x=0.0,
    tilt_y=0.0,
)

raytracer_convex = ElementRaytracer(
    surfaces=[surface_convex],
    wavelength=wavelength_um,
)

rays_out_convex = raytracer_convex.trace(rays_in)
opd_waves_raytracer_convex = raytracer_convex.get_relative_opd_waves()
valid_mask_convex = raytracer_convex.get_valid_ray_mask()

# 精确公式
opd_mm_exact_convex = calculate_exact_mirror_opd(r_sq, f)
opd_mm_exact_relative_convex = opd_mm_exact_convex - opd_mm_exact_convex[center_idx]
opd_waves_exact_convex = opd_mm_exact_relative_convex / wavelength_mm

# 比较
diff_waves_convex = opd_waves_raytracer_convex - opd_waves_exact_convex
diff_waves_valid_convex = diff_waves_convex[valid_mask_convex]

print(f"\n结果比较:")
print(f"  ElementRaytracer OPD 范围: {np.min(opd_waves_raytracer_convex[valid_mask_convex]):.4f} ~ {np.max(opd_waves_raytracer_convex[valid_mask_convex]):.4f} waves")
print(f"  精确公式 OPD 范围:         {np.min(opd_waves_exact_convex[valid_mask_convex]):.4f} ~ {np.max(opd_waves_exact_convex[valid_mask_convex]):.4f} waves")
print(f"  差异范围:                  {np.min(diff_waves_valid_convex):.6f} ~ {np.max(diff_waves_valid_convex):.6f} waves")
print(f"  差异 RMS:                  {np.std(diff_waves_valid_convex):.6f} waves")
print(f"  差异 PV:                   {np.max(diff_waves_valid_convex) - np.min(diff_waves_valid_convex):.6f} waves")


# ============================================================
# 结论
# ============================================================

print_section("结论")

if np.std(diff_waves_valid) < 0.01 and np.std(diff_waves_valid_convex) < 0.01:
    print("✓ ElementRaytracer 与精确公式一致（差异 < 0.01 waves）")
    print("  光线追迹精度验证通过！")
else:
    print("✗ ElementRaytracer 与精确公式存在差异")
    print(f"  凹面镜差异 RMS: {np.std(diff_waves_valid):.6f} waves")
    print(f"  凸面镜差异 RMS: {np.std(diff_waves_valid_convex):.6f} waves")
    print("  需要检查精确公式或 ElementRaytracer 的实现")
