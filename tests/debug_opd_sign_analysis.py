"""
调试 OPD 符号分析

详细分析 ElementRaytracer 的 OPD 符号与理想 OPD 公式的关系
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


# ============================================================
# 测试参数
# ============================================================

wavelength_um = 10.64
wavelength_mm = wavelength_um * 1e-3

print_section("测试参数")
print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")


# ============================================================
# 测试 1: 凹面镜（f > 0）
# ============================================================

print_section("测试 1: 凹面镜（f > 0）")

f_concave = 50.0  # mm, 凹面镜焦距（正值）
R_concave = 2 * f_concave  # mm, 顶点曲率半径

print(f"焦距: {f_concave} mm")
print(f"顶点曲率半径: {R_concave} mm")

surface_concave = SurfaceDefinition(
    surface_type='mirror',
    radius=R_concave,
    thickness=0.0,
    material='mirror',
    semi_aperture=20.0,
    conic=-1.0,  # 抛物面
    tilt_x=0.0,
    tilt_y=0.0,
)

# 创建测试光线
n_rays_1d = 5
half_size = 10.0
ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
ray_x = ray_X.flatten()
ray_y = ray_Y.flatten()
n_rays = len(ray_x)
r_sq = ray_x**2 + ray_y**2

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

raytracer_concave = ElementRaytracer(
    surfaces=[surface_concave],
    wavelength=wavelength_um,
)

rays_out_concave = raytracer_concave.trace(rays_in)
opd_waves_concave = raytracer_concave.get_relative_opd_waves()
valid_mask_concave = raytracer_concave.get_valid_ray_mask()

# 理想 OPD（使用 r²/(2f)/λ 公式）
ideal_opd_waves_concave = r_sq / (2 * f_concave * wavelength_mm)

print(f"\nElementRaytracer 结果:")
print(f"  相对 OPD 范围: {np.min(opd_waves_concave[valid_mask_concave]):.4f} ~ {np.max(opd_waves_concave[valid_mask_concave]):.4f} waves")
print(f"  边缘 OPD（r=10mm）: {opd_waves_concave[0]:.4f} waves")

print(f"\n理想 OPD（r²/(2f)/λ）:")
print(f"  范围: {np.min(ideal_opd_waves_concave):.4f} ~ {np.max(ideal_opd_waves_concave):.4f} waves")
print(f"  边缘 OPD（r=10mm）: {ideal_opd_waves_concave[0]:.4f} waves")

aberration_concave = opd_waves_concave - ideal_opd_waves_concave
print(f"\n像差（实际 - 理想）:")
print(f"  范围: {np.min(aberration_concave[valid_mask_concave]):.4f} ~ {np.max(aberration_concave[valid_mask_concave]):.4f} waves")
print(f"  RMS: {np.std(aberration_concave[valid_mask_concave]):.4f} waves")


# ============================================================
# 测试 2: 凸面镜（f < 0）
# ============================================================

print_section("测试 2: 凸面镜（f < 0）")

f_convex = -50.0  # mm, 凸面镜焦距（负值）
R_convex = 2 * f_convex  # mm, 顶点曲率半径

print(f"焦距: {f_convex} mm")
print(f"顶点曲率半径: {R_convex} mm")

surface_convex = SurfaceDefinition(
    surface_type='mirror',
    radius=R_convex,
    thickness=0.0,
    material='mirror',
    semi_aperture=20.0,
    conic=-1.0,  # 抛物面
    tilt_x=0.0,
    tilt_y=0.0,
)

raytracer_convex = ElementRaytracer(
    surfaces=[surface_convex],
    wavelength=wavelength_um,
)

rays_out_convex = raytracer_convex.trace(rays_in)
opd_waves_convex = raytracer_convex.get_relative_opd_waves()
valid_mask_convex = raytracer_convex.get_valid_ray_mask()

# 理想 OPD（使用 r²/(2f)/λ 公式）
ideal_opd_waves_convex = r_sq / (2 * f_convex * wavelength_mm)

print(f"\nElementRaytracer 结果:")
print(f"  相对 OPD 范围: {np.min(opd_waves_convex[valid_mask_convex]):.4f} ~ {np.max(opd_waves_convex[valid_mask_convex]):.4f} waves")
print(f"  边缘 OPD（r=10mm）: {opd_waves_convex[0]:.4f} waves")

print(f"\n理想 OPD（r²/(2f)/λ）:")
print(f"  范围: {np.min(ideal_opd_waves_convex):.4f} ~ {np.max(ideal_opd_waves_convex):.4f} waves")
print(f"  边缘 OPD（r=10mm）: {ideal_opd_waves_convex[0]:.4f} waves")

aberration_convex = opd_waves_convex - ideal_opd_waves_convex
print(f"\n像差（实际 - 理想）:")
print(f"  范围: {np.min(aberration_convex[valid_mask_convex]):.4f} ~ {np.max(aberration_convex[valid_mask_convex]):.4f} waves")
print(f"  RMS: {np.std(aberration_convex[valid_mask_convex]):.4f} waves")


# ============================================================
# 分析：符号约定
# ============================================================

print_section("分析：符号约定")

print("""
符号约定分析：

1. 凹面镜（f > 0）：
   - 边缘光线光程比中心光线长（因为要走到凹面再反射回来）
   - 所以边缘 OPD 应该为正
   - 理想 OPD = r²/(2f)/λ > 0（因为 f > 0）
   
2. 凸面镜（f < 0）：
   - 边缘光线光程比中心光线短（因为凸面向外凸出）
   - 所以边缘 OPD 应该为负
   - 理想 OPD = r²/(2f)/λ < 0（因为 f < 0）

3. ElementRaytracer 的 OPD：
   - 应该与上述符号约定一致
   - 如果不一致，需要检查 ElementRaytracer 的实现
""")

print(f"\n验证结果:")
print(f"  凹面镜边缘 OPD: {opd_waves_concave[0]:.4f} waves（期望 > 0）")
print(f"  凸面镜边缘 OPD: {opd_waves_convex[0]:.4f} waves（期望 < 0）")

if opd_waves_concave[0] > 0:
    print("  ✓ 凹面镜 OPD 符号正确")
else:
    print("  ✗ 凹面镜 OPD 符号错误！")

if opd_waves_convex[0] < 0:
    print("  ✓ 凸面镜 OPD 符号正确")
else:
    print("  ✗ 凸面镜 OPD 符号错误！")


# ============================================================
# 结论
# ============================================================

print_section("结论")

print("""
根据分析结果：

1. 如果 ElementRaytracer 的 OPD 符号与理想 OPD 公式一致：
   - 像差 = 实际 OPD - 理想 OPD 应该接近零
   - 当前代码是正确的

2. 如果 ElementRaytracer 的 OPD 符号与理想 OPD 公式相反：
   - 需要在代码中取反 OPD
   - 或者修改理想 OPD 公式的符号

3. 对于伽利略 OAP 扩束镜：
   - OAP1 是凸面镜（f < 0），边缘 OPD 应该为负
   - OAP2 是凹面镜（f > 0），边缘 OPD 应该为正
   - 如果符号正确，像差应该很小
""")

# 检查像差是否足够小
if np.std(aberration_concave[valid_mask_concave]) < 0.1 and np.std(aberration_convex[valid_mask_convex]) < 0.1:
    print("✓ 像差 RMS < 0.1 waves，符号约定正确")
else:
    print("✗ 像差 RMS > 0.1 waves，需要检查符号约定")
    print(f"  凹面镜像差 RMS: {np.std(aberration_concave[valid_mask_concave]):.4f} waves")
    print(f"  凸面镜像差 RMS: {np.std(aberration_convex[valid_mask_convex]):.4f} waves")
