"""
调试抛物面镜 OPD 计算

分析抛物面镜的 OPD 计算是否正确
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
f = -50.0  # mm, 凸面抛物面镜焦距
R = 2 * f  # mm, 顶点曲率半径

print_section("测试参数")
print(f"波长: {wavelength_um} μm = {wavelength_mm} mm")
print(f"焦距: {f} mm")
print(f"顶点曲率半径: {R} mm")


# ============================================================
# 测试 1: 抛物面镜（无倾斜）
# ============================================================

print_section("测试 1: 抛物面镜（conic=-1，无倾斜）")

surface_parabolic = SurfaceDefinition(
    surface_type='mirror',
    radius=R,
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
    surfaces=[surface_parabolic],
    wavelength=wavelength_um,
)

rays_out = raytracer.trace(rays_in)
opd_waves = raytracer.get_relative_opd_waves()
valid_mask = raytracer.get_valid_ray_mask()

print(f"\n光线追迹结果:")
print(f"  有效光线数量: {np.sum(valid_mask)}")
print(f"  相对 OPD 范围: {np.min(opd_waves[valid_mask]):.4f} ~ {np.max(opd_waves[valid_mask]):.4f} waves")
print(f"  相对 OPD 均值: {np.mean(opd_waves[valid_mask]):.4f} waves")
print(f"  相对 OPD 标准差: {np.std(opd_waves[valid_mask]):.4f} waves")

# 获取原始 OPD（mm）
rays_opd_mm = np.asarray(rays_out.opd)
print(f"\n原始 OPD (mm):")
print(f"  范围: {np.min(rays_opd_mm):.6f} ~ {np.max(rays_opd_mm):.6f} mm")


# ============================================================
# 分析：抛物面镜的理论 OPD
# ============================================================

print_section("理论分析：抛物面镜的 OPD")

print("""
对于抛物面镜，平行光入射后会聚焦到焦点。
抛物面的定义特性是：从无穷远来的平行光，经过反射后，
所有光线到达焦点的光程相等。

这意味着：
1. 所有光线的总光程（入射 + 反射到焦点）相等
2. 但 ElementRaytracer 计算的是到出射面的 OPD，不是到焦点的 OPD

关键问题：ElementRaytracer 的出射面在哪里？
- 出射面在元件顶点位置（z=0）
- 光线反射后，不同位置的光线到达出射面的距离不同
- 这就是为什么 OPD 不为零
""")

# 计算理论 OPD
# 对于抛物面 z = r² / (4f)，反射后光线方向改变
# 入射光线从 z = -∞ 到表面的光程 = z_surface
# 反射后从表面到出射面（z=0）的光程取决于反射角

# 简化分析：计算表面矢高
r_sq = ray_x**2 + ray_y**2
sag = r_sq / (4 * f)  # 抛物面矢高公式（注意 f 是负的）

print(f"\n表面矢高分析:")
print(f"  r² 范围: {np.min(r_sq):.2f} ~ {np.max(r_sq):.2f} mm²")
print(f"  矢高范围: {np.min(sag):.4f} ~ {np.max(sag):.4f} mm")
print(f"  矢高转换为波长数: {np.min(sag)/wavelength_mm:.2f} ~ {np.max(sag)/wavelength_mm:.2f} waves")


# ============================================================
# 测试 2: 球面镜对比
# ============================================================

print_section("测试 2: 球面镜（conic=0）对比")

surface_spherical = SurfaceDefinition(
    surface_type='mirror',
    radius=R,
    thickness=0.0,
    material='mirror',
    semi_aperture=20.0,
    conic=0.0,  # 球面
    tilt_x=0.0,
    tilt_y=0.0,
)

raytracer_sphere = ElementRaytracer(
    surfaces=[surface_spherical],
    wavelength=wavelength_um,
)

rays_out_sphere = raytracer_sphere.trace(rays_in)
opd_waves_sphere = raytracer_sphere.get_relative_opd_waves()
valid_mask_sphere = raytracer_sphere.get_valid_ray_mask()

print(f"\n光线追迹结果:")
print(f"  相对 OPD 范围: {np.min(opd_waves_sphere[valid_mask_sphere]):.4f} ~ {np.max(opd_waves_sphere[valid_mask_sphere]):.4f} waves")
print(f"  相对 OPD 标准差: {np.std(opd_waves_sphere[valid_mask_sphere]):.4f} waves")


# ============================================================
# 测试 3: 平面镜对比
# ============================================================

print_section("测试 3: 平面镜对比")

surface_flat = SurfaceDefinition(
    surface_type='mirror',
    radius=np.inf,
    thickness=0.0,
    material='mirror',
    semi_aperture=20.0,
    conic=0.0,
    tilt_x=0.0,
    tilt_y=0.0,
)

raytracer_flat = ElementRaytracer(
    surfaces=[surface_flat],
    wavelength=wavelength_um,
)

rays_out_flat = raytracer_flat.trace(rays_in)
opd_waves_flat = raytracer_flat.get_relative_opd_waves()
valid_mask_flat = raytracer_flat.get_valid_ray_mask()

print(f"\n光线追迹结果:")
print(f"  相对 OPD 范围: {np.min(opd_waves_flat[valid_mask_flat]):.6f} ~ {np.max(opd_waves_flat[valid_mask_flat]):.6f} waves")
print(f"  相对 OPD 标准差: {np.std(opd_waves_flat[valid_mask_flat]):.6f} waves")


# ============================================================
# 结论
# ============================================================

print_section("结论")

print("""
ElementRaytracer 计算的 OPD 是从入射面到出射面的总光程差。
对于曲面镜，这个 OPD 包含了：
1. 入射光线到表面的光程（与表面矢高相关）
2. 反射后从表面到出射面的光程

对于平面镜，OPD 应该接近零（因为所有光线的光程相等）。
对于曲面镜，OPD 不为零，因为不同位置的光线经历不同的光程。

关键洞察：
- ElementRaytracer 的 OPD 已经包含了元件的聚焦效果
- 理想 OPD 公式 r²/(2f)/λ 是 PROPER prop_lens 使用的相位公式
- 两者应该匹配，但由于计算方式不同，可能存在差异

正确的做法：
- 对于理想抛物面镜，像差应该为零
- 像差 = 实际 OPD - 理想 OPD
- 如果理想 OPD 公式不准确，需要使用更精确的公式
""")

# 计算理想 OPD 并比较
ideal_opd_waves = r_sq / (2 * f * wavelength_mm)
aberration = opd_waves - ideal_opd_waves

print(f"\n像差分析（使用 r²/(2f)/λ 公式）:")
print(f"  理想 OPD 范围: {np.min(ideal_opd_waves):.4f} ~ {np.max(ideal_opd_waves):.4f} waves")
print(f"  像差范围: {np.min(aberration[valid_mask]):.4f} ~ {np.max(aberration[valid_mask]):.4f} waves")
print(f"  像差 RMS: {np.std(aberration[valid_mask]):.4f} waves")

# 尝试使用实际 OPD 作为理想 OPD（即像差为零）
print(f"\n如果直接使用 ElementRaytracer 的 OPD 作为理想 OPD:")
print(f"  像差 = 0（因为实际 OPD = 理想 OPD）")
print(f"  这意味着不需要应用任何像差相位")
