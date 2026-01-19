"""
调试：修正凸面镜的精确 OPD 公式
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
# 分析凸面镜的几何
# ============================================================

print_section("分析凸面镜的几何")

print("""
对于凸面镜（f < 0），表面向入射方向凸出：
- 矢高 sag = r²/(4f) < 0（因为 f < 0）
- 表面在 z < 0 的位置

光程计算：
1. 入射光程：从 z=0 到表面 = |sag|（光线向 -z 方向走）
2. 反射后光程：从表面回到 z=0

关键点：
- 对于凸面镜，入射光程是 |sag| = -sag（因为 sag < 0）
- 反射后光线方向也需要正确计算

让我们重新推导精确公式...
""")


def calculate_exact_mirror_opd_v2(r_sq, focal_length_mm):
    """计算反射镜的精确 OPD（修正版）
    
    对于凹面镜（f > 0）和凸面镜（f < 0）都适用。
    """
    f = focal_length_mm
    
    # 表面矢高（凹面镜 > 0，凸面镜 < 0）
    sag = r_sq / (4 * f)
    
    # 归一化因子的平方
    n_mag_sq = 1 + r_sq / (4 * f**2)
    
    # 反射方向 z 分量
    # 对于凹面镜：rz < 0（反射后向 -z 方向）
    # 对于凸面镜：rz < 0（反射后也向 -z 方向）
    rz = 1 - 2 / n_mag_sq
    
    # 入射光程：从 z=0 到表面
    # 对于凹面镜：sag > 0，入射光程 = sag
    # 对于凸面镜：sag < 0，入射光程 = |sag| = -sag
    incident_path = np.abs(sag)
    
    # 反射光程：从表面到 z=0
    # t = -sag / rz
    # 对于凹面镜：sag > 0, rz < 0 → t > 0
    # 对于凸面镜：sag < 0, rz < 0 → t < 0
    t = -sag / rz
    reflected_path = np.abs(t)
    
    # 总光程
    total_path = incident_path + reflected_path
    
    return total_path


# ============================================================
# 测试凸面镜
# ============================================================

print_section("测试凸面镜（f = -50 mm）")

wavelength_um = 10.64
wavelength_mm = wavelength_um * 1e-3

f = -50.0  # mm
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

# 精确公式 v2
opd_mm_exact = calculate_exact_mirror_opd_v2(r_sq, f)
center_idx = n_rays // 2
opd_mm_exact_relative = opd_mm_exact - opd_mm_exact[center_idx]
opd_waves_exact = opd_mm_exact_relative / wavelength_mm

# 比较
diff_waves = opd_waves_raytracer - opd_waves_exact
diff_waves_valid = diff_waves[valid_mask]

print(f"\n结果比较:")
print(f"  ElementRaytracer OPD 范围: {np.min(opd_waves_raytracer[valid_mask]):.4f} ~ {np.max(opd_waves_raytracer[valid_mask]):.4f} waves")
print(f"  精确公式 v2 OPD 范围:      {np.min(opd_waves_exact[valid_mask]):.4f} ~ {np.max(opd_waves_exact[valid_mask]):.4f} waves")
print(f"  差异范围:                  {np.min(diff_waves_valid):.6f} ~ {np.max(diff_waves_valid):.6f} waves")
print(f"  差异 RMS:                  {np.std(diff_waves_valid):.6f} waves")


# ============================================================
# 详细分析：逐点比较
# ============================================================

print_section("详细分析：逐点比较（沿 x 轴）")

# 选择 y=0 的点
y_zero_mask = np.abs(ray_y) < 0.1
x_values = ray_x[y_zero_mask]
sort_idx = np.argsort(x_values)
x_sorted = x_values[sort_idx]

opd_rt = opd_waves_raytracer[y_zero_mask][sort_idx]
opd_ex = opd_waves_exact[y_zero_mask][sort_idx]

print(f"{'x (mm)':>10} | {'Raytracer':>12} | {'精确公式':>12} | {'差异':>12}")
print("-" * 55)
for i in range(len(x_sorted)):
    diff = opd_rt[i] - opd_ex[i]
    print(f"{x_sorted[i]:>10.2f} | {opd_rt[i]:>12.4f} | {opd_ex[i]:>12.4f} | {diff:>12.6f}")


# ============================================================
# 分析：为什么还有差异？
# ============================================================

print_section("分析：为什么还有差异？")

print("""
观察结果：
- ElementRaytracer 的 OPD 是负的（边缘光程短于中心）
- 精确公式 v2 的 OPD 是正的（边缘光程长于中心）

这说明精确公式的符号有问题。

让我们重新思考凸面镜的光程：
- 凸面镜向入射方向凸出（sag < 0）
- 边缘光线到达表面的距离比中心光线短
- 反射后，边缘光线回到 z=0 的距离也比中心光线短
- 因此，边缘光线的总光程比中心光线短
- 相对 OPD 应该是负的

问题在于：精确公式计算的是绝对光程，而不是相对于参考面的 OPD。
""")


# ============================================================
# 正确的理解：ElementRaytracer 的 OPD 定义
# ============================================================

print_section("正确的理解：ElementRaytracer 的 OPD 定义")

print("""
ElementRaytracer 计算的 OPD 是：
- 从入射面（z=0）到出射面（z=0）的总光程
- 相对于主光线（中心光线）的光程差

对于凸面镜：
- 边缘光线的光程比中心光线短
- 因此相对 OPD 是负的

对于凹面镜：
- 边缘光线的光程比中心光线长
- 因此相对 OPD 是正的

这与 PROPER 的 prop_lens 使用的公式一致：
- OPD = r²/(2f)
- 对于凹面镜（f > 0），OPD > 0
- 对于凸面镜（f < 0），OPD < 0

结论：
- ElementRaytracer 的 OPD 计算是正确的
- 精确公式需要考虑符号
""")


# ============================================================
# 最终修正的精确公式
# ============================================================

def calculate_exact_mirror_opd_final(r_sq, focal_length_mm):
    """计算反射镜的精确 OPD（最终版）
    
    返回相对于中心光线的 OPD，符号与 PROPER 一致：
    - 凹面镜（f > 0）：边缘 OPD > 0
    - 凸面镜（f < 0）：边缘 OPD < 0
    """
    f = focal_length_mm
    
    # 表面矢高
    sag = r_sq / (4 * f)
    
    # 归一化因子的平方
    n_mag_sq = 1 + r_sq / (4 * f**2)
    
    # 反射方向 z 分量
    rz = 1 - 2 / n_mag_sq
    
    # 入射光程（带符号）
    incident_path = sag
    
    # 反射光程（带符号）
    # t = -sag / rz
    reflected_path = -sag / rz
    
    # 总光程（带符号）
    total_path = incident_path + reflected_path
    
    # 相对于中心的 OPD
    # 中心光程 = 0（因为 r=0 时 sag=0）
    opd = total_path
    
    return opd


print_section("测试最终修正的精确公式")

opd_mm_final = calculate_exact_mirror_opd_final(r_sq, f)
opd_waves_final = opd_mm_final / wavelength_mm

diff_final = opd_waves_raytracer - opd_waves_final
diff_final_valid = diff_final[valid_mask]

print(f"凸面镜（f = {f} mm）:")
print(f"  ElementRaytracer OPD 范围: {np.min(opd_waves_raytracer[valid_mask]):.4f} ~ {np.max(opd_waves_raytracer[valid_mask]):.4f} waves")
print(f"  精确公式最终版 OPD 范围:   {np.min(opd_waves_final[valid_mask]):.4f} ~ {np.max(opd_waves_final[valid_mask]):.4f} waves")
print(f"  差异 RMS:                  {np.std(diff_final_valid):.6f} waves")

# 测试凹面镜
f_concave = 50.0
opd_mm_final_concave = calculate_exact_mirror_opd_final(r_sq, f_concave)
opd_waves_final_concave = opd_mm_final_concave / wavelength_mm

surface_concave = SurfaceDefinition(
    surface_type='mirror',
    radius=2*f_concave,
    thickness=0.0,
    material='mirror',
    semi_aperture=r_max * 1.1,
    conic=-1.0,
    tilt_x=0.0,
    tilt_y=0.0,
)

raytracer_concave = ElementRaytracer(
    surfaces=[surface_concave],
    wavelength=wavelength_um,
)

rays_out_concave = raytracer_concave.trace(rays_in)
opd_waves_raytracer_concave = raytracer_concave.get_relative_opd_waves()
valid_mask_concave = raytracer_concave.get_valid_ray_mask()

diff_concave = opd_waves_raytracer_concave - opd_waves_final_concave
diff_concave_valid = diff_concave[valid_mask_concave]

print(f"\n凹面镜（f = {f_concave} mm）:")
print(f"  ElementRaytracer OPD 范围: {np.min(opd_waves_raytracer_concave[valid_mask_concave]):.4f} ~ {np.max(opd_waves_raytracer_concave[valid_mask_concave]):.4f} waves")
print(f"  精确公式最终版 OPD 范围:   {np.min(opd_waves_final_concave[valid_mask_concave]):.4f} ~ {np.max(opd_waves_final_concave[valid_mask_concave]):.4f} waves")
print(f"  差异 RMS:                  {np.std(diff_concave_valid):.6f} waves")


# ============================================================
# 结论
# ============================================================

print_section("结论")

if np.std(diff_final_valid) < 0.01 and np.std(diff_concave_valid) < 0.01:
    print("✓ 精确公式最终版与 ElementRaytracer 一致（差异 < 0.01 waves）")
    print("  可以使用此公式作为理想 OPD 计算")
else:
    print("✗ 仍有差异，需要进一步分析")
    print(f"  凸面镜差异 RMS: {np.std(diff_final_valid):.6f} waves")
    print(f"  凹面镜差异 RMS: {np.std(diff_concave_valid):.6f} waves")
