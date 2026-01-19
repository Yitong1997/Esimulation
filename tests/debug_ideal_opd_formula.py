"""
调试理想 OPD 公式

分析 ElementRaytracer 的 OPD 与理想 OPD 公式的差异
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
# 理论分析：反射镜的 OPD 计算
# ============================================================

print_section("理论分析：反射镜的 OPD 计算")

print("""
对于反射镜，光线从入射面（z=0）到表面，再反射回出射面（z=0）。

设表面矢高为 sag(r)，则：
1. 入射光程：从 z=0 到表面 = sag(r)（沿 +z 方向）
2. 反射后光程：从表面回到 z=0

对于平行光入射（沿 +z 方向），反射后光线方向取决于表面法向量。

对于抛物面 z = r²/(4f)：
- 表面法向量：n = (-∂z/∂x, -∂z/∂y, 1) / |n|
- 入射方向：d = (0, 0, 1)
- 反射方向：r = d - 2(d·n)n

关键点：
- 入射光程 = sag(r)
- 反射光程取决于反射角和到出射面的距离
- 总 OPD = 入射光程 + 反射光程 - 参考光程

对于 PROPER 的 prop_lens：
- 使用相位公式 φ = -k * r² / (2f)
- 对应 OPD = r² / (2f)（单位：长度）
- 这是薄透镜近似，假设光线在透镜平面上折射

对于反射镜的精确计算：
- 需要考虑表面矢高和反射角
- OPD = 2 * sag(r)（对于垂直入射的近似）
- 对于抛物面：sag = r²/(4f)，所以 OPD ≈ 2 * r²/(4f) = r²/(2f)

但这只是近似！精确计算需要考虑反射角的影响。
""")


# ============================================================
# 测试：比较不同公式
# ============================================================

print_section("测试：比较不同公式")

# 创建测试光线
n_rays_1d = 11
half_size = 10.0
ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
ray_x = ray_X.flatten()
ray_y = ray_Y.flatten()
n_rays = len(ray_x)
r_sq = ray_x**2 + ray_y**2

# 抛物面镜
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

# 获取原始 OPD（mm）
rays_opd_mm = np.asarray(rays_out.opd)
chief_opd_mm = rays_opd_mm[n_rays // 2]  # 中心光线
relative_opd_mm = rays_opd_mm - chief_opd_mm

print(f"ElementRaytracer 结果:")
print(f"  主光线 OPD: {chief_opd_mm:.6f} mm")
print(f"  相对 OPD 范围: {np.min(relative_opd_mm):.6f} ~ {np.max(relative_opd_mm):.6f} mm")

# 计算不同公式的理想 OPD
# 公式 1: r²/(2f)（PROPER prop_lens 使用的公式）
ideal_opd_1_mm = r_sq / (2 * f)
ideal_opd_1_waves = ideal_opd_1_mm / wavelength_mm

# 公式 2: 2 * sag = 2 * r²/(4f) = r²/(2f)（与公式 1 相同）
sag = r_sq / (4 * f)
ideal_opd_2_mm = 2 * sag
ideal_opd_2_waves = ideal_opd_2_mm / wavelength_mm

# 公式 3: 精确计算（考虑反射角）
# 对于抛物面，入射光程 = sag，反射光程需要精确计算
# 反射后光线方向：r = d - 2(d·n)n
# 其中 d = (0, 0, 1)，n = (-x/(2f), -y/(2f), 1) / |n|

# 计算表面法向量
nx = -ray_x / (2 * f)
ny = -ray_y / (2 * f)
nz = np.ones_like(ray_x)
n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
nx /= n_mag
ny /= n_mag
nz /= n_mag

# 入射方向
dx, dy, dz = 0, 0, 1

# 反射方向
d_dot_n = dx * nx + dy * ny + dz * nz
rx = dx - 2 * d_dot_n * nx
ry = dy - 2 * d_dot_n * ny
rz = dz - 2 * d_dot_n * nz

# 入射光程（从 z=0 到表面）
incident_path = sag

# 反射光程（从表面到 z=0）
# 表面位置：(x, y, sag)
# 出射面位置：(x + t*rx, y + t*ry, 0)
# 解 sag + t*rz = 0 得 t = -sag/rz
t = -sag / rz
reflected_path = np.sqrt((t * rx)**2 + (t * ry)**2 + (t * rz)**2)

# 总光程
total_path = incident_path + reflected_path

# 参考光程（中心光线）
center_idx = n_rays // 2
ref_path = total_path[center_idx]

# 相对 OPD
ideal_opd_3_mm = total_path - ref_path
ideal_opd_3_waves = ideal_opd_3_mm / wavelength_mm

print(f"\n理想 OPD 公式比较:")
print(f"  公式 1 (r²/(2f)): 范围 {np.min(ideal_opd_1_mm):.6f} ~ {np.max(ideal_opd_1_mm):.6f} mm")
print(f"  公式 2 (2*sag):   范围 {np.min(ideal_opd_2_mm):.6f} ~ {np.max(ideal_opd_2_mm):.6f} mm")
print(f"  公式 3 (精确):    范围 {np.min(ideal_opd_3_mm):.6f} ~ {np.max(ideal_opd_3_mm):.6f} mm")
print(f"  ElementRaytracer: 范围 {np.min(relative_opd_mm):.6f} ~ {np.max(relative_opd_mm):.6f} mm")

# 计算像差
aberration_1 = relative_opd_mm - ideal_opd_1_mm
aberration_3 = relative_opd_mm - ideal_opd_3_mm

print(f"\n像差分析:")
print(f"  使用公式 1 的像差 RMS: {np.std(aberration_1[valid_mask])/wavelength_mm:.4f} waves")
print(f"  使用公式 3 的像差 RMS: {np.std(aberration_3[valid_mask])/wavelength_mm:.4f} waves")


# ============================================================
# 详细比较：逐点分析
# ============================================================

print_section("详细比较：逐点分析（沿 x 轴）")

# 选择 y=0 的点
y_zero_mask = np.abs(ray_y) < 0.1
x_values = ray_x[y_zero_mask]
sort_idx = np.argsort(x_values)
x_sorted = x_values[sort_idx]

opd_raytracer = relative_opd_mm[y_zero_mask][sort_idx]
opd_formula1 = ideal_opd_1_mm[y_zero_mask][sort_idx]
opd_formula3 = ideal_opd_3_mm[y_zero_mask][sort_idx]

print(f"{'x (mm)':>10} | {'Raytracer':>12} | {'r²/(2f)':>12} | {'精确':>12} | {'差异1':>10} | {'差异3':>10}")
print("-" * 80)
for i in range(len(x_sorted)):
    diff1 = opd_raytracer[i] - opd_formula1[i]
    diff3 = opd_raytracer[i] - opd_formula3[i]
    print(f"{x_sorted[i]:>10.2f} | {opd_raytracer[i]:>12.6f} | {opd_formula1[i]:>12.6f} | {opd_formula3[i]:>12.6f} | {diff1:>10.6f} | {diff3:>10.6f}")


# ============================================================
# 结论
# ============================================================

print_section("结论")

print("""
分析结果：

1. ElementRaytracer 的 OPD 与精确公式（公式 3）应该非常接近
   - 如果差异很小，说明 ElementRaytracer 计算正确
   - 如果差异较大，需要检查 ElementRaytracer 的实现

2. r²/(2f) 公式（公式 1）是近似公式
   - 对于小孔径（r << f），近似很好
   - 对于大孔径，会有明显偏差

3. 正确的做法：
   - 对于理想抛物面镜，应该使用精确公式计算理想 OPD
   - 或者，直接使用 ElementRaytracer 的 OPD 作为理想 OPD（像差 = 0）
   - 后者更简单，因为理想抛物面镜本身就是无像差的

4. 对于混合传播模式：
   - 如果元件是理想的（如理想抛物面镜），像差应该为零
   - 只有当元件有制造误差或设计偏差时，才会有非零像差
   - 因此，对于理想元件，不应该应用任何像差相位
""")

# 验证 ElementRaytracer 与精确公式的一致性
max_diff_3 = np.max(np.abs(aberration_3[valid_mask]))
print(f"\nElementRaytracer 与精确公式的最大差异: {max_diff_3/wavelength_mm:.6f} waves")

if max_diff_3 / wavelength_mm < 0.01:
    print("✓ ElementRaytracer 计算与精确公式一致（差异 < 0.01 waves）")
else:
    print("✗ ElementRaytracer 计算与精确公式存在差异，需要进一步分析")
