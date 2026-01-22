"""
调试 Surface 4 的配置

检查 ZMX 文件中 Surface 4 的配置是否正确传递给 ElementRaytracer。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 导入模块
# ============================================================================

print_section("导入模块")

from hybrid_optical_propagation import load_optical_system_from_zmx
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays

print("[OK] 模块导入成功")


# ============================================================================
# 加载光学系统
# ============================================================================

print_section("加载光学系统")

zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

print(f"表面数量: {len(optical_system)}")

# 找到 Surface 4 (M1)
surface_4 = None
for i, surface in enumerate(optical_system):
    print(f"  [{i}] 表面 {surface.index}: {surface.surface_type}, "
          f"R={surface.radius:.2f}mm, is_mirror={surface.is_mirror}, "
          f"位置={surface.vertex_position}")
    if surface.index == 4:
        surface_4 = surface
        surface_4_idx = i


# ============================================================================
# 检查 Surface 4 的详细配置
# ============================================================================

print_section("检查 Surface 4 的详细配置")

print(f"Surface 4 详细信息:")
print(f"  index: {surface_4.index}")
print(f"  surface_type: {surface_4.surface_type}")
print(f"  radius: {surface_4.radius}")
print(f"  thickness: {surface_4.thickness}")
print(f"  is_mirror: {surface_4.is_mirror}")
print(f"  material: {surface_4.material}")
print(f"  semi_aperture: {surface_4.semi_aperture}")
print(f"  conic: {surface_4.conic}")
print(f"  vertex_position: {surface_4.vertex_position}")
print(f"  surface_normal: {surface_4.surface_normal}")

# 检查是否有倾斜
if hasattr(surface_4, 'tilt_x'):
    print(f"  tilt_x: {surface_4.tilt_x}")
if hasattr(surface_4, 'tilt_y'):
    print(f"  tilt_y: {surface_4.tilt_y}")


# ============================================================================
# 创建 SurfaceDefinition
# ============================================================================

print_section("创建 SurfaceDefinition")

# 检查 GlobalSurfaceDefinition 的所有属性
print(f"GlobalSurfaceDefinition 属性: {dir(surface_4)}")

# 创建 SurfaceDefinition
surface_def = SurfaceDefinition(
    surface_type='mirror' if surface_4.is_mirror else 'refract',
    radius=surface_4.radius,
    thickness=surface_4.thickness,
    material='mirror' if surface_4.is_mirror else surface_4.material,
    semi_aperture=surface_4.semi_aperture,
    conic=surface_4.conic,
)

print(f"SurfaceDefinition: {surface_def}")


# ============================================================================
# 创建 ElementRaytracer
# ============================================================================

print_section("创建 ElementRaytracer")

# 入射光轴方向（沿 +Z）
chief_ray_direction = (0.0, 0.0, 1.0)

# 入射位置（Surface 4 的顶点位置）
entrance_position = tuple(surface_4.vertex_position)

print(f"主光线方向: {chief_ray_direction}")
print(f"入射位置: {entrance_position}")

raytracer = ElementRaytracer(
    surfaces=[surface_def],
    wavelength=0.55,
    chief_ray_direction=chief_ray_direction,
    entrance_position=entrance_position,
)

print(f"ElementRaytracer 创建完成")
print(f"  exit_chief_direction: {raytracer.exit_chief_direction}")
print(f"  exit_rotation_matrix:\n{raytracer.exit_rotation_matrix}")


# ============================================================================
# 检查 optiland 光学系统
# ============================================================================

print_section("检查 optiland 光学系统")

optic = raytracer.optic
print(f"表面数量: {len(optic.surface_group.surfaces)}")

for i, surface in enumerate(optic.surface_group.surfaces):
    print(f"  Surface {i}:")
    print(f"    material_pre: {surface.material_pre}")
    print(f"    material_post: {surface.material_post}")
    print(f"    interaction_model: {type(surface.interaction_model).__name__}")
    print(f"    geometry: {type(surface.geometry).__name__}")
    if hasattr(surface.geometry, 'cs'):
        cs = surface.geometry.cs
        print(f"    cs.rx: {cs.rx}")
        print(f"    cs.ry: {cs.ry}")


# ============================================================================
# 测试光线追迹
# ============================================================================

print_section("测试光线追迹")

# 创建测试光线（在入射面局部坐标系中）
input_rays = RealRays(
    x=np.array([0.0, 5.0, -5.0]),
    y=np.array([0.0, 0.0, 0.0]),
    z=np.array([0.0, 0.0, 0.0]),
    L=np.array([0.0, 0.0, 0.0]),
    M=np.array([0.0, 0.0, 0.0]),
    N=np.array([1.0, 1.0, 1.0]),
    intensity=np.array([1.0, 1.0, 1.0]),
    wavelength=np.array([0.55, 0.55, 0.55]),
)

print(f"输入光线:")
print(f"  x: {input_rays.x}")
print(f"  y: {input_rays.y}")
print(f"  z: {input_rays.z}")
print(f"  L: {input_rays.L}")
print(f"  M: {input_rays.M}")
print(f"  N: {input_rays.N}")

# 执行光线追迹
output_rays = raytracer.trace(input_rays)

print(f"\n输出光线:")
print(f"  x: {output_rays.x}")
print(f"  y: {output_rays.y}")
print(f"  z: {output_rays.z}")
print(f"  L: {output_rays.L}")
print(f"  M: {output_rays.M}")
print(f"  N: {output_rays.N}")


# ============================================================================
# 检查 Surface 4 的法向量和期望的反射方向
# ============================================================================

print_section("检查 Surface 4 的法向量和期望的反射方向")

# Surface 4 的法向量
normal = surface_4.surface_normal
print(f"Surface 4 法向量: {normal}")

# 入射方向
d = np.array([0.0, 0.0, 1.0])
print(f"入射方向: {d}")

# 反射公式: r = d - 2(d·n)n
# 注意：法向量应该指向入射侧
# 如果法向量指向 +Z，则需要取反
n = normal
if np.dot(d, n) > 0:
    n = -n
    print(f"法向量取反: {n}")

r = d - 2 * np.dot(d, n) * n
r = r / np.linalg.norm(r)
print(f"期望的反射方向: {r}")


print_section("调试完成")
