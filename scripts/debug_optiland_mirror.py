"""
调试 optiland 反射镜光线追迹

检查 optiland 是否正确处理反射镜。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 创建简单的反射镜系统
# ============================================================================

print_section("创建简单的反射镜系统")

optic = Optic()

# 设置系统参数
optic.set_aperture(aperture_type='EPD', value=20.0)
optic.set_field_type(field_type='angle')
optic.add_field(y=0, x=0)
optic.add_wavelength(value=0.55, is_primary=True)

# 添加物面
optic.add_surface(index=0, radius=np.inf, thickness=np.inf)

# 添加平面反射镜
optic.add_surface(
    index=1,
    radius=np.inf,
    thickness=0.0,
    material='mirror',
    is_stop=True,
)

# 添加出射面
optic.add_surface(
    index=2,
    radius=np.inf,
    thickness=0.0,
    material='air',
)

print("光学系统创建完成")
print(f"表面数量: {len(optic.surface_group.surfaces)}")

for i, surface in enumerate(optic.surface_group.surfaces):
    print(f"  Surface {i}: material={surface.material_post}, "
          f"interaction={type(surface.interaction_model).__name__}")


# ============================================================================
# 创建测试光线
# ============================================================================

print_section("创建测试光线")

# 创建沿 +Z 方向的光线
n_rays = 5
x = np.linspace(-5, 5, n_rays)
y = np.zeros(n_rays)
z = np.zeros(n_rays)
L = np.zeros(n_rays)
M = np.zeros(n_rays)
N = np.ones(n_rays)

rays = RealRays(
    x=x, y=y, z=z,
    L=L, M=M, N=N,
    intensity=np.ones(n_rays),
    wavelength=np.full(n_rays, 0.55),
)

print(f"光线数量: {n_rays}")
print(f"初始位置 z: {rays.z}")
print(f"初始方向 N: {rays.N}")


# ============================================================================
# 执行光线追迹
# ============================================================================

print_section("执行光线追迹")

# 使用 optiland 的追迹方法
optic.surface_group.trace(rays, skip=1)

print(f"追迹后位置 z: {rays.z}")
print(f"追迹后方向 L: {rays.L}")
print(f"追迹后方向 M: {rays.M}")
print(f"追迹后方向 N: {rays.N}")


# ============================================================================
# 检查反射镜的交互模型
# ============================================================================

print_section("检查反射镜的交互模型")

mirror_surface = optic.surface_group.surfaces[1]
print(f"反射镜表面:")
print(f"  material_pre: {mirror_surface.material_pre}")
print(f"  material_post: {mirror_surface.material_post}")
print(f"  interaction_model: {type(mirror_surface.interaction_model).__name__}")

# 检查 interaction_model 的属性
interaction = mirror_surface.interaction_model
print(f"  interaction_model 属性: {dir(interaction)}")


# ============================================================================
# 手动测试反射
# ============================================================================

print_section("手动测试反射")

# 创建新的光线
rays2 = RealRays(
    x=np.array([0.0]),
    y=np.array([0.0]),
    z=np.array([0.0]),
    L=np.array([0.0]),
    M=np.array([0.0]),
    N=np.array([1.0]),
    intensity=np.array([1.0]),
    wavelength=np.array([0.55]),
)

print(f"初始方向: L={rays2.L[0]}, M={rays2.M[0]}, N={rays2.N[0]}")

# 手动调用交互模型
result = mirror_surface.interaction_model.interact_real_rays(rays2)

print(f"反射后方向: L={result.L[0]}, M={result.M[0]}, N={result.N[0]}")


# ============================================================================
# 检查 45° 倾斜反射镜
# ============================================================================

print_section("检查 45° 倾斜反射镜")

optic2 = Optic()

optic2.set_aperture(aperture_type='EPD', value=20.0)
optic2.set_field_type(field_type='angle')
optic2.add_field(y=0, x=0)
optic2.add_wavelength(value=0.55, is_primary=True)

# 添加物面
optic2.add_surface(index=0, radius=np.inf, thickness=np.inf)

# 添加 45° 倾斜的平面反射镜
# 绕 X 轴旋转 45°
tilt_x = np.pi / 4 + 1e-10  # 避免精确 45°
optic2.add_surface(
    index=1,
    radius=np.inf,
    thickness=0.0,
    material='mirror',
    is_stop=True,
    rx=tilt_x,
)

# 添加出射面
optic2.add_surface(
    index=2,
    radius=np.inf,
    thickness=0.0,
    material='air',
)

print(f"45° 倾斜反射镜系统创建完成")

# 创建测试光线
rays3 = RealRays(
    x=np.array([0.0]),
    y=np.array([0.0]),
    z=np.array([0.0]),
    L=np.array([0.0]),
    M=np.array([0.0]),
    N=np.array([1.0]),
    intensity=np.array([1.0]),
    wavelength=np.array([0.55]),
)

print(f"初始方向: L={rays3.L[0]:.6f}, M={rays3.M[0]:.6f}, N={rays3.N[0]:.6f}")

# 执行追迹
optic2.surface_group.trace(rays3, skip=1)

print(f"追迹后方向: L={rays3.L[0]:.6f}, M={rays3.M[0]:.6f}, N={rays3.N[0]:.6f}")

# 期望的反射方向（绕 X 轴旋转 45° 的反射镜，入射沿 +Z）
# 反射后应该沿 +Y 方向
print(f"期望方向: L=0, M=1, N=0 (沿 +Y 方向)")


print_section("调试完成")
