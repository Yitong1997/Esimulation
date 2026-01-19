"""
详细调试 ElementRaytracer 的坐标转换和 OPD 计算

检查项目：
1. 入射面局部坐标系到全局坐标系的转换
2. optiland 光线追迹
3. 全局坐标系到出射面局部坐标系的转换
4. OPD 计算
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.rays import RealRays

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer, 
    SurfaceDefinition,
    compute_rotation_matrix,
    transform_rays_to_global,
    transform_rays_to_local,
)


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_rays(rays: RealRays, name: str, max_rays: int = 5):
    """打印光线信息"""
    x = np.asarray(rays.x)
    y = np.asarray(rays.y)
    z = np.asarray(rays.z)
    L = np.asarray(rays.L)
    M = np.asarray(rays.M)
    N = np.asarray(rays.N)
    opd = np.asarray(rays.opd)
    
    print(f"\n{name} (共 {len(x)} 条光线):")
    print(f"  {'idx':<5} {'x':<10} {'y':<10} {'z':<10} {'L':<10} {'M':<10} {'N':<10} {'OPD':<12}")
    print("  " + "-" * 75)
    
    for i in range(min(max_rays, len(x))):
        print(f"  {i:<5} {x[i]:<10.4f} {y[i]:<10.4f} {z[i]:<10.4f} "
              f"{L[i]:<10.4f} {M[i]:<10.4f} {N[i]:<10.4f} {opd[i]:<12.6f}")


def test_coordinate_transform():
    """测试坐标转换函数"""
    print_section("测试坐标转换函数")
    
    # 测试 1: 正入射（无旋转）
    print("\n--- 测试 1: 正入射 (0, 0, 1) ---")
    R = compute_rotation_matrix((0, 0, 1))
    print(f"旋转矩阵:\n{R}")
    print(f"是否为单位矩阵: {np.allclose(R, np.eye(3))}")
    
    # 测试 2: 45° 倾斜入射（在 YZ 平面内）
    print("\n--- 测试 2: 45° 倾斜入射 (0, sin(45°), cos(45°)) ---")
    angle = np.pi / 4
    direction = (0, np.sin(angle), np.cos(angle))
    R = compute_rotation_matrix(direction)
    print(f"入射方向: {direction}")
    print(f"旋转矩阵:\n{R}")
    
    # 验证：局部 Z 轴应该等于入射方向
    local_z = R[:, 2]
    print(f"局部 Z 轴: {local_z}")
    print(f"与入射方向一致: {np.allclose(local_z, direction)}")
    
    # 测试光线转换
    print("\n--- 测试光线坐标转换 ---")
    
    # 创建测试光线（在入射面局部坐标系中）
    # 光线沿局部 Z 轴方向
    test_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([0.55, 0.55, 0.55]),
    )
    test_rays.opd = np.array([0.0, 0.0, 0.0])
    
    print_rays(test_rays, "局部坐标系中的光线")
    
    # 转换到全局坐标系
    entrance_pos = (0, 0, 0)
    rays_global = transform_rays_to_global(test_rays, R, entrance_pos)
    print_rays(rays_global, "全局坐标系中的光线")
    
    # 验证方向转换
    print(f"\n方向验证:")
    print(f"  局部 (0, 0, 1) -> 全局 {(rays_global.L[0], rays_global.M[0], rays_global.N[0])}")
    print(f"  期望: {direction}")


def test_flat_mirror_45deg_detailed():
    """详细测试 45° 平面镜"""
    print_section("详细测试 45° 平面镜")
    
    wavelength_um = 0.55
    
    # 创建简单的输入光线（在入射面局部坐标系中）
    # 平行光，沿局部 Z 轴方向
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 5.0, -5.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 5),
    )
    input_rays.opd = np.zeros(5)
    
    print_rays(input_rays, "输入光线（入射面局部坐标系）")
    
    # 创建 45° 平面镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        tilt_x=np.pi/4,  # 45°
    )
    
    print(f"\n平面镜定义: {mirror}")
    
    # 创建光线追迹器
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),  # 正入射
        entrance_position=(0, 0, 0),
    )
    
    print(f"\n入射主光线方向: {raytracer.chief_ray_direction}")
    print(f"出射主光线方向: {raytracer.exit_chief_direction}")
    print(f"\n入射面旋转矩阵:\n{raytracer.rotation_matrix}")
    print(f"\n出射面旋转矩阵:\n{raytracer.exit_rotation_matrix}")
    
    # 手动执行坐标转换，检查中间步骤
    print("\n--- 手动检查坐标转换 ---")
    
    # 1. 转换到全局坐标系
    rays_global = transform_rays_to_global(
        input_rays,
        raytracer.rotation_matrix,
        raytracer.entrance_position,
    )
    print_rays(rays_global, "转换到全局坐标系后")
    
    # 2. 执行完整追迹
    output_rays = raytracer.trace(input_rays)
    print_rays(output_rays, "输出光线（出射面局部坐标系）")
    
    # 3. 检查 OPD
    opd_waves = raytracer.get_relative_opd_waves()
    print(f"\n相对 OPD (waves): {opd_waves}")
    
    # 4. 检查有效光线
    valid_mask = raytracer.get_valid_ray_mask()
    print(f"有效光线掩模: {valid_mask}")
    
    # 验证
    print("\n--- 验证 ---")
    print(f"平面镜应该:")
    print(f"  1. 不改变光线位置（在各自的局部坐标系中）")
    print(f"  2. 不引入 OPD（所有光线 OPD 相同）")
    print(f"  3. 改变光线方向（反射）")
    
    # 检查位置是否保持
    x_in = np.asarray(input_rays.x)
    y_in = np.asarray(input_rays.y)
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    
    print(f"\n位置变化:")
    print(f"  x: {x_in} -> {x_out}")
    print(f"  y: {y_in} -> {y_out}")
    
    # 检查 OPD
    print(f"\nOPD 范围: [{np.min(opd_waves):.4f}, {np.max(opd_waves):.4f}] waves")
    print(f"OPD 应该全部为 0（平面镜不引入 OPD）")


def test_optiland_direct():
    """直接测试 optiland 的光线追迹"""
    print_section("直接测试 optiland 光线追迹")
    
    from optiland.optic import Optic
    
    wavelength_um = 0.55
    
    # 创建简单的光学系统：45° 平面镜
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 平面镜
    # 注意：optiland 在精确 45° 时有数值问题
    tilt_x = np.pi/4 + 1e-10
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=tilt_x,
    )
    
    # 出射面（需要倾斜以垂直于出射光轴）
    # 45° 平面镜反射后，光线方向变为 (0, -1, 0)
    # 出射面应该垂直于 (0, -1, 0)，即绕 X 轴旋转 90°
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        material='air',
        rx=np.pi/2,  # 90° 倾斜
    )
    
    print("光学系统创建完成")
    print(f"表面数量: {len(optic.surface_group.surfaces)}")
    
    # 创建输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 3),
    )
    input_rays.opd = np.zeros(3)
    
    print_rays(input_rays, "输入光线")
    
    # 追迹
    surface_group = optic.surface_group
    traced_rays = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    traced_rays.opd = np.asarray(input_rays.opd).copy()
    
    surface_group.trace(traced_rays, skip=1)
    
    print_rays(traced_rays, "追迹后的光线")
    
    # 检查 OPD
    print(f"\nOPD 值: {np.asarray(traced_rays.opd)}")


def test_reflection_direction():
    """测试反射方向计算"""
    print_section("测试反射方向计算")
    
    # 入射方向：沿 +Z
    d = np.array([0.0, 0.0, 1.0])
    
    # 45° 平面镜的法向量
    # 初始法向量沿 -Z（指向入射侧）
    # 绕 X 轴旋转 45° 后
    tilt_x = np.pi / 4
    c, s = np.cos(tilt_x), np.sin(tilt_x)
    n_initial = np.array([0.0, 0.0, -1.0])
    
    # 旋转矩阵（绕 X 轴）
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    n = Rx @ n_initial
    print(f"入射方向: {d}")
    print(f"初始法向量: {n_initial}")
    print(f"旋转后法向量: {n}")
    
    # 反射公式：r = d - 2(d·n)n
    dot = np.dot(d, n)
    r = d - 2 * dot * n
    
    print(f"d·n = {dot}")
    print(f"反射方向: {r}")
    print(f"期望: (0, -1, 0)")


def test_exit_surface_orientation():
    """测试出射面方向"""
    print_section("测试出射面方向")
    
    wavelength_um = 0.55
    
    # 创建 45° 平面镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        tilt_x=np.pi/4,
    )
    
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength_um,
    )
    
    print(f"入射主光线方向: {raytracer.chief_ray_direction}")
    print(f"出射主光线方向: {raytracer.exit_chief_direction}")
    
    # 检查出射面旋转矩阵
    R_exit = raytracer.exit_rotation_matrix
    print(f"\n出射面旋转矩阵:\n{R_exit}")
    
    # 出射面的局部 Z 轴应该等于出射主光线方向
    local_z = R_exit[:, 2]
    print(f"\n出射面局部 Z 轴: {local_z}")
    print(f"出射主光线方向: {raytracer.exit_chief_direction}")
    print(f"一致性: {np.allclose(local_z, raytracer.exit_chief_direction)}")


if __name__ == "__main__":
    test_coordinate_transform()
    test_reflection_direction()
    test_exit_surface_orientation()
    test_optiland_direct()
    test_flat_mirror_45deg_detailed()
