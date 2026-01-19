"""
测试 ElementRaytracer 的修复方案

问题分析：
1. optiland 的 OPD 包含光线传播的总光程
2. 对于倾斜表面，不同位置的光线到达表面的距离不同
3. 需要正确设置出射面位置，使其垂直于出射光轴

修复方案：
1. 入射面和出射面都应该在元件顶点位置（z=0）
2. 出射面应该正确倾斜，垂直于出射光轴
3. OPD 应该是相对于主光线的差值
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


def print_rays(rays: RealRays, name: str, max_rays: int = 10):
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


def test_correct_45deg_flat_mirror():
    """测试正确配置的 45° 平面镜"""
    print_section("正确配置的 45° 平面镜")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    # 关键：出射面需要正确倾斜
    # 45° 平面镜反射后，光线方向变为 (0, 1, 0)
    # 出射面应该垂直于 (0, 1, 0)，即绕 X 轴旋转 90°
    # 但是，出射面的位置应该在 z=0（元件顶点）
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 倾斜平面镜
    # thickness=0 表示出射面在同一位置
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=tilt_x,
    )
    
    # 出射面：垂直于出射光轴 (0, 1, 0)
    # 需要绕 X 轴旋转 90°
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        rx=np.pi/2,  # 90° 倾斜
    )
    
    # 创建输入光线
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
    
    print_rays(input_rays, "输入光线")
    
    # 追迹
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
    
    optic.surface_group.trace(traced_rays, skip=1)
    
    print_rays(traced_rays, "追迹后的光线")
    
    # 分析 OPD
    opd = np.asarray(traced_rays.opd)
    print(f"\nOPD 分析:")
    print(f"  OPD 值: {opd}")
    print(f"  OPD 范围: [{np.min(opd):.6f}, {np.max(opd):.6f}]")
    print(f"  OPD 差异: {np.max(opd) - np.min(opd):.6f}")
    
    # 计算相对 OPD
    chief_opd = opd[0]  # 主光线
    relative_opd = opd - chief_opd
    print(f"\n相对 OPD (相对于主光线):")
    print(f"  {relative_opd}")
    
    print("\n预期结果:")
    print("  - 平面镜不引入 OPD 差异")
    print("  - 所有光线的相对 OPD 应该为 0")


def test_correct_45deg_parabolic_mirror():
    """测试正确配置的 45° 抛物面镜"""
    print_section("正确配置的 45° 抛物面镜 (f=-50mm)")
    
    wavelength_um = 0.55
    f = -50.0  # mm，凸面镜
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 倾斜抛物面镜
    optic.add_surface(
        index=1,
        radius=2*f,  # -100mm
        thickness=0.0,
        material='mirror',
        is_stop=True,
        conic=-1.0,  # 抛物面
        rx=tilt_x,
    )
    
    # 出射面：垂直于出射光轴 (0, 1, 0)
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        rx=np.pi/2,
    )
    
    # 创建输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0, 0.0, 0.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0, 5.0, -5.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 6),
    )
    input_rays.opd = np.zeros(6)
    
    print_rays(input_rays, "输入光线")
    
    # 追迹
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
    
    optic.surface_group.trace(traced_rays, skip=1)
    
    print_rays(traced_rays, "追迹后的光线")
    
    # 分析 OPD
    opd = np.asarray(traced_rays.opd)
    print(f"\nOPD 分析:")
    print(f"  OPD 值: {opd}")
    
    # 计算相对 OPD
    chief_opd = opd[0]  # 主光线
    relative_opd = opd - chief_opd
    print(f"\n相对 OPD (相对于主光线):")
    print(f"  {relative_opd}")
    
    # 转换为波长数
    wavelength_mm = wavelength_um * 1e-3
    relative_opd_waves = relative_opd / wavelength_mm
    print(f"\n相对 OPD (波长数):")
    print(f"  {relative_opd_waves}")
    
    # 理论 OPD（抛物面镜）
    # 对于抛物面 z = r²/(4f)，反射后 OPD = 2z = r²/(2f)
    print(f"\n理论 OPD 计算:")
    for i, (x, y) in enumerate(zip(input_rays.x, input_rays.y)):
        r_sq = x**2 + y**2
        z_sag = r_sq / (4 * abs(f))
        opd_theory_mm = 2 * z_sag  # 反射加倍
        opd_theory_waves = opd_theory_mm / wavelength_mm
        print(f"  光线 {i}: r²={r_sq:.1f}, 理论 OPD={opd_theory_waves:.4f} waves, "
              f"实测={relative_opd_waves[i]:.4f} waves")


def test_no_tilt_parabolic_mirror():
    """测试无倾斜的抛物面镜（作为参考）"""
    print_section("无倾斜的抛物面镜 (f=-50mm) - 参考")
    
    wavelength_um = 0.55
    f = -50.0  # mm，凸面镜
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 抛物面镜（无倾斜）
    optic.add_surface(
        index=1,
        radius=2*f,  # -100mm
        thickness=0.0,
        material='mirror',
        is_stop=True,
        conic=-1.0,  # 抛物面
    )
    
    # 出射面
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 创建输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0, 0.0, 0.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0, 5.0, -5.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 6),
    )
    input_rays.opd = np.zeros(6)
    
    print_rays(input_rays, "输入光线")
    
    # 追迹
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
    
    optic.surface_group.trace(traced_rays, skip=1)
    
    print_rays(traced_rays, "追迹后的光线")
    
    # 分析 OPD
    opd = np.asarray(traced_rays.opd)
    
    # 计算相对 OPD
    chief_opd = opd[0]  # 主光线
    relative_opd = opd - chief_opd
    
    # 转换为波长数
    wavelength_mm = wavelength_um * 1e-3
    relative_opd_waves = relative_opd / wavelength_mm
    print(f"\n相对 OPD (波长数):")
    print(f"  {relative_opd_waves}")
    
    # 理论 OPD
    print(f"\n理论 OPD 计算:")
    for i, (x, y) in enumerate(zip(input_rays.x, input_rays.y)):
        r_sq = x**2 + y**2
        z_sag = r_sq / (4 * abs(f))
        opd_theory_mm = 2 * z_sag  # 反射加倍
        opd_theory_waves = opd_theory_mm / wavelength_mm
        print(f"  光线 {i}: r²={r_sq:.1f}, 理论 OPD={opd_theory_waves:.4f} waves, "
              f"实测={relative_opd_waves[i]:.4f} waves")


def test_element_raytracer_fix():
    """测试 ElementRaytracer 的修复"""
    print_section("测试 ElementRaytracer 修复")
    
    from wavefront_to_rays.element_raytracer import (
        ElementRaytracer, 
        SurfaceDefinition,
    )
    
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
    
    # 创建光线追迹器
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength_um,
    )
    
    # 创建输入光线
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
    
    print_rays(input_rays, "输入光线")
    
    # 执行追迹
    output_rays = raytracer.trace(input_rays)
    
    print_rays(output_rays, "输出光线")
    
    # 获取相对 OPD
    opd_waves = raytracer.get_relative_opd_waves()
    print(f"\n相对 OPD (波长数): {opd_waves}")
    
    print("\n预期结果:")
    print("  - 平面镜不引入 OPD 差异")
    print("  - 所有光线的相对 OPD 应该为 0")
    
    # 检查出射面配置
    print(f"\n出射主光线方向: {raytracer.exit_chief_direction}")
    print(f"出射面旋转矩阵:\n{raytracer.exit_rotation_matrix}")


if __name__ == "__main__":
    test_correct_45deg_flat_mirror()
    test_no_tilt_parabolic_mirror()
    test_correct_45deg_parabolic_mirror()
    test_element_raytracer_fix()
