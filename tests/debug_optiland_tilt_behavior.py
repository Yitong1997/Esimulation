"""
测试 optiland 对倾斜表面的处理行为

目标：理解 optiland 如何处理倾斜表面的：
1. 光线追迹
2. OPD 计算
3. 坐标系统
"""
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


def test_normal_incidence_flat_mirror():
    """测试正入射平面镜（无倾斜）"""
    print_section("测试 1: 正入射平面镜（无倾斜）")
    
    wavelength_um = 0.55
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 平面镜（无倾斜）
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
    )
    
    # 像面
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
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
    
    print("\n预期结果:")
    print("  - 位置不变")
    print("  - 方向反转: N = -1")
    print("  - OPD = 0")


def test_tilted_flat_mirror_45deg():
    """测试 45° 倾斜平面镜"""
    print_section("测试 2: 45° 倾斜平面镜")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 倾斜平面镜
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=tilt_x,
    )
    
    # 像面（不倾斜）
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
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
    
    print("\n预期结果:")
    print("  - 方向变为 (0, 1, 0) 或 (0, -1, 0)")
    print("  - OPD 应该全部相同（平面镜不引入 OPD 差异）")
    
    # 分析 OPD
    opd = np.asarray(traced_rays.opd)
    print(f"\nOPD 分析:")
    print(f"  OPD 值: {opd}")
    print(f"  OPD 范围: [{np.min(opd):.6f}, {np.max(opd):.6f}]")
    print(f"  OPD 差异: {np.max(opd) - np.min(opd):.6f}")


def test_tilted_flat_mirror_with_tilted_exit():
    """测试 45° 倾斜平面镜 + 倾斜出射面"""
    print_section("测试 3: 45° 倾斜平面镜 + 倾斜出射面")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 倾斜平面镜
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=tilt_x,
    )
    
    # 倾斜出射面（垂直于出射光轴）
    # 出射光沿 +Y 方向，所以出射面应该绕 X 轴旋转 90°
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


def test_opd_source():
    """测试 OPD 的来源"""
    print_section("测试 4: OPD 来源分析")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10
    
    # 测试不同 y 位置的光线
    y_positions = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    print(f"\n{'y 位置':<10} {'OPD (mm)':<15} {'OPD (waves)':<15}")
    print("-" * 45)
    
    for y_pos in y_positions:
        optic = Optic()
        optic.set_aperture(aperture_type='EPD', value=30.0)
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        optic.add_wavelength(value=wavelength_um, is_primary=True)
        
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        optic.add_surface(
            index=1,
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            is_stop=True,
            rx=tilt_x,
        )
        optic.add_surface(index=2, radius=np.inf, thickness=0.0)
        
        # 单条光线
        input_rays = RealRays(
            x=np.array([0.0]),
            y=np.array([y_pos]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([wavelength_um]),
        )
        input_rays.opd = np.array([0.0])
        
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
        
        opd_mm = traced_rays.opd[0]
        opd_waves = opd_mm / (wavelength_um * 1e-3)
        
        print(f"{y_pos:<10.1f} {opd_mm:<15.6f} {opd_waves:<15.2f}")
    
    print("\n分析:")
    print("  如果 OPD 与 y 位置成正比，说明 optiland 计算了")
    print("  光线到达倾斜表面的额外光程")
    print("  这是因为倾斜表面的 z 坐标随 y 变化: z = y * tan(tilt)")


def test_understanding_optiland_coordinate():
    """理解 optiland 的坐标系统"""
    print_section("测试 5: 理解 optiland 坐标系统")
    
    wavelength_um = 0.55
    
    # 创建一个简单的系统：只有物面和像面
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=np.inf, thickness=10.0, is_stop=True)  # 10mm 厚度
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 创建输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([0.0, 0.0]),
        L=np.array([0.0, 0.0]),
        M=np.array([0.0, 0.0]),
        N=np.array([1.0, 1.0]),
        intensity=np.array([1.0, 1.0]),
        wavelength=np.array([wavelength_um, wavelength_um]),
    )
    input_rays.opd = np.zeros(2)
    
    print_rays(input_rays, "输入光线 (z=0)")
    
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
    
    print("\n分析:")
    print("  光线从 z=0 传播 10mm 到像面")
    print("  OPD 应该等于传播距离 = 10mm")


if __name__ == "__main__":
    test_normal_incidence_flat_mirror()
    test_tilted_flat_mirror_45deg()
    test_tilted_flat_mirror_with_tilted_exit()
    test_opd_source()
    test_understanding_optiland_coordinate()
