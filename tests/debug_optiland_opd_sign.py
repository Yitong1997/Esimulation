"""
测试 optiland 的 OPD 符号处理

核心问题：
1. 入射面垂直于入射光轴（z=0 平面）
2. 元件面相对于入射光轴倾斜（如 45°）
3. 出射面垂直于出射光轴
4. 光线从入射面出发，有的需要正向延长到达元件，有的需要反向延长
5. OPD 应该有正有负

测试场景：45° 平面镜
- 入射光沿 +Z 方向
- 平面镜绕 X 轴倾斜 45°
- 反射后光线沿 +Y 方向
- 入射面在 z=0，出射面也在 z=0（元件顶点位置）
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


def test_45deg_mirror_opd_sign():
    """测试 45° 平面镜的 OPD 符号
    
    几何分析：
    - 入射面在 z=0，垂直于 +Z
    - 平面镜绕 X 轴倾斜 45°，镜面方程：y + z = 0（通过原点）
    - 对于 y > 0 的光线，需要正向传播才能到达镜面
    - 对于 y < 0 的光线，需要反向传播才能到达镜面（镜面在 z < 0）
    - 出射面在 z=0，垂直于 +Y（反射后的光轴）
    
    OPD 分析：
    - 主光线（y=0）：入射点和出射点都在原点，OPD = 0
    - y > 0 的光线：需要正向传播到镜面，然后反射到出射面
    - y < 0 的光线：需要反向传播到镜面，然后反射到出射面
    
    对于平面镜，所有光线的总光程应该相同（平面镜不引入 OPD 差异）
    """
    print_section("测试 45° 平面镜的 OPD 符号")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    # 几何分析
    print("\n几何分析：")
    print("  入射面：z=0 平面，垂直于 +Z")
    print("  平面镜：绕 X 轴倾斜 45°")
    print("  镜面方程：y*sin(45°) + z*cos(45°) = 0，即 y + z = 0")
    print("  出射面：z=0 平面，垂直于 +Y（反射后光轴）")
    print()
    print("  对于入射面上 y > 0 的点：")
    print("    光线沿 +Z 传播，到达镜面的 z 坐标 = -y")
    print("    传播距离 = |z| = y（正向传播）")
    print()
    print("  对于入射面上 y < 0 的点：")
    print("    光线沿 +Z 传播，到达镜面的 z 坐标 = -y = |y|")
    print("    传播距离 = z = |y|（正向传播）")
    print()
    print("  实际上，所有光线都是正向传播到镜面！")
    print("  因为镜面方程 y + z = 0 意味着：")
    print("    y > 0 时，z = -y < 0（镜面在入射面后方）")
    print("    y < 0 时，z = -y > 0（镜面在入射面前方）")
    
    # 创建光学系统
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
    
    # 出射面：垂直于出射光轴 (0, 1, 0)
    # 需要绕 X 轴旋转 90°
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        rx=np.pi/2,
    )
    
    # 创建输入光线（在入射面 z=0 上）
    # 测试不同 y 位置的光线
    y_positions = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    n_rays = len(y_positions)
    
    input_rays = RealRays(
        x=np.zeros(n_rays),
        y=y_positions,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    input_rays.opd = np.zeros(n_rays)
    
    print_rays(input_rays, "输入光线（入射面 z=0）")
    
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
    
    print_rays(traced_rays, "追迹后的光线（出射面）")
    
    # 分析 OPD
    opd = np.asarray(traced_rays.opd)
    chief_opd = opd[2]  # y=0 的主光线
    relative_opd = opd - chief_opd
    
    print(f"\nOPD 分析：")
    print(f"  主光线 OPD: {chief_opd:.6f} mm")
    print(f"  相对 OPD: {relative_opd}")
    
    # 理论分析
    print(f"\n理论分析：")
    for i, y in enumerate(y_positions):
        # 光线从 (0, y, 0) 出发，沿 +Z 传播
        # 到达镜面的点：z = -y（镜面方程 y + z = 0）
        z_mirror = -y
        # 入射段光程
        path_in = abs(z_mirror)  # 从 z=0 到 z=z_mirror
        
        # 反射后光线沿 +Y 传播
        # 从镜面点 (0, y, z_mirror) 到出射面 z=0
        # 出射面垂直于 +Y，所以出射点的 y 坐标 = 0？
        # 不对，出射面是 z=0 平面但垂直于 +Y...
        # 这里有点混乱，让我重新分析
        
        print(f"  光线 {i}: y_in={y:.1f}, z_mirror={z_mirror:.1f}, "
              f"path_in={path_in:.1f}, OPD={opd[i]:.4f}")
    
    print(f"\n结论：")
    print(f"  平面镜不应引入 OPD 差异")
    print(f"  相对 OPD 应该全部为 0")
    print(f"  实际相对 OPD 范围: [{np.min(relative_opd):.4f}, {np.max(relative_opd):.4f}]")


def test_simple_propagation_opd():
    """测试简单传播的 OPD 计算"""
    print_section("测试简单传播的 OPD 计算")
    
    wavelength_um = 0.55
    
    # 创建简单系统：入射面 -> 10mm 传播 -> 出射面
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=np.inf, thickness=10.0, is_stop=True)
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
    
    print_rays(input_rays, "输入光线")
    
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
    
    print(f"\n预期：所有光线 OPD = 10mm（传播距离）")
    print(f"实际：OPD = {np.asarray(traced_rays.opd)}")


def test_negative_thickness():
    """测试负厚度（反向传播）的 OPD"""
    print_section("测试负厚度（反向传播）的 OPD")
    
    wavelength_um = 0.55
    
    # 创建系统：入射面 -> -10mm 传播 -> 出射面
    # 这意味着出射面在入射面的前方（z < 0）
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=np.inf, thickness=-10.0, is_stop=True)  # 负厚度
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
    
    print_rays(input_rays, "输入光线")
    
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
    
    print(f"\n预期：OPD = -10mm（反向传播，负光程）")
    print(f"实际：OPD = {np.asarray(traced_rays.opd)}")


def test_diverging_rays_to_tilted_surface():
    """测试发散光线到倾斜表面的 OPD"""
    print_section("测试发散光线到倾斜表面的 OPD")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10
    
    # 创建系统：发散光线 -> 45° 平面镜 -> 出射面
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
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
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        rx=np.pi/2,
    )
    
    # 创建发散光线（从点光源发出）
    # 光源在 z=-100mm 处
    source_z = -100.0
    y_positions = np.array([-5.0, 0.0, 5.0])
    n_rays = len(y_positions)
    
    # 计算方向余弦
    # 光线从 (0, 0, source_z) 发出，到达入射面 (0, y, 0)
    dx = np.zeros(n_rays)
    dy = y_positions - 0
    dz = 0 - source_z
    lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    L = dx / lengths
    M = dy / lengths
    N = dz / lengths
    
    input_rays = RealRays(
        x=np.zeros(n_rays),
        y=y_positions,
        z=np.zeros(n_rays),
        L=L,
        M=M,
        N=N,
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    input_rays.opd = np.zeros(n_rays)
    
    print(f"光源位置: (0, 0, {source_z})")
    print_rays(input_rays, "输入光线（发散）")
    
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
    
    # 分析
    opd = np.asarray(traced_rays.opd)
    chief_opd = opd[1]  # 主光线
    relative_opd = opd - chief_opd
    
    print(f"\n相对 OPD: {relative_opd}")
    print(f"对于发散光线，边缘光线的光程应该比主光线长")


def test_converging_rays_to_tilted_surface():
    """测试会聚光线到倾斜表面的 OPD"""
    print_section("测试会聚光线到倾斜表面的 OPD")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10
    
    # 创建系统：会聚光线 -> 45° 平面镜 -> 出射面
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
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
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=0.0,
        rx=np.pi/2,
    )
    
    # 创建会聚光线（会聚到 z=100mm 处的点）
    focus_z = 100.0
    y_positions = np.array([-5.0, 0.0, 5.0])
    n_rays = len(y_positions)
    
    # 计算方向余弦
    # 光线从入射面 (0, y, 0) 出发，会聚到 (0, 0, focus_z)
    dx = 0 - np.zeros(n_rays)
    dy = 0 - y_positions
    dz = focus_z - 0
    lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    L = dx / lengths
    M = dy / lengths
    N = dz / lengths
    
    input_rays = RealRays(
        x=np.zeros(n_rays),
        y=y_positions,
        z=np.zeros(n_rays),
        L=L,
        M=M,
        N=N,
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    input_rays.opd = np.zeros(n_rays)
    
    print(f"焦点位置: (0, 0, {focus_z})")
    print_rays(input_rays, "输入光线（会聚）")
    
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
    
    # 分析
    opd = np.asarray(traced_rays.opd)
    chief_opd = opd[1]  # 主光线
    relative_opd = opd - chief_opd
    
    print(f"\n相对 OPD: {relative_opd}")
    print(f"对于会聚光线，边缘光线的光程应该比主光线短（负 OPD）")


if __name__ == "__main__":
    test_simple_propagation_opd()
    test_negative_thickness()
    test_45deg_mirror_opd_sign()
    test_diverging_rays_to_tilted_surface()
    test_converging_rays_to_tilted_surface()
