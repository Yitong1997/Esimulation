"""
测试带符号的 OPD 计算方法

核心问题：
optiland 在计算 OPD 时使用了 abs(t)，导致无论光线是正向还是反向传播，
OPD 都是正的。这对于折叠光路是不正确的。

正确的 OPD 计算：
- 正向传播（t > 0）：OPD 增加
- 反向传播（t < 0）：OPD 减少（负的增量）

本测试验证新的带符号 OPD 计算方法。
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_signed_opd(
    rays_before: RealRays,
    rays_after: RealRays,
    t: np.ndarray,
    n: float,
) -> np.ndarray:
    """计算带符号的 OPD 增量
    
    参数:
        rays_before: 传播前的光线
        rays_after: 传播后的光线
        t: 传播距离（可正可负）
        n: 介质折射率
    
    返回:
        OPD 增量（带符号）
    """
    # OPD = n * t（保留符号）
    return n * t


def trace_with_signed_opd(optic: Optic, input_rays: RealRays, skip: int = 1) -> tuple:
    """执行光线追迹并计算带符号的 OPD
    
    仿照 optiland 的追迹过程，但使用带符号的 OPD 计算。
    
    参数:
        optic: 光学系统
        input_rays: 输入光线
        skip: 跳过的表面数量
    
    返回:
        (traced_rays, signed_opd_increments): 追迹后的光线和每个表面的 OPD 增量
    """
    # 复制光线
    rays = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    rays.opd = np.zeros(len(rays.x))  # 初始化 OPD 为 0
    
    # 记录每个表面的 OPD 增量
    opd_increments = []
    
    # 获取表面组
    surface_group = optic.surface_group
    surfaces = surface_group.surfaces
    
    # 追迹每个表面
    for i, surface in enumerate(surfaces):
        if i < skip:
            continue
        
        # 保存追迹前的状态
        z_before = np.asarray(rays.z).copy()
        
        # 坐标变换到表面局部坐标系
        surface.geometry.localize(rays)
        
        # 计算到表面的距离
        t = surface.geometry.distance(rays)
        t = np.asarray(t)
        
        # 获取介质折射率
        n = surface.material_pre.n(rays.w)
        n = np.asarray(n)
        if n.ndim == 0:
            n = float(n)
        
        # 计算带符号的 OPD 增量
        # 关键：不使用 abs(t)，保留符号
        opd_increment = n * t
        opd_increments.append({
            'surface_index': i,
            't': t.copy(),
            'n': n,
            'opd_increment': opd_increment.copy(),
        })
        
        # 传播光线
        surface.material_pre.propagation_model.propagate(rays, t)
        
        # 更新 OPD（使用带符号的值）
        rays.opd = rays.opd + opd_increment
        
        # 与表面交互（反射/折射）
        rays = surface.interaction_model.interact_real_rays(rays)
        
        # 坐标变换回全局坐标系
        surface.geometry.globalize(rays)
    
    return rays, opd_increments


def test_flat_mirror_45deg():
    """测试 45° 平面镜的 OPD 计算"""
    print_section("测试 45° 平面镜的 OPD 计算")
    
    wavelength_um = 0.55
    tilt_x = np.pi/4 + 1e-10  # 避免精确 45°
    
    # 创建系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 平面镜
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=tilt_x,
    )
    
    # 出射面（垂直于出射光轴，即 -Y 方向）
    optic.add_surface(index=2, radius=np.inf, thickness=0.0, rx=np.pi/2)
    
    # 创建测试光线
    # 光线沿 +Z 方向入射，在 y 方向有不同位置
    y_positions = np.array([-5.0, 0.0, 5.0])
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
    
    # 使用带符号的 OPD 追迹
    traced_rays, opd_increments = trace_with_signed_opd(optic, input_rays, skip=1)
    
    print(f"\n45° 平面镜追迹结果（带符号 OPD）：")
    print(f"\n{'y_in':<10} {'t':<12} {'OPD增量':<12} {'总OPD':<12}")
    print("-" * 50)
    
    for i, y in enumerate(y_positions):
        t = opd_increments[0]['t'][i]
        opd_inc = opd_increments[0]['opd_increment'][i]
        total_opd = traced_rays.opd[i]
        print(f"{y:<10.2f} {t:<12.4f} {opd_inc:<12.4f} {total_opd:<12.4f}")
    
    # 分析
    print(f"\n分析：")
    print(f"  - y=-5 的光线需要正向传播更远才能到达倾斜的镜面")
    print(f"  - y=+5 的光线需要反向传播（t<0）才能到达倾斜的镜面")
    print(f"  - 平面镜本身不引入 OPD，但到达镜面的几何光程不同")
    
    # 计算相对于主光线的 OPD
    chief_opd = traced_rays.opd[1]  # y=0 的光线
    relative_opd = traced_rays.opd - chief_opd
    
    print(f"\n相对于主光线的 OPD：")
    for i, y in enumerate(y_positions):
        print(f"  y={y:+.1f}: 相对 OPD = {relative_opd[i]:.4f} mm")
    
    # 对于平面镜，相对 OPD 应该反映到达镜面的几何光程差
    # 这个几何光程差在折叠光路中不应该计入元件 OPD
    print(f"\n结论：")
    print(f"  带符号的 OPD 计算正确反映了光线到达镜面的几何光程差")
    print(f"  但对于折叠光路，这个几何光程差不应计入元件引入的 OPD")


def test_parabolic_mirror_45deg():
    """测试 45° 抛物面镜的 OPD 计算"""
    print_section("测试 45° 抛物面镜的 OPD 计算")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    f = -50.0  # mm，凸面镜
    R = 2 * f  # -100mm
    tilt_x = np.pi/4 + 1e-10
    
    # 创建系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(
        index=1,
        radius=R,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        conic=-1.0,
        rx=tilt_x,
    )
    optic.add_surface(index=2, radius=np.inf, thickness=0.0, rx=np.pi/2)
    
    # 测试光线（只在 x 方向，避免 y 方向的复杂性）
    x_positions = np.array([0.0, 5.0, 10.0])
    n_rays = len(x_positions)
    
    input_rays = RealRays(
        x=x_positions,
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 使用带符号的 OPD 追迹
    traced_rays, opd_increments = trace_with_signed_opd(optic, input_rays, skip=1)
    
    print(f"\n45° 抛物面镜追迹结果（带符号 OPD）：")
    print(f"\n{'x_in':<10} {'t':<12} {'OPD增量':<12} {'总OPD':<12}")
    print("-" * 50)
    
    for i, x in enumerate(x_positions):
        t = opd_increments[0]['t'][i]
        opd_inc = opd_increments[0]['opd_increment'][i]
        total_opd = traced_rays.opd[i]
        print(f"{x:<10.2f} {t:<12.4f} {opd_inc:<12.4f} {total_opd:<12.4f}")
    
    # 计算相对于主光线的 OPD
    chief_opd = traced_rays.opd[0]  # x=0 的光线
    relative_opd = traced_rays.opd - chief_opd
    relative_opd_waves = relative_opd / wavelength_mm
    
    print(f"\n相对于主光线的 OPD（波长数）：")
    for i, x in enumerate(x_positions):
        print(f"  x={x:.1f}: 相对 OPD = {relative_opd_waves[i]:.4f} waves")
    
    # 计算理论元件 OPD（基于表面矢高）
    print(f"\n理论元件 OPD（基于表面矢高）：")
    for x in x_positions:
        r = abs(x)
        sag = r**2 / (2 * R)  # 抛物面矢高
        element_opd_mm = 2 * sag  # 反射加倍
        element_opd_waves = element_opd_mm / wavelength_mm
        print(f"  x={x:.1f}: 元件 OPD = {element_opd_waves:.4f} waves")


def test_opd_sign_interpretation():
    """测试 OPD 符号的物理意义"""
    print_section("OPD 符号的物理意义")
    
    print("""
OPD 符号约定：

1. 传播距离 t 的符号：
   - t > 0: 光线沿传播方向前进（正向传播）
   - t < 0: 光线沿传播方向后退（反向传播）

2. OPD 增量的符号：
   - OPD_increment = n * t
   - t > 0: OPD 增加（光程增加）
   - t < 0: OPD 减少（光程减少）

3. 对于折叠光路：
   - 入射面和出射面都垂直于各自的光轴
   - 光线从入射面到元件表面的距离可正可负
   - 这个几何光程差不应计入元件引入的 OPD

4. 元件引入的 OPD：
   - 只与表面形状有关
   - 平面镜：元件 OPD = 0
   - 曲面镜：元件 OPD = 2 * sag（反射加倍）
   - 与光线到达表面的几何光程无关

5. 正确的混合光学仿真方法：
   a. 使用 optiland 追迹光线，获取光线与表面的交点
   b. 根据交点坐标计算表面矢高
   c. 元件 OPD = 2 * sag（反射镜）
   d. 不使用 optiland 的 OPD 输出
""")


def test_element_opd_from_sag():
    """测试基于表面矢高的元件 OPD 计算"""
    print_section("基于表面矢高的元件 OPD 计算")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    
    # 测试参数
    R = -100.0  # mm，凸面镜曲率半径
    k = -1.0    # 抛物面
    
    def calculate_sag(x, y, R, k):
        """计算圆锥曲面的矢高"""
        r2 = x**2 + y**2
        if k == -1:
            # 抛物面简化公式
            return r2 / (2 * R)
        else:
            # 一般圆锥曲线
            return r2 / (R * (1 + np.sqrt(1 - (1 + k) * r2 / R**2)))
    
    def calculate_element_opd(x, y, R, k, is_mirror=True):
        """计算元件引入的 OPD
        
        参数:
            x, y: 光线在表面上的交点坐标
            R: 曲率半径
            k: 圆锥常数
            is_mirror: 是否为反射镜
        
        返回:
            元件 OPD（mm）
        """
        sag = calculate_sag(x, y, R, k)
        if is_mirror:
            return 2 * sag  # 反射加倍
        else:
            # 折射面需要考虑折射率差
            # OPD = (n2 - n1) * sag
            raise NotImplementedError("折射面 OPD 计算待实现")
    
    # 测试
    print(f"抛物面镜参数：R = {R} mm, k = {k}")
    print(f"\n{'x (mm)':<10} {'y (mm)':<10} {'sag (mm)':<12} {'元件OPD (mm)':<15} {'元件OPD (waves)':<15}")
    print("-" * 65)
    
    test_points = [
        (0, 0),
        (5, 0),
        (0, 5),
        (10, 0),
        (7.07, 7.07),
    ]
    
    for x, y in test_points:
        sag = calculate_sag(x, y, R, k)
        element_opd_mm = calculate_element_opd(x, y, R, k, is_mirror=True)
        element_opd_waves = element_opd_mm / wavelength_mm
        print(f"{x:<10.2f} {y:<10.2f} {sag:<12.6f} {element_opd_mm:<15.6f} {element_opd_waves:<15.4f}")
    
    print(f"\n结论：")
    print(f"  元件 OPD 只与光线在表面上的交点位置有关")
    print(f"  与光线到达表面的几何光程无关")
    print(f"  这是正确的混合光学仿真方法")


if __name__ == "__main__":
    test_flat_mirror_45deg()
    test_parabolic_mirror_45deg()
    test_opd_sign_interpretation()
    test_element_opd_from_sag()
