"""
测试元件引入的 OPD 计算方法

核心思路：
1. 元件引入的 OPD = 由于表面形状引入的光程差
2. 对于平面镜：元件 OPD = 0（不改变波前形状）
3. 对于曲面镜：元件 OPD = 2 * sag（反射加倍）
4. 折叠光路的几何光程不应计入 OPD

计算方法：
- 方法 A：使用 optiland 追迹，但只取元件表面处的 OPD 变化
- 方法 B：直接计算表面矢高，OPD = 2 * sag（反射镜）
- 方法 C：使用参考光线（主光线）计算相对 OPD
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_parabolic_sag(x, y, R, k=-1):
    """计算抛物面的矢高
    
    抛物面方程：z = (x² + y²) / (R * (1 + sqrt(1 - (1+k)(x² + y²)/R²)))
    对于抛物面 k = -1，简化为：z = (x² + y²) / (2R)
    
    参数:
        x, y: 坐标
        R: 顶点曲率半径
        k: 圆锥常数，抛物面 k = -1
    
    返回:
        矢高 z
    """
    r_sq = x**2 + y**2
    if k == -1:
        # 抛物面简化公式
        return r_sq / (2 * R)
    else:
        # 一般圆锥曲线
        return r_sq / (R * (1 + np.sqrt(1 - (1 + k) * r_sq / R**2)))


def test_parabolic_mirror_sag_opd():
    """测试抛物面镜的矢高 OPD 计算"""
    print_section("测试抛物面镜的矢高 OPD 计算")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    f = -50.0  # mm，凸面镜
    R = 2 * f  # -100mm，曲率半径
    
    # 测试点
    positions = [
        (0, 0),
        (5, 0),
        (0, 5),
        (10, 0),
        (7.07, 7.07),  # r = 10
    ]
    
    print(f"抛物面镜参数：f = {f} mm, R = {R} mm")
    print(f"\n{'位置':<15} {'r':<10} {'sag':<12} {'OPD(2*sag)':<12} {'OPD(waves)':<12}")
    print("-" * 65)
    
    for x, y in positions:
        r = np.sqrt(x**2 + y**2)
        sag = calculate_parabolic_sag(x, y, R, k=-1)
        opd_mm = 2 * sag  # 反射加倍
        opd_waves = opd_mm / wavelength_mm
        
        print(f"({x:5.2f}, {y:5.2f})  {r:<10.2f} {sag:<12.6f} {opd_mm:<12.6f} {opd_waves:<12.4f}")
    
    print(f"\n理论公式：")
    print(f"  抛物面矢高：sag = r² / (2R) = r² / {2*R}")
    print(f"  反射 OPD：OPD = 2 * sag = r² / R = r² / {R}")


def test_optiland_surface_opd():
    """测试 optiland 在单个表面处的 OPD 计算"""
    print_section("测试 optiland 在单个表面处的 OPD 计算")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    f = -50.0  # mm
    R = 2 * f  # -100mm
    
    # 创建简单系统：物面 -> 抛物面镜 -> 像面（紧贴镜面）
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=30.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(
        index=1,
        radius=R,
        thickness=0.0,  # 像面紧贴镜面
        material='mirror',
        is_stop=True,
        conic=-1.0,  # 抛物面
    )
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 测试光线
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
    input_rays.opd = np.zeros(n_rays)
    
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
    
    # 分析
    opd_mm = np.asarray(traced_rays.opd)
    chief_opd = opd_mm[0]
    relative_opd_mm = opd_mm - chief_opd
    relative_opd_waves = relative_opd_mm / wavelength_mm
    
    print(f"optiland 追迹结果（无倾斜）：")
    print(f"\n{'x':<10} {'optiland OPD':<15} {'相对 OPD':<15} {'理论 OPD':<15} {'误差':<10}")
    print("-" * 70)
    
    for i, x in enumerate(x_positions):
        r = abs(x)
        sag = calculate_parabolic_sag(x, 0, R, k=-1)
        theory_opd_mm = 2 * sag
        theory_opd_waves = theory_opd_mm / wavelength_mm
        
        # 相对于主光线的理论 OPD
        theory_relative = theory_opd_waves - 0  # 主光线 OPD = 0
        
        error = relative_opd_waves[i] - theory_relative
        
        print(f"{x:<10.1f} {opd_mm[i]:<15.6f} {relative_opd_waves[i]:<15.4f} "
              f"{theory_relative:<15.4f} {error:<10.4f}")


def test_tilted_parabolic_mirror_opd():
    """测试倾斜抛物面镜的 OPD 计算"""
    print_section("测试倾斜抛物面镜的 OPD 计算")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    f = -50.0  # mm
    R = 2 * f  # -100mm
    tilt_x = np.pi/4 + 1e-10
    
    # 创建系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=30.0)
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
    # 出射面垂直于出射光轴
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
    input_rays.opd = np.zeros(n_rays)
    
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
    
    # 分析
    opd_mm = np.asarray(traced_rays.opd)
    chief_opd = opd_mm[0]
    relative_opd_mm = opd_mm - chief_opd
    relative_opd_waves = relative_opd_mm / wavelength_mm
    
    print(f"optiland 追迹结果（45° 倾斜）：")
    print(f"\n{'x':<10} {'optiland OPD':<15} {'相对 OPD':<15} {'理论 OPD':<15}")
    print("-" * 55)
    
    for i, x in enumerate(x_positions):
        r = abs(x)
        sag = calculate_parabolic_sag(x, 0, R, k=-1)
        theory_opd_waves = 2 * sag / wavelength_mm
        
        print(f"{x:<10.1f} {opd_mm[i]:<15.6f} {relative_opd_waves[i]:<15.4f} "
              f"{theory_opd_waves:<15.4f}")
    
    print(f"\n注意：倾斜情况下，optiland 的 OPD 包含了到达倾斜表面的几何光程")
    print(f"这不是我们想要的元件 OPD")


def test_correct_element_opd_method():
    """测试正确的元件 OPD 计算方法"""
    print_section("正确的元件 OPD 计算方法")
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    f = -50.0  # mm
    R = 2 * f  # -100mm
    
    print("""
正确的方法：直接计算表面矢高引入的 OPD

对于反射镜：
  OPD = 2 * sag(x, y)
  
其中 sag 是表面矢高（相对于顶点切平面）

对于抛物面（k = -1）：
  sag = r² / (2R)
  OPD = 2 * r² / (2R) = r² / R

对于球面（k = 0）：
  sag = R - sqrt(R² - r²) ≈ r² / (2R) （近轴近似）
  OPD ≈ r² / R

这个方法的优点：
1. 不依赖 optiland 的 OPD 计算
2. 不受倾斜角度影响
3. 物理意义明确：只计算表面形状引入的光程差
""")
    
    # 验证
    print(f"\n验证：抛物面镜 (f={f}mm, R={R}mm)")
    print(f"\n{'r (mm)':<10} {'sag (mm)':<12} {'OPD (mm)':<12} {'OPD (waves)':<12}")
    print("-" * 50)
    
    for r in [0, 5, 10, 15, 20]:
        sag = r**2 / (2 * R)
        opd_mm = 2 * sag
        opd_waves = opd_mm / wavelength_mm
        print(f"{r:<10} {sag:<12.6f} {opd_mm:<12.6f} {opd_waves:<12.4f}")


def test_flat_mirror_element_opd():
    """测试平面镜的元件 OPD"""
    print_section("平面镜的元件 OPD")
    
    print("""
对于平面镜：
  sag = 0（平面没有矢高）
  OPD = 2 * sag = 0

因此，平面镜不引入任何元件 OPD。
折叠光路的几何光程变化不应计入元件 OPD。

这与 optiland 的计算结果不同：
- optiland 计算的是从入射面到出射面的总几何光程
- 我们需要的是元件表面形状引入的光程差

结论：
对于折叠光路，不能直接使用 optiland 的 OPD 输出。
需要单独计算元件引入的 OPD（基于表面矢高）。
""")


def test_implementation_strategy():
    """测试实现策略"""
    print_section("实现策略")
    
    print("""
ElementRaytracer 的正确实现策略：

1. 使用 optiland 进行光线追迹：
   - 计算光线与表面的交点
   - 计算反射/折射后的光线方向
   - 获取出射光线的位置和方向

2. 单独计算元件 OPD：
   - 对于每条光线，计算其在表面上的交点坐标
   - 根据表面方程计算该点的矢高
   - OPD = 2 * sag（反射镜）或 OPD = (n-1) * sag（折射面）

3. 不使用 optiland 的 OPD 输出：
   - optiland 的 OPD 包含几何光程，不适用于折叠光路
   - 只使用 optiland 的光线位置和方向输出

4. 坐标转换：
   - 入射光线：从入射面局部坐标系转换到 optiland 坐标系
   - 出射光线：从 optiland 坐标系转换到出射面局部坐标系
   - 这些转换不影响 OPD 计算

代码示例：

```python
def calculate_element_opd(self, rays_at_surface):
    '''计算元件引入的 OPD'''
    x = rays_at_surface.x
    y = rays_at_surface.y
    
    surface = self.surfaces[0]
    
    if surface.is_plane:
        # 平面镜：OPD = 0
        return np.zeros(len(x))
    
    # 计算矢高
    r_sq = x**2 + y**2
    R = surface.radius
    k = surface.conic
    
    if k == -1:
        # 抛物面
        sag = r_sq / (2 * R)
    else:
        # 一般圆锥曲线
        sag = r_sq / (R * (1 + np.sqrt(1 - (1+k) * r_sq / R**2)))
    
    # 反射镜 OPD = 2 * sag
    if surface.is_mirror:
        opd_mm = 2 * sag
    else:
        # 折射面：需要考虑折射率
        opd_mm = (n - 1) * sag
    
    return opd_mm
```
""")


if __name__ == "__main__":
    test_parabolic_mirror_sag_opd()
    test_optiland_surface_opd()
    test_tilted_parabolic_mirror_opd()
    test_correct_element_opd_method()
    test_flat_mirror_element_opd()
    test_implementation_strategy()
