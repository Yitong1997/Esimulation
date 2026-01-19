"""
详细分析 OPD 计算过程，理解带符号 OPD 的正确性
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def analyze_normal_incidence_mirror():
    """分析正入射曲面镜的 OPD 计算"""
    print("=" * 70)
    print("正入射曲面镜 OPD 详细分析")
    print("=" * 70)
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    R = 200.0  # 凹面镜
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=30.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=R, thickness=0.0, 
                      material='mirror', is_stop=True)
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 测试光线
    x_vals = [0.0, 5.0, 10.0]
    n_rays = len(x_vals)
    
    input_rays = RealRays(
        x=np.array(x_vals),
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 手动追迹，记录每一步
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
    rays.opd = np.zeros(n_rays)
    
    surface = optic.surface_group.surfaces[1]
    
    print(f"\n初始状态:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  z = {np.asarray(rays.z)}")
    print(f"  N = {np.asarray(rays.N)}")
    
    # 坐标变换
    surface.geometry.localize(rays)
    print(f"\n局部坐标系:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  z = {np.asarray(rays.z)}")
    
    # 计算距离
    t = np.asarray(surface.geometry.distance(rays))
    print(f"\n传播距离 t = {t}")
    
    # 计算矢高（理论值）
    r2 = np.asarray(rays.x)**2 + np.asarray(rays.y)**2
    sag_theory = r2 / (R * (1 + np.sqrt(1 - r2 / R**2)))
    print(f"理论矢高 sag = {sag_theory}")
    
    # 传播
    surface.material_pre.propagation_model.propagate(rays, t)
    print(f"\n传播后:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  z = {np.asarray(rays.z)}")
    
    # 带符号 OPD
    n = 1.0  # 空气
    opd_signed = n * t
    print(f"\n带符号 OPD = n * t = {opd_signed}")
    
    # 理论 OPD（基于矢高）
    opd_theory = 2 * sag_theory  # 反射加倍
    print(f"理论 OPD = 2 * sag = {opd_theory}")
    
    # 分析
    print(f"\n分析:")
    print(f"  对于正入射，光线从 z=0 传播到表面")
    print(f"  传播距离 t = sag（表面矢高）")
    print(f"  带符号 OPD = t（单程）")
    print(f"  但反射镜的 OPD 应该是 2 * sag（往返）")
    print(f"\n  问题：带符号 OPD 只计算了单程，没有考虑反射")
    
    # 检查 optiland 的 OPD
    rays2 = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    rays2.opd = np.zeros(n_rays)
    optic.surface_group.trace(rays2, skip=1)
    
    print(f"\noptiland 的 OPD = {np.asarray(rays2.opd)}")
    print(f"optiland 使用 abs(t)，所以 OPD = |t| = {np.abs(t)}")
    
    # 相对 OPD
    chief_opd_signed = opd_signed[0]
    relative_opd_signed = opd_signed - chief_opd_signed
    
    chief_opd_theory = opd_theory[0]
    relative_opd_theory = opd_theory - chief_opd_theory
    
    print(f"\n相对 OPD（带符号）= {relative_opd_signed}")
    print(f"相对 OPD（理论）= {relative_opd_theory}")
    print(f"比值 = {relative_opd_signed / (relative_opd_theory + 1e-10)}")
    
    print(f"\n结论:")
    print(f"  带符号 OPD 计算的是单程光程")
    print(f"  对于反射镜，需要乘以 2 才能得到正确的 OPD")
    print(f"  但相对 OPD 的比值是 0.5，说明带符号方法只计算了一半")


def analyze_tilted_mirror():
    """分析倾斜镜的 OPD 计算"""
    print("\n" + "=" * 70)
    print("45° 倾斜曲面镜 OPD 详细分析")
    print("=" * 70)
    
    wavelength_um = 0.55
    R = -100.0  # 凸面镜
    tilt_x = np.pi/4 + 1e-10
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=20.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=R, thickness=0.0, 
                      material='mirror', is_stop=True, conic=-1.0, rx=tilt_x)
    optic.add_surface(index=2, radius=np.inf, thickness=0.0, rx=np.pi/2)
    
    # 测试光线
    x_vals = [0.0, 5.0, 10.0]
    n_rays = len(x_vals)
    
    input_rays = RealRays(
        x=np.array(x_vals),
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 手动追迹
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
    
    surface = optic.surface_group.surfaces[1]
    
    print(f"\n初始状态（全局坐标系）:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  y = {np.asarray(rays.y)}")
    print(f"  z = {np.asarray(rays.z)}")
    
    # 坐标变换到表面局部坐标系
    surface.geometry.localize(rays)
    print(f"\n表面局部坐标系（45° 倾斜）:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  y = {np.asarray(rays.y)}")
    print(f"  z = {np.asarray(rays.z)}")
    print(f"  L = {np.asarray(rays.L)}")
    print(f"  M = {np.asarray(rays.M)}")
    print(f"  N = {np.asarray(rays.N)}")
    
    # 计算距离
    t = np.asarray(surface.geometry.distance(rays))
    print(f"\n传播距离 t = {t}")
    
    # 传播到表面
    surface.material_pre.propagation_model.propagate(rays, t)
    print(f"\n到达表面后（局部坐标系）:")
    print(f"  x = {np.asarray(rays.x)}")
    print(f"  y = {np.asarray(rays.y)}")
    print(f"  z = {np.asarray(rays.z)}")
    
    # 计算表面矢高
    x_at_surface = np.asarray(rays.x)
    y_at_surface = np.asarray(rays.y)
    r2 = x_at_surface**2 + y_at_surface**2
    sag = r2 / (R * (1 + np.sqrt(1 - 2 * r2 / R**2)))  # k=-1 抛物面
    print(f"\n表面矢高 sag = {sag}")
    
    # 带符号 OPD
    opd_signed = t  # n=1
    print(f"带符号 OPD = t = {opd_signed}")
    
    # 理论元件 OPD
    opd_element = 2 * sag
    print(f"理论元件 OPD = 2 * sag = {opd_element}")
    
    print(f"\n分析:")
    print(f"  对于倾斜镜，t 包含两部分：")
    print(f"  1. 到达倾斜平面的几何光程（与 y 位置有关）")
    print(f"  2. 表面矢高引入的光程")
    print(f"  带符号 OPD 正确地将这两部分抵消/累加")


if __name__ == "__main__":
    analyze_normal_incidence_mirror()
    analyze_tilted_mirror()
