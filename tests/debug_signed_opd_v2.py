"""
详细调试带符号 OPD 计算

分析：
1. 对于正入射曲面镜，光线从 z=0 传播到表面
2. 传播距离 t = sag（表面矢高）
3. 带符号 OPD = n * t = sag（单程）
4. 但反射镜的 OPD 应该是什么？

关键问题：反射镜的 OPD 是单程还是双程？

在光学设计中，OPD 通常定义为：
- 从参考球面到实际波前的光程差
- 对于反射镜，光线到达表面后反射回来
- 但 OPD 计算的是到达表面的光程，不是往返光程

让我们验证这一点。
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def calculate_sag(x, y, R, k=0.0):
    """计算圆锥曲面的矢高"""
    r2 = np.asarray(x)**2 + np.asarray(y)**2
    if np.isinf(R):
        return np.zeros_like(r2)
    return r2 / (R * (1 + np.sqrt(1 - (1 + k) * r2 / R**2)))


def test_opd_definition():
    """测试 OPD 的定义"""
    print("=" * 70)
    print("OPD 定义测试")
    print("=" * 70)
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    R = 200.0  # 凹面镜
    
    # 创建光学系统
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
    x_vals = [0.0, 5.0, 10.0, 15.0]
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
    
    # 坐标变换
    surface.geometry.localize(rays)
    
    # 计算距离
    t = np.asarray(surface.geometry.distance(rays))
    
    # 计算矢高（理论值）
    x_at_start = np.asarray(rays.x)
    y_at_start = np.asarray(rays.y)
    sag_theory = calculate_sag(x_at_start, y_at_start, R, k=0)
    
    print(f"\n光线位置 x: {x_vals}")
    print(f"传播距离 t: {t}")
    print(f"理论矢高 sag: {sag_theory}")
    print(f"t 与 sag 的差异: {t - sag_theory}")
    
    # 带符号 OPD
    n = 1.0  # 空气
    opd_signed = n * t
    
    # 相对 OPD
    chief_opd = opd_signed[0]
    relative_opd = opd_signed - chief_opd
    relative_opd_waves = relative_opd / wavelength_mm
    
    # 理论相对 OPD（基于矢高）
    # 问题：反射镜的 OPD 是 sag 还是 2*sag？
    
    # 方案 1：OPD = sag（单程）
    opd_theory_single = sag_theory
    relative_theory_single = opd_theory_single - opd_theory_single[0]
    relative_theory_single_waves = relative_theory_single / wavelength_mm
    
    # 方案 2：OPD = 2*sag（双程）
    opd_theory_double = 2 * sag_theory
    relative_theory_double = opd_theory_double - opd_theory_double[0]
    relative_theory_double_waves = relative_theory_double / wavelength_mm
    
    print(f"\n{'x':<10} {'实测OPD':<15} {'理论(单程)':<15} {'理论(双程)':<15}")
    print("-" * 55)
    for i, x in enumerate(x_vals):
        print(f"{x:<10.1f} {relative_opd_waves[i]:<15.4f} "
              f"{relative_theory_single_waves[i]:<15.4f} "
              f"{relative_theory_double_waves[i]:<15.4f}")
    
    print(f"\n分析：")
    print(f"  实测 OPD 与单程理论的比值: {relative_opd_waves[-1] / relative_theory_single_waves[-1]:.4f}")
    print(f"  实测 OPD 与双程理论的比值: {relative_opd_waves[-1] / relative_theory_double_waves[-1]:.4f}")
    
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
    
    optiland_opd = np.asarray(rays2.opd)
    optiland_relative = optiland_opd - optiland_opd[0]
    optiland_relative_waves = optiland_relative / wavelength_mm
    
    print(f"\noptiland 的相对 OPD (waves): {optiland_relative_waves}")
    print(f"optiland 与单程理论的比值: {optiland_relative_waves[-1] / relative_theory_single_waves[-1]:.4f}")
    print(f"optiland 与双程理论的比值: {optiland_relative_waves[-1] / relative_theory_double_waves[-1]:.4f}")


def test_reflection_opd():
    """测试反射镜的 OPD 物理意义"""
    print("\n" + "=" * 70)
    print("反射镜 OPD 物理意义测试")
    print("=" * 70)
    
    print("""
    在光学设计中，OPD（光程差）的定义：
    
    1. 对于透射系统：
       OPD = 光线实际光程 - 参考光线光程
       
    2. 对于反射系统：
       - 光线到达反射面的光程
       - 反射后继续传播的光程
       - OPD 是累积的
       
    关键问题：反射面本身引入的 OPD 是什么？
    
    答案：反射面引入的 OPD = 2 * sag
    
    原因：
    - 光线从参考平面传播到反射面，光程 = sag
    - 反射后，光线从反射面传播回参考平面，光程 = sag
    - 总光程差 = 2 * sag
    
    但是，在 optiland 的实现中：
    - trace() 方法只计算到达表面的光程
    - 反射后的光程在下一个表面计算
    - 所以单个反射面的 OPD = sag（单程）
    
    这意味着：
    - 如果我们只追迹到反射面，OPD = sag
    - 如果我们追迹到反射面后的出射面，OPD = 2 * sag
    """)
    
    wavelength_um = 0.55
    wavelength_mm = wavelength_um * 1e-3
    R = 200.0  # 凹面镜
    
    # 创建光学系统（只有反射面，没有出射面）
    optic1 = Optic()
    optic1.set_aperture(aperture_type='EPD', value=30.0)
    optic1.set_field_type(field_type='angle')
    optic1.add_field(y=0, x=0)
    optic1.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic1.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic1.add_surface(index=1, radius=R, thickness=0.0, 
                       material='mirror', is_stop=True)
    
    # 创建光学系统（有反射面和出射面）
    optic2 = Optic()
    optic2.set_aperture(aperture_type='EPD', value=30.0)
    optic2.set_field_type(field_type='angle')
    optic2.add_field(y=0, x=0)
    optic2.add_wavelength(value=wavelength_um, is_primary=True)
    
    optic2.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic2.add_surface(index=1, radius=R, thickness=0.0, 
                       material='mirror', is_stop=True)
    optic2.add_surface(index=2, radius=np.inf, thickness=0.0)  # 出射面
    
    # 测试光线
    x_vals = [0.0, 10.0]
    n_rays = len(x_vals)
    
    def create_rays():
        return RealRays(
            x=np.array(x_vals),
            y=np.zeros(n_rays),
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
    
    # 追迹系统 1（只到反射面）
    rays1 = create_rays()
    rays1.opd = np.zeros(n_rays)
    optic1.surface_group.trace(rays1, skip=1)
    opd1 = np.asarray(rays1.opd)
    
    # 追迹系统 2（到出射面）
    rays2 = create_rays()
    rays2.opd = np.zeros(n_rays)
    optic2.surface_group.trace(rays2, skip=1)
    opd2 = np.asarray(rays2.opd)
    
    # 理论矢高
    sag = calculate_sag(np.array(x_vals), np.zeros(n_rays), R, k=0)
    
    print(f"\n光线位置 x: {x_vals}")
    print(f"理论矢高 sag: {sag}")
    print(f"\n系统 1（只到反射面）OPD: {opd1}")
    print(f"系统 2（到出射面）OPD: {opd2}")
    print(f"\n系统 1 OPD / sag: {opd1 / sag}")
    print(f"系统 2 OPD / sag: {opd2 / sag}")
    print(f"系统 2 OPD / (2*sag): {opd2 / (2*sag)}")
    
    print(f"\n结论：")
    print(f"  系统 1（只到反射面）：OPD ≈ sag（单程）")
    print(f"  系统 2（到出射面）：OPD ≈ 2*sag（双程）")


def test_signed_opd_with_exit_plane():
    """测试带符号 OPD 在有出射面时的行为"""
    print("\n" + "=" * 70)
    print("带符号 OPD 与出射面测试")
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
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)  # 出射面
    
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
    
    # 手动追迹，使用带符号 OPD
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
    
    surfaces = optic.surface_group.surfaces
    
    print(f"\n逐表面追迹：")
    for i, surface in enumerate(surfaces):
        if i < 1:  # 跳过物面
            continue
        
        print(f"\n--- 表面 {i} ---")
        print(f"  追迹前位置: z = {np.asarray(rays.z)}")
        print(f"  追迹前方向: N = {np.asarray(rays.N)}")
        
        # 坐标变换
        surface.geometry.localize(rays)
        
        # 计算距离
        t = np.asarray(surface.geometry.distance(rays))
        print(f"  传播距离 t: {t}")
        
        # 带符号 OPD
        n = surface.material_pre.n(rays.w)
        n = np.asarray(n)
        if n.ndim == 0:
            n = float(n)
        opd_increment = n * t
        print(f"  OPD 增量: {opd_increment}")
        
        # 传播
        surface.material_pre.propagation_model.propagate(rays, t)
        
        # 更新 OPD
        rays.opd = rays.opd + opd_increment
        print(f"  累积 OPD: {np.asarray(rays.opd)}")
        
        # 交互
        rays = surface.interaction_model.interact_real_rays(rays)
        print(f"  交互后方向: N = {np.asarray(rays.N)}")
        
        # 全局化
        surface.geometry.globalize(rays)
    
    # 计算相对 OPD
    opd_final = np.asarray(rays.opd)
    relative_opd = opd_final - opd_final[0]
    relative_opd_waves = relative_opd / wavelength_mm
    
    # 理论值
    sag = calculate_sag(np.array(x_vals), np.zeros(n_rays), R, k=0)
    theoretical_opd = 2 * sag  # 双程
    theoretical_relative = theoretical_opd - theoretical_opd[0]
    theoretical_relative_waves = theoretical_relative / wavelength_mm
    
    print(f"\n最终结果：")
    print(f"  带符号相对 OPD (waves): {relative_opd_waves}")
    print(f"  理论相对 OPD (waves): {theoretical_relative_waves}")
    print(f"  误差 (waves): {relative_opd_waves - theoretical_relative_waves}")


if __name__ == "__main__":
    test_opd_definition()
    test_reflection_opd()
    test_signed_opd_with_exit_plane()
