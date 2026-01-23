"""
详细分析球面镜 OPD 计算的每一步

目标：找出球面镜残差 OPD ~0.8 waves 的来源

理论分析：
1. 对于球面镜，入射平面波反射后变成球面波
2. 球面波的 OPD = r²/(2R)，其中 R 是曲率半径
3. 反射镜的 OPD = 2 × sag = 2 × r²/(2R) = r²/R（因为光线往返）

但是：
- optiland 计算的是从入射面到出射面的几何路径长度
- 出射面不在镜面顶点，而是在主光线与出射面的交点
- 这可能导致 OPD 计算的差异
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_spherical_mirror_opd():
    """详细分析球面镜 OPD"""
    
    print("=" * 70)
    print("球面镜 OPD 详细分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    R = 200.0  # 曲率半径 mm
    f = R / 2  # 焦距 100 mm
    
    # 测试光线位置
    r_values = np.array([0, 2, 4, 6, 8, 10])  # mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm = {wavelength_mm} mm")
    print(f"  曲率半径 R: {R} mm")
    print(f"  焦距 f: {f} mm")
    
    # ========== 理论计算 ==========
    print(f"\n{'='*70}")
    print("理论计算")
    print(f"{'='*70}")
    
    # 1. 表面矢高 (sag)
    # sag = r²/(2R) 对于球面
    sag = r_values**2 / (2 * R)
    print(f"\n1. 表面矢高 sag = r²/(2R):")
    for i, r in enumerate(r_values):
        print(f"   r = {r:5.1f} mm: sag = {sag[i]:.6f} mm")
    
    # 2. 反射镜 OPD（几何路径差）
    # 边缘光线比主光线多走的路径 = 2 × sag（往返）
    opd_geometric = 2 * sag
    print(f"\n2. 几何 OPD = 2 × sag（往返）:")
    for i, r in enumerate(r_values):
        opd_waves = opd_geometric[i] / wavelength_mm
        print(f"   r = {r:5.1f} mm: OPD = {opd_geometric[i]:.6f} mm = {opd_waves:.2f} waves")
    
    # 3. Pilot Beam OPD
    # 出射波前曲率半径 R_out = -f = -R/2 = -100 mm
    R_out = -f
    pilot_opd = r_values**2 / (2 * R_out)
    print(f"\n3. Pilot Beam OPD = r²/(2R_out), R_out = {R_out} mm:")
    for i, r in enumerate(r_values):
        opd_waves = pilot_opd[i] / wavelength_mm
        print(f"   r = {r:5.1f} mm: OPD = {pilot_opd[i]:.6f} mm = {opd_waves:.2f} waves")
    
    # 4. 理论残差 OPD
    # 残差 = 几何 OPD + Pilot Beam OPD
    # = r²/R + r²/(2R_out)
    # = r²/R + r²/(2×(-R/2))
    # = r²/R - r²/R
    # = 0 ✓
    residual_theory = opd_geometric + pilot_opd
    print(f"\n4. 理论残差 OPD = 几何 OPD + Pilot Beam OPD:")
    for i, r in enumerate(r_values):
        opd_waves = residual_theory[i] / wavelength_mm
        print(f"   r = {r:5.1f} mm: 残差 = {residual_theory[i]:.6f} mm = {opd_waves:.4f} waves")
    
    # ========== 实际光线追迹 ==========
    print(f"\n{'='*70}")
    print("实际光线追迹")
    print(f"{'='*70}")
    
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
    # 创建表面定义
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
    )
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=(0, 0, -1),
    )
    
    # 创建测试光线（沿 Y 轴分布）
    n_rays = len(r_values)
    input_rays = RealRays(
        x=np.zeros(n_rays),
        y=r_values.astype(float),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    input_rays.opd = np.zeros(n_rays)
    
    # 执行光线追迹
    output_rays = raytracer.trace(input_rays)
    
    # 获取结果
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    opd_out = np.asarray(output_rays.opd)
    
    print(f"\n5. 光线追迹结果:")
    print(f"   {'r_in':>6} {'y_out':>10} {'z_out':>10} {'OPD(mm)':>12} {'OPD(waves)':>12}")
    for i, r in enumerate(r_values):
        opd_waves = opd_out[i] / wavelength_mm
        print(f"   {r:6.1f} {y_out[i]:10.4f} {z_out[i]:10.6f} {opd_out[i]:12.6f} {opd_waves:12.2f}")
    
    # 相对于主光线的 OPD
    relative_opd = opd_out - opd_out[0]
    print(f"\n6. 相对 OPD（相对于主光线）:")
    print(f"   {'r_in':>6} {'相对OPD(mm)':>14} {'相对OPD(waves)':>16} {'理论(waves)':>14} {'差异(waves)':>14}")
    for i, r in enumerate(r_values):
        rel_opd_waves = relative_opd[i] / wavelength_mm
        theory_waves = opd_geometric[i] / wavelength_mm
        diff_waves = rel_opd_waves - theory_waves
        print(f"   {r:6.1f} {relative_opd[i]:14.6f} {rel_opd_waves:16.4f} {theory_waves:14.4f} {diff_waves:14.4f}")
    
    # ========== 分析出射面位置 ==========
    print(f"\n{'='*70}")
    print("出射面位置分析")
    print(f"{'='*70}")
    
    # 主光线在镜面上的交点
    # 对于正入射，主光线与球面镜的交点在 z = 0（顶点）
    # 但出射面是垂直于出射光轴的平面
    
    # 出射光轴方向：(0, 0, -1)
    # 出射面原点：主光线与镜面的交点 = (0, 0, 0)
    
    print(f"\n主光线与镜面交点: (0, 0, 0)")
    print(f"出射光轴方向: (0, 0, -1)")
    print(f"出射面: z = 0 平面")
    
    # 边缘光线与镜面的交点
    # 对于 y = r 的入射光线，与球面的交点在 z = sag
    print(f"\n边缘光线与镜面交点:")
    for i, r in enumerate(r_values):
        print(f"   r = {r:5.1f} mm: 交点 z = {sag[i]:.6f} mm")
    
    # 边缘光线从镜面到出射面的额外路径
    # 出射面在 z = 0，边缘光线从 z = sag 反射后需要传播到 z = 0
    # 但反射后光线方向改变了！
    
    print(f"\n边缘光线反射后的方向:")
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    for i, r in enumerate(r_values):
        print(f"   r = {r:5.1f} mm: 方向 = ({L_out[i]:.6f}, {M_out[i]:.6f}, {N_out[i]:.6f})")
    
    # ========== 详细路径分析 ==========
    print(f"\n{'='*70}")
    print("详细路径分析")
    print(f"{'='*70}")
    
    # 入射路径：从 z=0 到镜面
    # 对于主光线：路径 = 0（镜面顶点在 z=0）
    # 对于边缘光线：路径 = sag / cos(入射角) ≈ sag（小角度近似）
    
    # 但实际上，optiland 计算的是光线到表面的距离 t
    # 对于球面，t 不等于 sag
    
    # 球面方程：x² + y² + (z - R)² = R²
    # 简化：x² + y² + z² - 2Rz = 0
    # 对于 z 轴入射光线 (0, y, 0) + t*(0, 0, 1)
    # 交点：y² + t² - 2Rt = 0
    # t = R - sqrt(R² - y²)
    
    print(f"\n入射光线到镜面的距离 t:")
    t_to_surface = R - np.sqrt(R**2 - r_values**2)
    for i, r in enumerate(r_values):
        print(f"   r = {r:5.1f} mm: t = {t_to_surface[i]:.6f} mm, sag = {sag[i]:.6f} mm, 差异 = {t_to_surface[i] - sag[i]:.6f} mm")
    
    # 反射后光线方向
    # 表面法向量在交点处：n = (x, y, z-R) / R
    # 对于 (0, y, sag) 点：n = (0, y/R, (sag-R)/R) = (0, y/R, -sqrt(1-(y/R)²))
    
    print(f"\n表面法向量（在交点处）:")
    for i, r in enumerate(r_values):
        ny = r / R
        nz = -np.sqrt(1 - (r/R)**2)
        print(f"   r = {r:5.1f} mm: n = (0, {ny:.6f}, {nz:.6f})")
    
    # 反射方向：d' = d - 2(d·n)n
    # 入射方向 d = (0, 0, 1)
    # d·n = nz
    # d' = (0, 0, 1) - 2*nz*(0, ny, nz) = (0, -2*nz*ny, 1 - 2*nz²)
    
    print(f"\n理论反射方向:")
    for i, r in enumerate(r_values):
        ny = r / R
        nz = -np.sqrt(1 - (r/R)**2)
        L_theory = 0
        M_theory = -2 * nz * ny
        N_theory = 1 - 2 * nz**2
        print(f"   r = {r:5.1f} mm: d' = ({L_theory:.6f}, {M_theory:.6f}, {N_theory:.6f})")
        print(f"              实际: d' = ({L_out[i]:.6f}, {M_out[i]:.6f}, {N_out[i]:.6f})")
    
    # ========== OPD 详细计算 ==========
    print(f"\n{'='*70}")
    print("OPD 详细计算")
    print(f"{'='*70}")
    
    # optiland 计算的 OPD = |t| * n
    # 对于空气 n = 1，所以 OPD = |t|
    # 这是从入射面到镜面的路径
    
    # 但是，光线追迹后还需要从镜面传播到出射面
    # 出射面在 z = 0
    # 边缘光线在镜面的位置是 (0, y, sag)
    # 反射后需要传播到 z = 0 平面
    
    # 从 (0, y, sag) 沿方向 (L', M', N') 传播到 z = 0
    # z + t * N' = 0
    # t = -sag / N'
    
    print(f"\n从镜面到出射面的传播距离:")
    for i, r in enumerate(r_values):
        if abs(N_out[i]) > 1e-10:
            t_to_exit = -sag[i] / N_out[i]
            print(f"   r = {r:5.1f} mm: t_exit = {t_to_exit:.6f} mm")
        else:
            print(f"   r = {r:5.1f} mm: N' ≈ 0, 无法计算")
    
    # 总 OPD = 入射路径 + 出射路径
    print(f"\n总 OPD 计算:")
    print(f"   {'r':>6} {'t_in':>10} {'t_out':>10} {'总OPD':>12} {'实际OPD':>12} {'差异':>12}")
    for i, r in enumerate(r_values):
        t_in = t_to_surface[i]
        if abs(N_out[i]) > 1e-10:
            t_out = -sag[i] / N_out[i]
        else:
            t_out = 0
        total_opd = t_in + abs(t_out)
        diff = opd_out[i] - total_opd
        print(f"   {r:6.1f} {t_in:10.6f} {abs(t_out):10.6f} {total_opd:12.6f} {opd_out[i]:12.6f} {diff:12.6f}")
    
    # ========== 残差分析 ==========
    print(f"\n{'='*70}")
    print("残差 OPD 分析")
    print(f"{'='*70}")
    
    # 计算 Pilot Beam OPD
    R_out = -f  # -100 mm
    pilot_opd_mm = r_values**2 / (2 * R_out)
    pilot_opd_waves = pilot_opd_mm / wavelength_mm
    
    # 相对于主光线
    pilot_opd_rel = pilot_opd_mm - pilot_opd_mm[0]
    pilot_opd_rel_waves = pilot_opd_rel / wavelength_mm
    
    # 残差 = 实际 OPD + Pilot Beam OPD
    residual = relative_opd + pilot_opd_rel
    residual_waves = residual / wavelength_mm
    
    print(f"\n残差 OPD = 实际相对 OPD + Pilot Beam 相对 OPD:")
    print(f"   {'r':>6} {'实际OPD':>14} {'Pilot OPD':>14} {'残差':>14} {'残差(waves)':>14}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {relative_opd[i]:14.6f} {pilot_opd_rel[i]:14.6f} {residual[i]:14.6f} {residual_waves[i]:14.4f}")
    
    # ========== 根本原因分析 ==========
    print(f"\n{'='*70}")
    print("根本原因分析")
    print(f"{'='*70}")
    
    # 理论上，对于球面镜：
    # 几何 OPD = 2 × sag = r²/R
    # Pilot Beam OPD = r²/(2R_out) = r²/(2×(-R/2)) = -r²/R
    # 残差 = r²/R + (-r²/R) = 0
    
    # 但实际上：
    # 1. 入射路径 t_in = R - sqrt(R² - r²) ≠ sag = r²/(2R)
    # 2. 出射路径取决于反射方向和出射面位置
    
    print(f"\n比较 t_in 与 sag:")
    print(f"   t_in = R - sqrt(R² - r²)")
    print(f"   sag = r²/(2R)")
    print(f"   差异 = t_in - sag")
    print(f"\n   {'r':>6} {'t_in':>12} {'sag':>12} {'差异':>12} {'差异(waves)':>14}")
    for i, r in enumerate(r_values):
        diff = t_to_surface[i] - sag[i]
        diff_waves = diff / wavelength_mm
        print(f"   {r:6.1f} {t_to_surface[i]:12.6f} {sag[i]:12.6f} {diff:12.6f} {diff_waves:14.4f}")
    
    # 泰勒展开分析
    # t_in = R - sqrt(R² - r²)
    #      = R - R*sqrt(1 - (r/R)²)
    #      ≈ R - R*(1 - (r/R)²/2 - (r/R)⁴/8 - ...)
    #      = r²/(2R) + r⁴/(8R³) + ...
    #      = sag + r⁴/(8R³) + O(r⁶)
    
    print(f"\n泰勒展开分析:")
    print(f"   t_in ≈ sag + r⁴/(8R³) + O(r⁶)")
    print(f"   高阶项 = r⁴/(8R³)")
    print(f"\n   {'r':>6} {'高阶项':>12} {'高阶项(waves)':>16} {'实际差异(waves)':>18}")
    for i, r in enumerate(r_values):
        higher_order = r**4 / (8 * R**3)
        higher_order_waves = higher_order / wavelength_mm
        actual_diff = (t_to_surface[i] - sag[i]) / wavelength_mm
        print(f"   {r:6.1f} {higher_order:12.6f} {higher_order_waves:16.4f} {actual_diff:18.4f}")
    
    print(f"\n结论:")
    print(f"   残差 OPD 主要来自球面的高阶项 r⁴/(8R³)")
    print(f"   这是球面与抛物面的差异（球差）")
    print(f"   对于 r=10mm, R=200mm: 高阶项 ≈ {10**4/(8*200**3):.6f} mm = {10**4/(8*200**3)/wavelength_mm:.4f} waves")


if __name__ == "__main__":
    analyze_spherical_mirror_opd()
