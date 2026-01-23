"""
详细分析抛物面镜的 OPD 计算

抛物面镜应该没有球差，但残差仍然不为零。
需要找出原因。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_parabola_opd():
    """分析抛物面镜 OPD"""
    
    print("=" * 70)
    print("抛物面镜 OPD 详细分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    R = 200.0  # 顶点曲率半径 mm
    f = R / 2  # 焦距 100 mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  顶点曲率半径 R: {R} mm")
    print(f"  焦距 f: {f} mm")
    
    # ========== 理论分析 ==========
    print(f"\n{'='*70}")
    print("理论分析：抛物面镜")
    print(f"{'='*70}")
    
    print(f"""
抛物面方程：z = r² / (2R)（与球面的近轴近似相同）

对于平行光入射：
1. 所有光线都聚焦到焦点 F = (0, 0, f)
2. 从入射面到焦点的光程相等（费马原理）
3. 反射后的波前是以焦点为中心的球面波

关键问题：
- 出射波前的曲率半径是多少？
- 对于抛物面镜，出射波前是否是理想球面波？
""")
    
    # ========== 光线追迹 ==========
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        conic=-1.0,  # 抛物面
    )
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=(0, 0, -1),
    )
    
    # 测试光线
    r_values = np.array([0, 2, 4, 6, 8, 10])
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
    
    output_rays = raytracer.trace(input_rays)
    
    y_in = r_values
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    opd_out = np.asarray(output_rays.opd)
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    
    print(f"\n光线追迹结果:")
    print(f"   {'y_in':>6} {'y_out':>10} {'z_out':>10} {'OPD(mm)':>12} {'M':>10} {'N':>10}")
    for i in range(n_rays):
        print(f"   {y_in[i]:6.1f} {y_out[i]:10.4f} {z_out[i]:10.6f} {opd_out[i]:12.6f} {M_out[i]:10.6f} {N_out[i]:10.6f}")
    
    # ========== 抛物面几何分析 ==========
    print(f"\n{'='*70}")
    print("抛物面几何分析")
    print(f"{'='*70}")
    
    # 抛物面方程：z = r² / (4f) = r² / (2R)
    # 对于 y = r 的入射光线，与抛物面的交点在 z = r² / (2R)
    sag = r_values**2 / (2 * R)
    
    print(f"\n抛物面矢高 sag = r²/(2R):")
    for i, r in enumerate(r_values):
        print(f"   r = {r:5.1f} mm: sag = {sag[i]:.6f} mm")
    
    # 抛物面法向量
    # z = r²/(2R)
    # ∂z/∂r = r/R
    # 法向量 n = (-∂z/∂r, 0, 1) / |...| = (-r/R, 0, 1) / sqrt(1 + (r/R)²)
    # 对于 y 方向：n = (0, -r/R, 1) / sqrt(1 + (r/R)²)
    
    print(f"\n抛物面法向量（在交点处）:")
    for i, r in enumerate(r_values):
        norm = np.sqrt(1 + (r/R)**2)
        ny = -r / R / norm
        nz = 1 / norm
        print(f"   r = {r:5.1f} mm: n = (0, {ny:.6f}, {nz:.6f})")
    
    # 反射方向
    # d = (0, 0, 1)
    # d' = d - 2(d·n)n
    print(f"\n反射方向:")
    for i, r in enumerate(r_values):
        norm = np.sqrt(1 + (r/R)**2)
        ny = -r / R / norm
        nz = 1 / norm
        d_dot_n = nz
        L_theory = 0
        M_theory = -2 * d_dot_n * ny
        N_theory = 1 - 2 * d_dot_n * nz
        print(f"   r = {r:5.1f} mm: 理论 d' = (0, {M_theory:.6f}, {N_theory:.6f})")
        print(f"              实际 d' = (0, {M_out[i]:.6f}, {N_out[i]:.6f})")
    
    # ========== 光程分析 ==========
    print(f"\n{'='*70}")
    print("光程分析")
    print(f"{'='*70}")
    
    # 对于抛物面镜，从入射面到焦点的光程应该相等
    # 入射路径：从 z=0 到镜面
    # 出射路径：从镜面到焦点
    
    # 入射路径 = sag（对于平行光）
    # 出射路径 = sqrt((y_mirror - 0)² + (z_mirror - f)²)
    #          = sqrt(r² + (sag - f)²)
    
    y_mirror = r_values
    z_mirror = sag
    
    path_in = sag
    path_out = np.sqrt(y_mirror**2 + (z_mirror - f)**2)
    total_path = path_in + path_out
    
    print(f"\n从入射面到焦点的光程:")
    print(f"   {'r':>6} {'入射路径':>12} {'出射路径':>12} {'总光程':>12} {'差异':>12}")
    for i, r in enumerate(r_values):
        diff = total_path[i] - total_path[0]
        print(f"   {r:6.1f} {path_in[i]:12.6f} {path_out[i]:12.6f} {total_path[i]:12.6f} {diff:12.6f}")
    
    # 验证：对于抛物面，总光程应该相等
    # 理论：入射路径 + 出射路径 = f（常数）
    # 证明：
    # sag + sqrt(r² + (sag - f)²) = r²/(2R) + sqrt(r² + (r²/(2R) - R/2)²)
    # = r²/(2R) + sqrt(r² + (r² - R²)²/(4R²))
    # = r²/(2R) + sqrt((4R²r² + r⁴ - 2R²r² + R⁴)/(4R²))
    # = r²/(2R) + sqrt((r⁴ + 2R²r² + R⁴)/(4R²))
    # = r²/(2R) + sqrt((r² + R²)²/(4R²))
    # = r²/(2R) + (r² + R²)/(2R)
    # = (r² + r² + R²)/(2R)
    # = (2r² + R²)/(2R)
    # = r²/R + R/2
    # = r²/R + f
    # 
    # 这不是常数！说明我的推导有误。
    
    # 重新推导：
    # 抛物面方程：z = r²/(4f)，其中 f = R/2
    # 所以 z = r²/(2R)
    # 
    # 从 (0, r, 0) 入射，到达镜面 (0, r, sag)
    # 入射路径 = sag = r²/(2R)
    # 
    # 从镜面反射到焦点 (0, 0, f)
    # 出射路径 = sqrt(r² + (sag - f)²)
    #          = sqrt(r² + (r²/(2R) - R/2)²)
    
    print(f"\n理论验证：")
    print(f"   对于抛物面，从入射面到焦点的光程应该是常数")
    print(f"   但上面的计算显示光程不是常数...")
    print(f"   这是因为出射面不在焦点，而是在 z=0 平面！")
    
    # ========== 出射面在 z=0 的情况 ==========
    print(f"\n{'='*70}")
    print("出射面在 z=0 的情况")
    print(f"{'='*70}")
    
    # 出射面在 z=0，不是焦点
    # 光线从镜面 (0, r, sag) 沿反射方向传播到 z=0
    
    # 反射方向 d' = (0, M', N')
    # 从 (0, r, sag) 沿 d' 传播到 z=0
    # z + t * N' = 0
    # t = -sag / N'
    
    print(f"\n从镜面到出射面 (z=0) 的传播:")
    print(f"   {'r':>6} {'sag':>10} {'N':>10} {'t_out':>12} {'y_out理论':>12} {'y_out实际':>12}")
    for i, r in enumerate(r_values):
        if abs(N_out[i]) > 1e-10:
            t_out = -sag[i] / N_out[i]
            y_out_theory = r + t_out * M_out[i]
            print(f"   {r:6.1f} {sag[i]:10.6f} {N_out[i]:10.6f} {t_out:12.6f} {y_out_theory:12.6f} {y_out[i]:12.6f}")
    
    # ========== OPD 分析 ==========
    print(f"\n{'='*70}")
    print("OPD 分析")
    print(f"{'='*70}")
    
    # optiland 计算的 OPD = 入射路径 + 出射路径
    # 入射路径 = 从 z=0 到镜面的距离
    # 出射路径 = 从镜面到 z=0 的距离
    
    # 相对 OPD
    relative_opd = opd_out - opd_out[0]
    relative_opd_waves = relative_opd / wavelength_mm
    
    print(f"\n光线追迹 OPD（相对于主光线）:")
    print(f"   {'r':>6} {'OPD(mm)':>12} {'OPD(waves)':>14}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {relative_opd[i]:12.6f} {relative_opd_waves[i]:14.4f}")
    
    # Pilot Beam OPD
    # 出射波前曲率半径 R_out = -f = -100 mm
    R_out = -f
    pilot_opd = y_out**2 / (2 * R_out)
    pilot_opd_rel = pilot_opd - pilot_opd[0]
    pilot_opd_waves = pilot_opd_rel / wavelength_mm
    
    print(f"\nPilot Beam OPD（使用 y_out，R_out = {R_out} mm）:")
    print(f"   {'r':>6} {'y_out':>10} {'Pilot OPD(mm)':>14} {'Pilot OPD(waves)':>16}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {y_out[i]:10.4f} {pilot_opd_rel[i]:14.6f} {pilot_opd_waves[i]:16.4f}")
    
    # 残差
    residual = relative_opd_waves + pilot_opd_waves
    
    print(f"\n残差 OPD = 光线追迹 OPD + Pilot Beam OPD:")
    print(f"   {'r':>6} {'追迹OPD':>14} {'Pilot OPD':>14} {'残差':>14}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {relative_opd_waves[i]:14.4f} {pilot_opd_waves[i]:14.4f} {residual[i]:14.4f}")
    
    # ========== 问题分析 ==========
    print(f"\n{'='*70}")
    print("问题分析")
    print(f"{'='*70}")
    
    print(f"""
发现：
1. 抛物面镜的残差 OPD 不为零
2. 残差随 r 增加而增加

原因分析：
1. Pilot Beam 假设出射波前是以焦点为中心的球面波
2. 但出射面在 z=0，不是焦点
3. 在 z=0 平面上，波前不是理想球面

关键问题：
- 出射波前的曲率半径 R_out = -f 是在焦点附近的近似
- 在 z=0 平面上，波前的实际形状与理想球面有偏差
- 这个偏差就是残差 OPD

解决方案：
1. 接受残差作为实际像差
2. 或者修改 Pilot Beam 模型，考虑出射面位置
""")
    
    # ========== 验证：出射波前的实际形状 ==========
    print(f"\n{'='*70}")
    print("验证：出射波前的实际形状")
    print(f"{'='*70}")
    
    # 对于抛物面镜，反射后的光线都指向焦点
    # 在 z=0 平面上，波前的相位应该是：
    # φ(y) = k × (从焦点到 (0, y, 0) 的距离)
    #      = k × sqrt(y² + f²)
    
    # 相对于主光线（y=0）：
    # Δφ(y) = k × (sqrt(y² + f²) - f)
    
    # 对应的 OPD：
    # OPD(y) = sqrt(y² + f²) - f
    
    actual_wavefront_opd = np.sqrt(y_out**2 + f**2) - f
    actual_wavefront_opd_rel = actual_wavefront_opd - actual_wavefront_opd[0]
    actual_wavefront_opd_waves = actual_wavefront_opd_rel / wavelength_mm
    
    print(f"\n实际波前 OPD（基于到焦点的距离）:")
    print(f"   OPD(y) = sqrt(y² + f²) - f")
    print(f"\n   {'r':>6} {'y_out':>10} {'实际OPD(mm)':>14} {'实际OPD(waves)':>16}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {y_out[i]:10.4f} {actual_wavefront_opd_rel[i]:14.6f} {actual_wavefront_opd_waves[i]:16.4f}")
    
    # 比较实际波前 OPD 与 Pilot Beam OPD
    print(f"\n比较实际波前 OPD 与 Pilot Beam OPD:")
    print(f"   {'r':>6} {'实际OPD':>14} {'Pilot OPD':>14} {'差异':>14}")
    for i, r in enumerate(r_values):
        diff = actual_wavefront_opd_waves[i] - pilot_opd_waves[i]
        print(f"   {r:6.1f} {actual_wavefront_opd_waves[i]:14.4f} {pilot_opd_waves[i]:14.4f} {diff:14.4f}")
    
    # 比较光线追迹 OPD 与实际波前 OPD
    print(f"\n比较光线追迹 OPD 与实际波前 OPD（取负）:")
    print(f"   {'r':>6} {'追迹OPD':>14} {'-实际OPD':>14} {'差异':>14}")
    for i, r in enumerate(r_values):
        neg_actual = -actual_wavefront_opd_waves[i]
        diff = relative_opd_waves[i] - neg_actual
        print(f"   {r:6.1f} {relative_opd_waves[i]:14.4f} {neg_actual:14.4f} {diff:14.4f}")
    
    # ========== 使用实际波前 OPD 计算残差 ==========
    print(f"\n{'='*70}")
    print("使用实际波前 OPD 计算残差")
    print(f"{'='*70}")
    
    # 残差 = 光线追迹 OPD + 实际波前 OPD
    residual_actual = relative_opd_waves + actual_wavefront_opd_waves
    
    print(f"\n残差 OPD = 光线追迹 OPD + 实际波前 OPD:")
    print(f"   {'r':>6} {'追迹OPD':>14} {'实际波前OPD':>14} {'残差':>14}")
    for i, r in enumerate(r_values):
        print(f"   {r:6.1f} {relative_opd_waves[i]:14.4f} {actual_wavefront_opd_waves[i]:14.4f} {residual_actual[i]:14.4f}")
    
    print(f"\n残差 RMS（使用实际波前 OPD）: {np.sqrt(np.mean(residual_actual**2))*1000:.4f} milli-waves")
    print(f"残差 RMS（使用 Pilot Beam OPD）: {np.sqrt(np.mean(residual**2))*1000:.4f} milli-waves")
    
    # ========== 结论 ==========
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    
    print(f"""
关键发现：
1. Pilot Beam OPD 使用 r²/(2R) 近似
2. 实际波前 OPD 是 sqrt(r² + f²) - f
3. 两者的差异导致残差不为零

泰勒展开：
sqrt(r² + f²) - f = f × (sqrt(1 + (r/f)²) - 1)
                  ≈ f × ((r/f)²/2 - (r/f)⁴/8 + ...)
                  = r²/(2f) - r⁴/(8f³) + ...
                  = r²/(2f) × (1 - r²/(4f²) + ...)

而 Pilot Beam OPD = r²/(2R_out) = r²/(2×(-f)) = -r²/(2f)

所以：
实际波前 OPD ≈ -r²/(2f) × (1 - r²/(4f²))
Pilot Beam OPD = -r²/(2f)

差异 ≈ r⁴/(8f³)

这就是残差的来源！
""")


if __name__ == "__main__":
    analyze_parabola_opd()
