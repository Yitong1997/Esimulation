"""
分析出射面坐标与 Pilot Beam OPD 计算的关系

问题：验证脚本中使用出射光线的 (x_out, y_out) 计算 Pilot Beam OPD
但这些坐标是在出射面局部坐标系中的，而不是入射面坐标系中的。

对于球面镜：
- 入射光线位置 (x_in, y_in)
- 出射光线位置 (x_out, y_out) ≠ (x_in, y_in)
- 因为球面镜会改变光线位置（聚焦效应）

Pilot Beam OPD 应该使用哪个坐标？
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_exit_plane_coordinates():
    """分析出射面坐标"""
    
    print("=" * 70)
    print("出射面坐标与 Pilot Beam OPD 分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    R = 200.0  # 曲率半径 mm
    f = R / 2  # 焦距 100 mm
    R_out = -f  # 出射波前曲率半径 -100 mm
    
    # 测试光线位置
    r_values = np.array([0, 2, 4, 6, 8, 10])  # mm
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  曲率半径 R: {R} mm")
    print(f"  出射波前曲率半径 R_out: {R_out} mm")
    
    # ========== 光线追迹 ==========
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    
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
    opd_out = np.asarray(output_rays.opd)
    
    # 相对 OPD
    relative_opd = opd_out - opd_out[0]
    relative_opd_waves = relative_opd / wavelength_mm
    
    print(f"\n光线追迹结果:")
    print(f"   {'y_in':>6} {'y_out':>10} {'相对OPD(waves)':>16}")
    for i in range(n_rays):
        print(f"   {y_in[i]:6.1f} {y_out[i]:10.4f} {relative_opd_waves[i]:16.4f}")
    
    # ========== Pilot Beam OPD 计算方式比较 ==========
    print(f"\n{'='*70}")
    print("Pilot Beam OPD 计算方式比较")
    print(f"{'='*70}")
    
    # 方式 1：使用入射坐标 y_in
    pilot_opd_in = y_in**2 / (2 * R_out)
    pilot_opd_in_rel = pilot_opd_in - pilot_opd_in[0]
    pilot_opd_in_waves = pilot_opd_in_rel / wavelength_mm
    
    # 方式 2：使用出射坐标 y_out
    pilot_opd_out = y_out**2 / (2 * R_out)
    pilot_opd_out_rel = pilot_opd_out - pilot_opd_out[0]
    pilot_opd_out_waves = pilot_opd_out_rel / wavelength_mm
    
    print(f"\n方式 1：使用入射坐标 y_in 计算 Pilot Beam OPD")
    print(f"   Pilot OPD = y_in² / (2 × R_out)")
    print(f"\n方式 2：使用出射坐标 y_out 计算 Pilot Beam OPD")
    print(f"   Pilot OPD = y_out² / (2 × R_out)")
    
    print(f"\n比较:")
    print(f"   {'y_in':>6} {'y_out':>10} {'Pilot(y_in)':>14} {'Pilot(y_out)':>14} {'差异':>14}")
    for i in range(n_rays):
        diff = pilot_opd_out_waves[i] - pilot_opd_in_waves[i]
        print(f"   {y_in[i]:6.1f} {y_out[i]:10.4f} {pilot_opd_in_waves[i]:14.4f} {pilot_opd_out_waves[i]:14.4f} {diff:14.4f}")
    
    # ========== 残差 OPD 比较 ==========
    print(f"\n{'='*70}")
    print("残差 OPD 比较")
    print(f"{'='*70}")
    
    # 残差 = 实际 OPD + Pilot Beam OPD
    residual_in = relative_opd_waves + pilot_opd_in_waves
    residual_out = relative_opd_waves + pilot_opd_out_waves
    
    print(f"\n残差 OPD = 实际相对 OPD + Pilot Beam OPD")
    print(f"\n   {'y_in':>6} {'实际OPD':>12} {'残差(y_in)':>14} {'残差(y_out)':>14}")
    for i in range(n_rays):
        print(f"   {y_in[i]:6.1f} {relative_opd_waves[i]:12.4f} {residual_in[i]:14.4f} {residual_out[i]:14.4f}")
    
    print(f"\n残差 RMS:")
    print(f"   使用 y_in: {np.sqrt(np.mean(residual_in**2))*1000:.4f} milli-waves")
    print(f"   使用 y_out: {np.sqrt(np.mean(residual_out**2))*1000:.4f} milli-waves")
    
    # ========== 物理分析 ==========
    print(f"\n{'='*70}")
    print("物理分析")
    print(f"{'='*70}")
    
    print(f"""
问题分析：

1. 光线追迹计算的 OPD 是从入射面到出射面的几何路径差
   - 入射面：z = 0 平面，光线位置 (0, y_in, 0)
   - 出射面：z = 0 平面，光线位置 (0, y_out, 0)
   - OPD = 入射路径 + 出射路径

2. Pilot Beam OPD 的物理意义
   - Pilot Beam 描述的是理想球面波
   - 球面波的相位 φ = k × r² / (2R)
   - 这里 r 是到光轴的距离

3. 关键问题：r 应该用哪个坐标？
   
   对于入射面：
   - 入射波前是平面波（R_in = ∞）
   - 入射 Pilot Beam OPD = 0
   
   对于出射面：
   - 出射波前是球面波（R_out = -100 mm）
   - 出射 Pilot Beam OPD = r² / (2 × R_out)
   - 这里 r 应该是出射面上的坐标 y_out！

4. 但是，光线追迹的 OPD 是相对于入射位置计算的
   - 入射位置 y_in 的光线，出射位置是 y_out
   - 光线追迹 OPD 对应的是 y_in 处入射的光线
   - Pilot Beam OPD 应该对应 y_out 处的相位

5. 正确的残差计算：
   - 光线追迹 OPD（相对于主光线）：对应 y_in 处入射的光线
   - Pilot Beam OPD：对应 y_out 处的出射波前相位
   - 残差 = 光线追迹 OPD + Pilot Beam OPD(y_out)
""")
    
    # ========== 验证：使用 y_out 计算残差 ==========
    print(f"\n{'='*70}")
    print("验证：使用 y_out 计算残差")
    print(f"{'='*70}")
    
    # 理论分析：
    # 对于球面镜，入射平面波反射后变成球面波
    # 出射波前的相位 φ_out(y_out) = k × y_out² / (2 × R_out)
    # 
    # 光线追迹 OPD 包含：
    # 1. 入射路径：从 z=0 到镜面
    # 2. 出射路径：从镜面到 z=0
    # 
    # 对于理想球面镜，光线追迹 OPD 应该等于 -Pilot Beam OPD(y_out)
    # 因为：
    # - 光线追迹 OPD > 0（边缘光线走更长路径）
    # - Pilot Beam OPD < 0（R_out < 0，会聚波）
    # - 两者大小相等，符号相反
    
    print(f"\n理论验证：")
    print(f"   对于理想球面镜：光线追迹 OPD ≈ -Pilot Beam OPD(y_out)")
    print(f"\n   {'y_in':>6} {'y_out':>10} {'追迹OPD':>12} {'-Pilot(y_out)':>14} {'差异':>12}")
    for i in range(n_rays):
        neg_pilot = -pilot_opd_out_waves[i]
        diff = relative_opd_waves[i] - neg_pilot
        print(f"   {y_in[i]:6.1f} {y_out[i]:10.4f} {relative_opd_waves[i]:12.4f} {neg_pilot:14.4f} {diff:12.4f}")
    
    # ========== 结论 ==========
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    
    print(f"""
发现：
1. 使用 y_out 计算 Pilot Beam OPD 时，残差 RMS = {np.sqrt(np.mean(residual_out**2))*1000:.4f} milli-waves
2. 使用 y_in 计算 Pilot Beam OPD 时，残差 RMS = {np.sqrt(np.mean(residual_in**2))*1000:.4f} milli-waves

问题根源：
- 验证脚本使用 y_out 计算 Pilot Beam OPD
- 但 y_out ≠ y_in（球面镜会改变光线位置）
- 这导致 Pilot Beam OPD 计算不准确

但是，使用 y_out 是正确的！
- 出射波前的相位分布是在出射面上定义的
- 出射面上位置 y_out 处的相位 = k × y_out² / (2 × R_out)
- 光线追迹 OPD 对应的是从 y_in 入射、到达 y_out 的光线

残差不为零的原因：
1. 球面镜的高阶像差（球差）
2. 出射面位置与镜面顶点不重合导致的额外路径差

这是物理上正确的结果，不是计算错误！
""")


if __name__ == "__main__":
    analyze_exit_plane_coordinates()
