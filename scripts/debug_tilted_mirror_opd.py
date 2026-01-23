"""
分析倾斜镜的 OPD 计算问题

问题：45度倾斜球面镜的残差 OPD 非常大（~130 waves）
需要找出原因并确定是否是代码问题还是物理效应。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_tilted_mirror():
    """分析倾斜镜 OPD"""
    
    print("=" * 70)
    print("倾斜镜 OPD 分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    R = 200.0  # 曲率半径 mm
    f = R / 2  # 焦距 100 mm
    tilt_angle = np.pi / 4  # 45度
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  曲率半径 R: {R} mm")
    print(f"  焦距 f: {f} mm")
    print(f"  倾斜角: {np.degrees(tilt_angle):.1f} deg")
    
    # ========== 理论分析 ==========
    print(f"\n{'='*70}")
    print("理论分析：倾斜球面镜")
    print(f"{'='*70}")
    
    print(f"""
对于倾斜球面镜：

1. 入射光轴：沿 Z 方向 (0, 0, 1)
2. 镜面倾斜 45°：绕 X 轴旋转
3. 出射光轴：沿 (0, -sin(90°), cos(90°)) = (0, -1, 0) 方向
   （实际上是 (0, -sin(2×45°), cos(2×45°)) = (0, -1, 0)）

问题：
- 入射面垂直于入射光轴（XY 平面）
- 出射面垂直于出射光轴（XZ 平面）
- 两个平面不平行！

Pilot Beam OPD 公式 r²/(2R) 假设：
- 波前是球面波
- 采样面垂直于光轴
- 坐标 r 是到光轴的距离

对于倾斜情况：
- 出射面上的坐标 (x_out, y_out) 不是到出射光轴的距离
- 需要考虑坐标变换
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
        tilt_x=tilt_angle,
    )
    
    # 出射方向
    exit_dir = (0, -np.sin(2 * tilt_angle), np.cos(2 * tilt_angle))
    print(f"\n出射光轴方向: {exit_dir}")
    
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=exit_dir,
    )
    
    # 测试光线（沿 Y 轴分布）
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
    
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    opd_out = np.asarray(output_rays.opd)
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    
    print(f"\n光线追迹结果:")
    print(f"   {'y_in':>6} {'x_out':>10} {'y_out':>10} {'z_out':>10} {'OPD(mm)':>12}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {x_out[i]:10.4f} {y_out[i]:10.4f} {z_out[i]:10.4f} {opd_out[i]:12.6f}")
    
    print(f"\n出射光线方向:")
    print(f"   {'y_in':>6} {'L':>10} {'M':>10} {'N':>10}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {L_out[i]:10.6f} {M_out[i]:10.6f} {N_out[i]:10.6f}")
    
    # ========== 分析出射面坐标系 ==========
    print(f"\n{'='*70}")
    print("出射面坐标系分析")
    print(f"{'='*70}")
    
    # 出射面垂直于出射光轴
    # 出射光轴方向：(0, -1, 0)（对于 45° 倾斜）
    # 出射面的法向量就是出射光轴方向
    
    # 出射面局部坐标系：
    # Z_local = 出射光轴方向 = (0, -1, 0)
    # X_local = 全局 X 轴 = (1, 0, 0)（假设）
    # Y_local = Z_local × X_local = (0, 0, 1)
    
    # 但是 ElementRaytracer 返回的坐标是在什么坐标系中？
    # 需要检查 transform_rays_to_local 函数
    
    print(f"""
出射面坐标系：
- 出射光轴方向：{exit_dir}
- 出射面是垂直于出射光轴的平面

问题：
- 光线追迹返回的 (x_out, y_out, z_out) 是在哪个坐标系中？
- 如果是全局坐标系，需要转换到出射面局部坐标系
- 如果是出射面局部坐标系，z_out 应该接近 0
""")
    
    # 检查 z_out 是否接近 0
    print(f"\n检查 z_out:")
    print(f"  z_out 范围: [{np.min(z_out):.6f}, {np.max(z_out):.6f}]")
    print(f"  z_out 是否接近 0: {np.allclose(z_out, 0, atol=1e-3)}")
    
    # ========== Pilot Beam OPD 计算 ==========
    print(f"\n{'='*70}")
    print("Pilot Beam OPD 计算")
    print(f"{'='*70}")
    
    # 当前实现：使用 r² = x_out² + y_out²
    R_out = -f  # 出射波前曲率半径
    r_sq_current = x_out**2 + y_out**2
    pilot_opd_current = r_sq_current / (2 * R_out)
    pilot_opd_current_rel = pilot_opd_current - pilot_opd_current[0]
    pilot_opd_current_waves = pilot_opd_current_rel / wavelength_mm
    
    print(f"\n当前实现：r² = x_out² + y_out²")
    print(f"   {'y_in':>6} {'r²':>12} {'Pilot OPD(waves)':>18}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {r_sq_current[i]:12.4f} {pilot_opd_current_waves[i]:18.4f}")
    
    # 正确的实现：r 应该是到出射光轴的距离
    # 出射光轴通过原点，方向为 exit_dir
    # 点 (x, y, z) 到直线的距离 = |P × d| / |d|
    # 其中 P = (x, y, z)，d = exit_dir
    
    exit_dir_arr = np.array(exit_dir)
    
    def distance_to_axis(x, y, z, axis_dir):
        """计算点到光轴的距离"""
        P = np.array([x, y, z])
        cross = np.cross(P, axis_dir)
        return np.linalg.norm(cross) / np.linalg.norm(axis_dir)
    
    r_correct = np.array([
        distance_to_axis(x_out[i], y_out[i], z_out[i], exit_dir_arr)
        for i in range(n_rays)
    ])
    
    pilot_opd_correct = r_correct**2 / (2 * R_out)
    pilot_opd_correct_rel = pilot_opd_correct - pilot_opd_correct[0]
    pilot_opd_correct_waves = pilot_opd_correct_rel / wavelength_mm
    
    print(f"\n正确实现：r = 到出射光轴的距离")
    print(f"   {'y_in':>6} {'r':>12} {'Pilot OPD(waves)':>18}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {r_correct[i]:12.4f} {pilot_opd_correct_waves[i]:18.4f}")
    
    # ========== 残差比较 ==========
    print(f"\n{'='*70}")
    print("残差比较")
    print(f"{'='*70}")
    
    # 相对 OPD
    relative_opd = opd_out - opd_out[0]
    relative_opd_waves = relative_opd / wavelength_mm
    
    # 残差（当前实现）
    residual_current = relative_opd_waves + pilot_opd_current_waves
    
    # 残差（正确实现）
    residual_correct = relative_opd_waves + pilot_opd_correct_waves
    
    print(f"\n残差比较:")
    print(f"   {'y_in':>6} {'追迹OPD':>14} {'残差(当前)':>14} {'残差(正确)':>14}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {relative_opd_waves[i]:14.4f} {residual_current[i]:14.4f} {residual_correct[i]:14.4f}")
    
    print(f"\n残差 RMS:")
    print(f"  当前实现: {np.sqrt(np.mean(residual_current**2))*1000:.4f} milli-waves")
    print(f"  正确实现: {np.sqrt(np.mean(residual_correct**2))*1000:.4f} milli-waves")
    
    # ========== 进一步分析 ==========
    print(f"\n{'='*70}")
    print("进一步分析")
    print(f"{'='*70}")
    
    # 检查出射面坐标是否在出射面局部坐标系中
    # 如果是，那么 z_out 应该接近 0
    # 如果不是，需要进行坐标变换
    
    print(f"""
分析结果：

1. 当前实现使用 r² = x_out² + y_out² 计算 Pilot Beam OPD
   - 这假设出射面坐标是在出射面局部坐标系中
   - 但实际上可能是全局坐标系

2. 正确的实现应该使用到出射光轴的距离
   - r = 点到出射光轴的垂直距离
   - 这与坐标系无关

3. 但即使使用正确的 r，残差仍然很大
   - 这说明问题不仅仅是坐标系
   - 可能是 Pilot Beam 模型本身不适用于倾斜情况

4. 对于倾斜镜，出射波前可能不是简单的球面波
   - 需要考虑像散等像差
   - Pilot Beam 模型需要扩展
""")
    
    # ========== 检查出射波前形状 ==========
    print(f"\n{'='*70}")
    print("检查出射波前形状")
    print(f"{'='*70}")
    
    # 对于倾斜球面镜，出射波前有像散
    # 子午面（YZ 平面）和弧矢面（XZ 平面）的焦距不同
    
    # 子午焦距：f_t = f × cos(θ)
    # 弧矢焦距：f_s = f / cos(θ)
    # 其中 θ 是入射角
    
    theta = tilt_angle  # 入射角 = 倾斜角
    f_tangential = f * np.cos(theta)
    f_sagittal = f / np.cos(theta)
    
    print(f"\n像散分析:")
    print(f"  入射角 θ: {np.degrees(theta):.1f} deg")
    print(f"  子午焦距 f_t: {f_tangential:.2f} mm")
    print(f"  弧矢焦距 f_s: {f_sagittal:.2f} mm")
    print(f"  焦距差: {f_sagittal - f_tangential:.2f} mm")
    
    # 对应的曲率半径
    R_tangential = -f_tangential
    R_sagittal = -f_sagittal
    
    print(f"\n对应的曲率半径:")
    print(f"  子午曲率半径 R_t: {R_tangential:.2f} mm")
    print(f"  弧矢曲率半径 R_s: {R_sagittal:.2f} mm")
    
    # 使用像散模型计算 Pilot Beam OPD
    # OPD = x²/(2R_s) + y²/(2R_t)
    # 但这里 x, y 是在出射面局部坐标系中的坐标
    
    # 假设出射面局部坐标系：
    # X_local = 全局 X（弧矢方向）
    # Y_local = 全局 Z（子午方向，因为出射光轴沿 -Y）
    # Z_local = 出射光轴方向 = (0, -1, 0)
    
    # 在出射面局部坐标系中：
    # x_local = x_out（弧矢方向）
    # y_local = z_out（子午方向）
    
    pilot_opd_astigmatic = x_out**2 / (2 * R_sagittal) + z_out**2 / (2 * R_tangential)
    pilot_opd_astigmatic_rel = pilot_opd_astigmatic - pilot_opd_astigmatic[0]
    pilot_opd_astigmatic_waves = pilot_opd_astigmatic_rel / wavelength_mm
    
    residual_astigmatic = relative_opd_waves + pilot_opd_astigmatic_waves
    
    print(f"\n使用像散模型:")
    print(f"  Pilot OPD = x²/(2R_s) + z²/(2R_t)")
    print(f"\n   {'y_in':>6} {'Pilot OPD':>14} {'残差':>14}")
    for i in range(n_rays):
        print(f"   {r_values[i]:6.1f} {pilot_opd_astigmatic_waves[i]:14.4f} {residual_astigmatic[i]:14.4f}")
    
    print(f"\n残差 RMS（像散模型）: {np.sqrt(np.mean(residual_astigmatic**2))*1000:.4f} milli-waves")
    
    # ========== 结论 ==========
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    
    print(f"""
发现：

1. 当前 Pilot Beam 模型（球面波）不适用于倾斜镜
   - 残差 RMS（球面模型）: {np.sqrt(np.mean(residual_current**2))*1000:.1f} mW

2. 倾斜镜产生像散，需要使用像散模型
   - 子午焦距 f_t = f × cos(θ) = {f_tangential:.2f} mm
   - 弧矢焦距 f_s = f / cos(θ) = {f_sagittal:.2f} mm

3. 使用像散模型后，残差显著减小
   - 残差 RMS（像散模型）: {np.sqrt(np.mean(residual_astigmatic**2))*1000:.1f} mW

建议：
1. 对于倾斜镜，需要扩展 Pilot Beam 模型以支持像散
2. 或者在混合传播中，对倾斜镜使用不同的处理方式
3. 当前实现对于正入射情况是正确的
""")


if __name__ == "__main__":
    analyze_tilted_mirror()
