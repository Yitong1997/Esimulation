"""
调试 OPD 符号约定问题

分析光线追迹 OPD 和 Pilot Beam OPD 的物理含义差异。

关键问题：
- 光线追迹 OPD：实际几何光程增加量（总是正的，因为边缘光线走更长路径）
- Pilot Beam OPD：相位延迟对应的等效光程（可正可负，取决于曲率半径符号）

对于凹面镜反射：
- 入射平面波 → 出射会聚波
- 光线追迹 OPD = 2 × sag = r²/R_mirror（正值）
- Pilot Beam OPD = r²/(2R_out)，其中 R_out = -R_mirror/2（负值）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import PilotBeamParams


def analyze_opd_convention():
    """分析 OPD 符号约定"""
    
    print("=" * 70)
    print("OPD 符号约定分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    R_mirror = 200.0  # 凹面镜曲率半径
    r = 10.0  # 边缘位置
    
    wavelength_mm = wavelength_um * 1e-3
    
    print(f"\n参数:")
    print(f"  波长: {wavelength_um} um")
    print(f"  镜面曲率半径: {R_mirror} mm")
    print(f"  边缘位置 r: {r} mm")
    
    # 1. 镜面 sag
    sag = r**2 / (2 * R_mirror)
    print(f"\n1. 镜面 sag:")
    print(f"   sag = r²/(2R) = {r}²/(2×{R_mirror}) = {sag:.6f} mm")
    
    # 2. 反射 OPD（几何光程增加）
    # 边缘光线比主光线多走 2×sag 的距离
    raytracing_opd_mm = 2 * sag
    raytracing_opd_waves = raytracing_opd_mm / wavelength_mm
    print(f"\n2. 光线追迹 OPD（几何光程增加）:")
    print(f"   OPD = 2 × sag = {raytracing_opd_mm:.6f} mm = {raytracing_opd_waves:.2f} waves")
    print(f"   物理含义：边缘光线比主光线多走的距离")
    
    # 3. 出射 Pilot Beam 参数
    # 入射平面波（R_in = inf）经过凹面镜（f = R/2）反射后
    # 出射曲率半径 R_out = -f = -R/2
    R_out = -R_mirror / 2
    print(f"\n3. 出射 Pilot Beam:")
    print(f"   入射曲率半径: inf（平面波）")
    print(f"   镜面焦距: f = R/2 = {R_mirror/2} mm")
    print(f"   出射曲率半径: R_out = -f = {R_out} mm（会聚波）")
    
    # 4. Pilot Beam OPD（相位延迟对应的等效光程）
    pilot_opd_mm = r**2 / (2 * R_out)
    pilot_opd_waves = pilot_opd_mm / wavelength_mm
    print(f"\n4. Pilot Beam OPD（相位延迟等效光程）:")
    print(f"   OPD = r²/(2R_out) = {r}²/(2×{R_out}) = {pilot_opd_mm:.6f} mm = {pilot_opd_waves:.2f} waves")
    print(f"   物理含义：相位延迟 φ = k × OPD")
    print(f"   负值表示相位超前（会聚波边缘相位超前于主光线）")
    
    # 5. 残差分析
    residual_mm = raytracing_opd_mm - pilot_opd_mm
    residual_waves = residual_mm / wavelength_mm
    print(f"\n5. 残差分析:")
    print(f"   残差 = 光线追迹 OPD - Pilot Beam OPD")
    print(f"        = {raytracing_opd_mm:.6f} - ({pilot_opd_mm:.6f})")
    print(f"        = {residual_mm:.6f} mm = {residual_waves:.2f} waves")
    print(f"   [问题] 残差非常大！")
    
    # 6. 正确的理解
    print(f"\n" + "=" * 70)
    print("正确的理解")
    print("=" * 70)
    
    print("""
问题分析：

光线追迹 OPD 和 Pilot Beam OPD 描述的是不同的物理量：

1. 光线追迹 OPD：
   - 定义：边缘光线相对于主光线的几何光程增加量
   - 对于凹面镜：OPD = 2 × sag = r²/R_mirror（总是正的）
   - 物理含义：边缘光线走更长的路径

2. Pilot Beam OPD：
   - 定义：相位延迟对应的等效光程
   - 公式：OPD = r²/(2R)，其中 R 是波前曲率半径
   - 物理含义：φ = k × OPD，正 OPD 表示相位滞后

3. 关键区别：
   - 光线追迹 OPD 使用镜面曲率半径 R_mirror
   - Pilot Beam OPD 使用波前曲率半径 R_out = -R_mirror/2
   - 两者差了一个因子 -4！

4. 正确的残差计算：
   - 对于理想球面镜，出射波前应该是完美的球面波
   - 光线追迹 OPD 应该等于 Pilot Beam 相位对应的光程
   - 但符号约定不同！
""")
    
    # 7. 验证：相位和 OPD 的关系
    print(f"\n" + "=" * 70)
    print("验证：相位和 OPD 的关系")
    print("=" * 70)
    
    k = 2 * np.pi / wavelength_mm
    
    # 出射波前相位（相对于主光线）
    # 会聚波：边缘相位超前于主光线
    # φ = k × r²/(2R_out) = k × r²/(2×(-100)) < 0
    phase_pilot = k * r**2 / (2 * R_out)
    print(f"\n出射波前相位（Pilot Beam）:")
    print(f"  φ = k × r²/(2R_out) = {phase_pilot:.2f} rad = {phase_pilot/(2*np.pi):.2f} waves")
    print(f"  负值表示相位超前")
    
    # 光线追迹的相位
    # 正 OPD 对应正相位（相位滞后）
    phase_raytracing = k * raytracing_opd_mm
    print(f"\n光线追迹相位:")
    print(f"  φ = k × OPD = {phase_raytracing:.2f} rad = {phase_raytracing/(2*np.pi):.2f} waves")
    print(f"  正值表示相位滞后")
    
    print(f"\n相位差:")
    print(f"  Δφ = {phase_raytracing - phase_pilot:.2f} rad = {(phase_raytracing - phase_pilot)/(2*np.pi):.2f} waves")
    
    # 8. 结论
    print(f"\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    print("""
根本问题：光线追迹 OPD 和 Pilot Beam OPD 的物理含义不同！

光线追迹 OPD：
- 描述的是几何光程增加量
- 对于反射镜：OPD = 2 × sag（边缘光线多走的距离）
- 总是正的（边缘光线走更长路径）

Pilot Beam OPD：
- 描述的是相位延迟对应的等效光程
- 公式：OPD = r²/(2R)，R 是波前曲率半径
- 可正可负（取决于波前是发散还是会聚）

正确的处理方式：

方案 A：修改 Pilot Beam OPD 计算
- 使用光线追迹的 OPD 定义
- Pilot Beam OPD = 2 × sag_equivalent
- 对于会聚波，sag_equivalent = r²/(2|R|)，OPD 为正

方案 B：修改残差计算
- 残差 = 光线追迹 OPD - |Pilot Beam OPD|
- 或者：残差 = 光线追迹 OPD + Pilot Beam OPD（当 R < 0 时）

方案 C：统一符号约定
- 定义：正 OPD = 相位滞后 = 光程增加
- 对于会聚波（R < 0），边缘相位超前，OPD 应该为负
- 但光线追迹的 OPD 是正的...

需要仔细考虑符号约定！
""")


def verify_with_actual_raytracing():
    """使用实际光线追迹验证"""
    
    print("\n" + "=" * 70)
    print("使用实际光线追迹验证")
    print("=" * 70)
    
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 64
    physical_size_mm = 20.0
    num_rays = 50
    R_mirror = 200.0
    
    wavelength_mm = wavelength_um * 1e-3
    
    # 创建入射平面波
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    amplitude = np.exp(-r_sq / w0_mm**2)
    phase = np.zeros_like(r_sq)  # 平面波
    
    # 光线采样
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size_mm,
        wavelength=wavelength_um,
        num_rays=num_rays,
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    
    # 光线追迹
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R_mirror,
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
    
    output_rays = raytracer.trace(input_rays)
    
    # 获取结果
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    # 选择一条边缘光线进行分析
    r_out = np.sqrt(x_out**2 + y_out**2)
    edge_idx = np.argmax(r_out[valid_mask])
    edge_indices = np.where(valid_mask)[0]
    idx = edge_indices[edge_idx]
    
    r_edge = r_out[idx]
    opd_edge = opd_waves[idx]
    
    print(f"\n边缘光线分析:")
    print(f"  位置 r: {r_edge:.4f} mm")
    print(f"  光线追迹 OPD: {opd_edge:.2f} waves")
    
    # 理论计算
    sag = r_edge**2 / (2 * R_mirror)
    theoretical_opd_mm = 2 * sag
    theoretical_opd_waves = theoretical_opd_mm / wavelength_mm
    
    print(f"\n理论计算:")
    print(f"  sag = r²/(2R) = {sag:.6f} mm")
    print(f"  理论 OPD = 2×sag = {theoretical_opd_mm:.6f} mm = {theoretical_opd_waves:.2f} waves")
    
    # Pilot Beam OPD
    R_out = -R_mirror / 2
    pilot_opd_mm = r_edge**2 / (2 * R_out)
    pilot_opd_waves = pilot_opd_mm / wavelength_mm
    
    print(f"\nPilot Beam OPD:")
    print(f"  R_out = {R_out} mm")
    print(f"  OPD = r²/(2R_out) = {pilot_opd_mm:.6f} mm = {pilot_opd_waves:.2f} waves")
    
    # 比较
    print(f"\n比较:")
    print(f"  光线追迹 OPD: {opd_edge:.2f} waves")
    print(f"  理论 OPD (2×sag): {theoretical_opd_waves:.2f} waves")
    print(f"  Pilot Beam OPD: {pilot_opd_waves:.2f} waves")
    print(f"  差异 (追迹 - 理论): {opd_edge - theoretical_opd_waves:.4f} waves")
    print(f"  差异 (追迹 - Pilot): {opd_edge - pilot_opd_waves:.2f} waves")


if __name__ == "__main__":
    analyze_opd_convention()
    verify_with_actual_raytracing()
