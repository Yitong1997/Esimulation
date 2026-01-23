"""
深入调试 ElementRaytracer 的 OPD 计算问题

关键发现：
- 5° 倾斜时，出射 OPD 范围为 [-7.28, 7.28] mm，而理论值应接近 0
- 这个 OPD 值与出射 z 坐标完全相同！
- 说明 OPD 计算错误地包含了 z 坐标偏移

问题分析：
- 出射面应该垂直于出射光轴
- 出射光线位置应该在出射面上（z=0 在出射面坐标系中）
- 但实际输出的 z 坐标不为 0，说明坐标变换有问题
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import warnings
warnings.filterwarnings('ignore')


def analyze_coordinate_transform(tilt_deg: float):
    """分析坐标变换问题"""
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    
    print(f"\n{'='*70}")
    print(f"分析坐标变换: {tilt_deg}°")
    print(f"{'='*70}")
    
    # 参数
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 256
    physical_size_mm = 40.0
    z_mm = 50.0
    
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    w_z = w0_mm * np.sqrt(1 + (z_mm / z_R)**2)
    amplitude = np.exp(-r_sq / w_z**2)
    
    R = z_mm * (1 + (z_R / z_mm)**2) if z_mm != 0 else np.inf
    k = 2 * np.pi / wavelength_mm
    if np.isinf(R):
        phase = np.zeros_like(r_sq)
    else:
        phase = k * r_sq / (2 * R)
    
    # 采样光线
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size_mm,
        wavelength=wavelength_um,
        num_rays=50,  # 减少光线数量便于分析
        distribution="hexapolar",
    )
    
    input_rays = sampler.get_output_rays()
    
    x_in = np.asarray(input_rays.x)
    y_in = np.asarray(input_rays.y)
    z_in = np.asarray(input_rays.z)
    
    # 表面定义
    tilt_rad = np.radians(tilt_deg)
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=30.0,
        conic=0.0,
        tilt_x=tilt_rad,
        tilt_y=0.0,
    )
    
    # 计算出射方向
    d_in = np.array([0, 0, 1])
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    
    print(f"\n入射方向: {d_in}")
    print(f"表面法向量: {n}")
    print(f"出射方向: {d_out}")
    
    # 光线追迹
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=tuple(d_out),
    )
    
    output_rays = raytracer.trace(input_rays)
    
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    opd_out = np.asarray(output_rays.opd)
    
    print(f"\n--- 出射光线分析 ---")
    print(f"出射 z 范围: [{np.min(z_out):.6f}, {np.max(z_out):.6f}] mm")
    print(f"出射 OPD 范围: [{np.min(opd_out):.6f}, {np.max(opd_out):.6f}] mm")
    
    # 关键检查：z_out 和 opd_out 的关系
    print(f"\n--- z_out 与 opd_out 的关系 ---")
    correlation = np.corrcoef(z_out, opd_out)[0, 1]
    print(f"相关系数: {correlation:.6f}")
    
    if abs(correlation) > 0.99:
        print("⚠️ z_out 和 opd_out 高度相关！")
        print("   这说明 OPD 错误地包含了 z 坐标偏移")
    
    # 分析几个典型光线
    print(f"\n--- 典型光线分析 ---")
    print(f"{'光线':>4} | {'x_in':>8} | {'y_in':>8} | {'x_out':>8} | {'y_out':>8} | {'z_out':>10} | {'OPD':>10}")
    print("-" * 80)
    
    # 选择几条典型光线
    indices = [0, len(x_in)//4, len(x_in)//2, 3*len(x_in)//4, len(x_in)-1]
    for i in indices:
        print(f"{i:>4} | {x_in[i]:>8.3f} | {y_in[i]:>8.3f} | "
              f"{x_out[i]:>8.3f} | {y_out[i]:>8.3f} | {z_out[i]:>10.4f} | {opd_out[i]:>10.4f}")
    
    # 理论分析
    print(f"\n--- 理论分析 ---")
    print(f"对于平面镜反射，出射面应该垂直于出射方向 {d_out}")
    print(f"出射面上的点应该满足: (r - r0) · d_out = 0")
    print(f"其中 r0 是出射面原点（主光线与镜面交点）")
    
    # 检查出射光线是否在出射面上
    # 出射面方程：d_out · (r - r0) = 0
    # 假设 r0 = (0, 0, 0)（主光线在原点反射）
    # 则出射面方程：d_out[0]*x + d_out[1]*y + d_out[2]*z = 0
    
    # 计算每条光线到出射面的距离
    distance_to_exit_plane = d_out[0]*x_out + d_out[1]*y_out + d_out[2]*z_out
    
    print(f"\n光线到出射面的距离:")
    print(f"  范围: [{np.min(distance_to_exit_plane):.6f}, {np.max(distance_to_exit_plane):.6f}] mm")
    print(f"  RMS: {np.std(distance_to_exit_plane):.6f} mm")
    
    if np.std(distance_to_exit_plane) > 0.001:
        print("⚠️ 出射光线不在出射面上！")
        print("   这说明坐标变换有问题")
    
    # 检查出射方向
    print(f"\n出射光线方向:")
    print(f"  L 范围: [{np.min(L_out):.6f}, {np.max(L_out):.6f}]")
    print(f"  M 范围: [{np.min(M_out):.6f}, {np.max(M_out):.6f}]")
    print(f"  N 范围: [{np.min(N_out):.6f}, {np.max(N_out):.6f}]")
    print(f"  期望方向: ({d_out[0]:.6f}, {d_out[1]:.6f}, {d_out[2]:.6f})")
    
    # 方向偏差
    dL = L_out - d_out[0]
    dM = M_out - d_out[1]
    dN = N_out - d_out[2]
    
    print(f"\n方向偏差:")
    print(f"  ΔL 范围: [{np.min(dL):.6f}, {np.max(dL):.6f}]")
    print(f"  ΔM 范围: [{np.min(dM):.6f}, {np.max(dM):.6f}]")
    print(f"  ΔN 范围: [{np.min(dN):.6f}, {np.max(dN):.6f}]")
    
    return {
        'angle_deg': tilt_deg,
        'z_out': z_out,
        'opd_out': opd_out,
        'correlation': correlation,
        'distance_to_exit_plane': distance_to_exit_plane,
    }


def main():
    print("=" * 70)
    print("深入调试 ElementRaytracer 的 OPD 计算问题")
    print("=" * 70)
    
    # 分析几个角度
    results = []
    for angle in [0, 5, 22.5, 45]:
        try:
            r = analyze_coordinate_transform(angle)
            results.append(r)
        except Exception as e:
            print(f"\n{angle}° 分析失败: {e}")
    
    # 总结
    print("\n" + "=" * 70)
    print("问题总结")
    print("=" * 70)
    
    print("""
关键发现：

1. 出射光线的 z 坐标不为 0
   - 0° 和 45° 时，z_out ≈ 0（正确）
   - 5° 时，z_out 范围为 [-7.28, 7.28] mm（错误）
   
2. OPD 与 z_out 高度相关
   - 这说明 OPD 计算错误地包含了 z 坐标偏移
   - OPD 应该只反映光程差，不应该包含坐标偏移

3. 出射光线方向不正确
   - 出射光线方向应该是反射后的方向
   - 但实际输出的方向余弦 (L, M, N) 仍然是 (0, 0, 1)
   - 这说明 ElementRaytracer 没有正确处理出射方向

4. 问题根源推测：
   - ElementRaytracer 可能没有将出射光线变换到出射面坐标系
   - 或者 OPD 计算没有考虑坐标系变换
   - 0° 和 45° 是特殊情况，入射面和出射面重合或正交，
     所以坐标变换问题不明显

5. 为什么 0° 和 45° 正确：
   - 0°：入射面和出射面重合，无需坐标变换
   - 45°：入射面和出射面正交，z_out 自然为 0
   - 其他角度：入射面和出射面成任意角度，需要正确的坐标变换
""")


if __name__ == "__main__":
    main()
