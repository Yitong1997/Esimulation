"""
调试脚本：检查伽利略OAP系统中采样面的波前情况

目的：
1. 检查采样面的位置和方向
2. 检查波前是否有倾斜
3. 分析波前的相位分布
4. 检查相位折叠（phase wrapping）问题

作者：混合光学仿真项目
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


def simple_unwrap_phase(phase, mask=None):
    """简单的相位解包裹（沿行和列）
    
    参数:
        phase: 2D 相位数组
        mask: 有效区域掩模
    
    返回:
        解包裹后的相位
    """
    # 使用 numpy 的 unwrap 函数
    unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
    return unwrapped


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze_wavefront_tilt(wavefront, sampling_mm, name):
    """分析波前倾斜
    
    通过拟合波前相位的线性项来估计倾斜。
    
    参数:
        wavefront: 复振幅数组
        sampling_mm: 采样间隔 (mm)
        name: 采样面名称
    """
    n = wavefront.shape[0]
    amplitude = np.abs(wavefront)
    phase = np.angle(wavefront)
    
    # 创建坐标网格 (mm)
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    
    # 创建掩模（只使用振幅较大的区域）
    threshold = 0.1 * np.max(amplitude)
    mask = amplitude > threshold
    
    if np.sum(mask) < 100:
        print(f"  {name}: 有效点太少，无法分析")
        return None
    
    # 尝试相位解包裹
    try:
        phase_unwrapped = simple_unwrap_phase(phase)
        phase_unwrapped = np.where(mask, phase_unwrapped, np.nan)
    except:
        phase_unwrapped = phase
    
    # 提取有效区域的数据
    x_valid = X[mask]
    y_valid = Y[mask]
    phase_valid = phase_unwrapped[mask]
    
    # 检查是否有 NaN
    valid_phase_mask = np.isfinite(phase_valid)
    if np.sum(valid_phase_mask) < 100:
        print(f"  {name}: 解包裹后有效点太少")
        return None
    
    x_valid = x_valid[valid_phase_mask]
    y_valid = y_valid[valid_phase_mask]
    phase_valid = phase_valid[valid_phase_mask]
    
    # 去除 piston（平均相位）
    phase_centered = phase_valid - np.mean(phase_valid)
    
    # 最小二乘拟合：phase = a*x + b*y + c*r^2
    # 构建设计矩阵
    A = np.column_stack([
        x_valid,           # 线性 x 项（倾斜）
        y_valid,           # 线性 y 项（倾斜）
        x_valid**2 + y_valid**2,  # 二次项（曲率/离焦）
    ])
    
    # 最小二乘求解
    coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_centered, rcond=None)
    
    tilt_x_coeff = coeffs[0]  # rad/mm
    tilt_y_coeff = coeffs[1]  # rad/mm
    curvature_coeff = coeffs[2]  # rad/mm^2
    
    # 计算倾斜角度
    # phase = k * x * sin(theta) => tilt_coeff = k * sin(theta)
    # 对于小角度：theta ≈ tilt_coeff / k
    wavelength_mm = 10.64e-3  # mm (CO2 激光)
    k = 2 * np.pi / wavelength_mm  # 1/mm
    
    tilt_x_rad = tilt_x_coeff / k
    tilt_y_rad = tilt_y_coeff / k
    tilt_x_deg = np.degrees(tilt_x_rad)
    tilt_y_deg = np.degrees(tilt_y_rad)
    
    # 计算倾斜引入的 PV（在光束范围内）
    beam_radius = np.sqrt(np.mean(x_valid**2 + y_valid**2))
    tilt_pv_x = 2 * beam_radius * abs(tilt_x_coeff)  # rad
    tilt_pv_y = 2 * beam_radius * abs(tilt_y_coeff)  # rad
    tilt_pv_total = np.sqrt(tilt_pv_x**2 + tilt_pv_y**2)
    tilt_pv_waves = tilt_pv_total / (2 * np.pi)
    
    # 计算拟合残差
    phase_fit = A @ coeffs
    residual = phase_centered - phase_fit
    residual_rms = np.sqrt(np.mean(residual**2))
    residual_rms_waves = residual_rms / (2 * np.pi)
    
    # 检查相位范围（是否有折叠）
    phase_range = np.max(phase) - np.min(phase)
    phase_range_unwrapped = np.nanmax(phase_unwrapped) - np.nanmin(phase_unwrapped)
    has_wrapping = phase_range > 0.9 * 2 * np.pi
    
    return {
        'tilt_x_coeff': tilt_x_coeff,
        'tilt_y_coeff': tilt_y_coeff,
        'tilt_x_deg': tilt_x_deg,
        'tilt_y_deg': tilt_y_deg,
        'tilt_pv_waves': tilt_pv_waves,
        'curvature_coeff': curvature_coeff,
        'residual_rms_waves': residual_rms_waves,
        'beam_radius': beam_radius,
        'phase_range_raw': phase_range,
        'phase_range_unwrapped': phase_range_unwrapped,
        'has_wrapping': has_wrapping,
        'phase_unwrapped': phase_unwrapped,
    }


# ============================================================
# 创建伽利略OAP系统
# ============================================================

print_section("创建伽利略OAP扩束镜系统")

# 光源参数
wavelength = 10.64      # μm, CO2 激光
w0_input = 10.0         # mm

# 扩束镜焦距
f1 = -500.0             # mm, OAP1 (凸面)
f2 = 1500.0             # mm, OAP2 (凹面)
magnification = -f2 / f1

# 离轴参数
d_off_oap1 = 2 * abs(f1)
d_off_oap2 = 2 * f2
theta = np.radians(45.0)

# 几何参数
d_oap1_to_fold = 500.0
d_fold_to_oap2 = 500.0
d_oap2_to_output = 100.0
total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output

print(f"系统参数:")
print(f"  波长: {wavelength} μm")
print(f"  输入束腰: {w0_input} mm")
print(f"  OAP1 焦距: {f1} mm")
print(f"  OAP2 焦距: {f2} mm")
print(f"  放大倍率: {magnification}x")
print(f"  总光程: {total_path} mm")

# 创建光源
source = GaussianBeamSource(
    wavelength=wavelength,
    w0=w0_input,
    z0=0.0,
)

# 创建系统
system = SequentialOpticalSystem(
    source=source,
    grid_size=512,
    beam_ratio=2.0,
)

# 添加采样面和元件
system.add_sampling_plane(distance=0.0, name="Input")

system.add_surface(ParabolicMirror(
    parent_focal_length=f1,
    thickness=d_oap1_to_fold,
    semi_aperture=20.0,
    off_axis_distance=d_off_oap1,
    tilt_x=theta,
    name="OAP1",
))

system.add_sampling_plane(distance=d_oap1_to_fold, name="After_OAP1")

system.add_surface(FlatMirror(
    thickness=d_fold_to_oap2,
    semi_aperture=30.0,
    tilt_x=theta,
    name="Fold",
))

system.add_sampling_plane(distance=d_oap1_to_fold + d_fold_to_oap2, name="After_Fold")

system.add_surface(ParabolicMirror(
    parent_focal_length=f2,
    thickness=d_oap2_to_output,
    semi_aperture=50.0,
    off_axis_distance=d_off_oap2,
    tilt_x=theta,
    name="OAP2",
))

system.add_sampling_plane(distance=total_path, name="Output")

print("\n系统配置完成！")
print(system.summary())


# ============================================================
# 运行仿真
# ============================================================

print_section("运行仿真")
results = system.run()
print("仿真完成！")


# ============================================================
# 分析各采样面的波前
# ============================================================

print_section("分析各采样面的波前")

print(f"\n{'采样面':<15} {'距离(mm)':<10} {'光束半径':<10} {'相位范围':<12} {'解包裹范围':<12} {'有折叠':<8} {'残差RMS':<12}")
print(f"{'':15} {'':10} {'(mm)':<10} {'(rad)':<12} {'(rad)':<12} {'':8} {'(waves)':<12}")
print("-" * 95)

analysis_results = []

for result in results:
    wavefront = result.wavefront
    sampling = result.sampling
    
    analysis = analyze_wavefront_tilt(wavefront, sampling, result.name)
    
    if analysis:
        wrapping_str = "是" if analysis['has_wrapping'] else "否"
        print(f"{result.name:<15} {result.distance:<10.1f} {analysis['beam_radius']:<10.2f} "
              f"{analysis['phase_range_raw']:<12.4f} {analysis['phase_range_unwrapped']:<12.4f} "
              f"{wrapping_str:<8} {analysis['residual_rms_waves']:<12.6f}")
        
        analysis['name'] = result.name
        analysis['distance'] = result.distance
        analysis['wavefront'] = wavefront
        analysis['sampling'] = sampling
        analysis_results.append(analysis)
    else:
        print(f"{result.name:<15} {result.distance:<10.1f} {'N/A':<10}")


# ============================================================
# 检查光轴状态
# ============================================================

print_section("检查各采样面的光轴状态")

for result in results:
    axis_state = result.axis_state
    if axis_state:
        pos = axis_state.position
        dir = axis_state.direction
        print(f"\n{result.name}:")
        print(f"  位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) mm")
        print(f"  方向: (L={dir.L:.4f}, M={dir.M:.4f}, N={dir.N:.4f})")
        print(f"  光程: {axis_state.path_length:.2f} mm")
    else:
        print(f"\n{result.name}: 无光轴状态信息")


# ============================================================
# 可视化
# ============================================================

print_section("生成可视化图像")

n_planes = len(analysis_results)
fig, axes = plt.subplots(n_planes, 5, figsize=(25, 5*n_planes))

for i, ar in enumerate(analysis_results):
    wavefront = ar['wavefront']
    sampling = ar['sampling']
    n = wavefront.shape[0]
    
    amplitude = np.abs(wavefront)
    phase = np.angle(wavefront)
    phase_unwrapped = ar.get('phase_unwrapped', phase)
    
    # 坐标
    half_size = sampling * n / 2
    extent = [-half_size, half_size, -half_size, half_size]
    coords = np.linspace(-half_size, half_size, n)
    
    # 掩模
    mask = amplitude > 0.1 * np.max(amplitude)
    
    # 1. 振幅分布
    ax1 = axes[i, 0]
    im1 = ax1.imshow(amplitude, extent=extent, cmap='hot', origin='lower')
    ax1.set_title(f"{ar['name']}: Amplitude")
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=ax1, label='Amplitude')
    
    # 2. 原始相位分布（可能有折叠）
    ax2 = axes[i, 1]
    phase_masked = np.where(mask, phase, np.nan)
    im2 = ax2.imshow(phase_masked, extent=extent, cmap='twilight', origin='lower',
                     vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f"{ar['name']}: Phase (raw, wrapped)")
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    # 标注是否有折叠
    if ar['has_wrapping']:
        ax2.text(0.5, 0.95, '⚠️ 有相位折叠', transform=ax2.transAxes, 
                 ha='center', va='top', fontsize=10, color='red',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 3. 解包裹后的相位
    ax3 = axes[i, 2]
    phase_unwrapped_masked = np.where(mask, phase_unwrapped, np.nan)
    im3 = ax3.imshow(phase_unwrapped_masked, extent=extent, cmap='RdBu_r', origin='lower')
    ax3.set_title(f"{ar['name']}: Phase (unwrapped)")
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=ax3, label='Phase (rad)')
    
    # 4. 相位剖面（X方向）
    ax4 = axes[i, 3]
    center = n // 2
    phase_profile_x = phase[center, :]
    phase_unwrapped_profile_x = phase_unwrapped[center, :]
    amp_profile_x = amplitude[center, :]
    
    valid_x = amp_profile_x > 0.1 * np.max(amplitude)
    
    ax4.plot(coords[valid_x], phase_profile_x[valid_x], 'b-', linewidth=2, 
             label='Raw (wrapped)', alpha=0.5)
    ax4.plot(coords[valid_x], phase_unwrapped_profile_x[valid_x], 'r-', linewidth=2, 
             label='Unwrapped')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title(f"{ar['name']}: X Profile")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
    
    # 5. 相位剖面（Y方向）
    ax5 = axes[i, 4]
    phase_profile_y = phase[:, center]
    phase_unwrapped_profile_y = phase_unwrapped[:, center]
    amp_profile_y = amplitude[:, center]
    
    valid_y = amp_profile_y > 0.1 * np.max(amplitude)
    
    ax5.plot(coords[valid_y], phase_profile_y[valid_y], 'b-', linewidth=2, 
             label='Raw (wrapped)', alpha=0.5)
    ax5.plot(coords[valid_y], phase_unwrapped_profile_y[valid_y], 'r-', linewidth=2, 
             label='Unwrapped')
    ax5.set_xlabel('Y (mm)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_title(f"{ar['name']}: Y Profile")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('tests/output/sampling_plane_wavefront_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ 保存: tests/output/sampling_plane_wavefront_analysis.png")


# ============================================================
# 总结
# ============================================================

print_section("总结")

print("\n相位分析结果:")
print("-" * 70)

for ar in analysis_results:
    print(f"\n{ar['name']}:")
    print(f"  原始相位范围: {ar['phase_range_raw']:.4f} rad ({ar['phase_range_raw']/(2*np.pi):.4f} waves)")
    print(f"  解包裹相位范围: {ar['phase_range_unwrapped']:.4f} rad ({ar['phase_range_unwrapped']/(2*np.pi):.4f} waves)")
    print(f"  有相位折叠: {'是' if ar['has_wrapping'] else '否'}")
    print(f"  倾斜: X={ar['tilt_x_deg']:.6f}°, Y={ar['tilt_y_deg']:.6f}°")
    print(f"  倾斜 PV: {ar['tilt_pv_waves']:.6f} waves")
    print(f"  残差 RMS (去除倾斜和曲率后): {ar['residual_rms_waves']:.6f} waves")

# 检查是否有相位折叠问题
has_any_wrapping = any(ar['has_wrapping'] for ar in analysis_results)

print("\n" + "=" * 70)
print("关键发现:")
print("=" * 70)

if has_any_wrapping:
    print("\n⚠️ 检测到相位折叠！")
    print("   原因分析：")
    print("   1. 波前相位变化超过 2π（一个波长）")
    print("   2. 可能是由于：")
    print("      - 波前曲率（发散/会聚光束）")
    print("      - 元件引入的像差")
    print("      - 采样面不在束腰位置")
    print("\n   这是正常现象，因为：")
    print("   - PROPER 存储的是相对于参考球面的相位偏差")
    print("   - 对于发散/会聚光束，相位会随位置变化")
    print("   - 解包裹后可以看到真实的相位分布")
else:
    print("\n✓ 未检测到相位折叠")
    print("   波前相位变化在 2π 范围内")

# 检查波前倾斜
max_tilt_pv = max(ar['tilt_pv_waves'] for ar in analysis_results)
print(f"\n波前倾斜分析:")
print(f"  最大倾斜 PV: {max_tilt_pv:.6f} waves")
if max_tilt_pv < 0.01:
    print("  ✓ 波前基本无倾斜（< 0.01 waves）")
    print("    采样面垂直于光轴，符合预期")
else:
    print("  ! 波前有一定倾斜")

# 检查残差（像差）
max_residual = max(ar['residual_rms_waves'] for ar in analysis_results)
print(f"\n波前像差分析（去除倾斜和曲率后）:")
print(f"  最大残差 RMS: {max_residual:.6f} waves")
if max_residual < 0.1:
    print("  ✓ 波前质量良好（< 0.1 waves RMS）")
else:
    print("  ! 存在一定像差，可能来自：")
    print("    - 元件的几何像差")
    print("    - 数值计算误差")
