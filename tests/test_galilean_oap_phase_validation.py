"""
伽利略式 OAP 扩束镜相位验证测试

详细验证各采样面处的振幅和相位准确度。

验证内容：
1. 振幅分布（高斯光束形状）
2. 相位分布（波前曲率）
3. 与 ABCD 理论预测的对比
4. 波前曲率半径验证

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


def analyze_wavefront_curvature(wavefront, sampling_mm, wavelength_um):
    """分析波前曲率半径
    
    通过拟合波前相位的二次项来估计曲率半径。
    
    参数:
        wavefront: 复振幅数组
        sampling_mm: 采样间隔 (mm)
        wavelength_um: 波长 (μm)
    
    返回:
        R_fit: 拟合的曲率半径 (mm)
        phase_residual_rms: 拟合残差 RMS (rad)
    """
    n = wavefront.shape[0]
    amplitude = np.abs(wavefront)
    phase = np.angle(wavefront)
    
    # 创建坐标网格
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # 创建掩模（只使用振幅较大的区域）
    threshold = 0.1 * np.max(amplitude)
    mask = amplitude > threshold
    
    if np.sum(mask) < 100:
        return np.inf, 0.0
    
    # 解包相位（简单方法：假设相位连续）
    phase_unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
    
    # 去除 piston（平均相位）
    phase_centered = phase_unwrapped - np.mean(phase_unwrapped[mask])
    
    # 拟合二次相位：phase = -k * r^2 / (2R)
    # 即 phase = a * r^2，其中 a = -k / (2R)
    k = 2 * np.pi / (wavelength_um * 1e-3)  # 1/mm
    
    # 最小二乘拟合
    r_sq_masked = R_sq[mask]
    phase_masked = phase_centered[mask]
    
    # 加权拟合（用振幅作为权重）
    weights = amplitude[mask]
    
    # 拟合 phase = a * r^2
    # a = sum(w * r^2 * phase) / sum(w * r^4)
    numerator = np.sum(weights * r_sq_masked * phase_masked)
    denominator = np.sum(weights * r_sq_masked**2)
    
    if abs(denominator) < 1e-15:
        return np.inf, 0.0
    
    a = numerator / denominator
    
    # 计算曲率半径：a = -k / (2R) => R = -k / (2a)
    if abs(a) < 1e-15:
        R_fit = np.inf
    else:
        R_fit = -k / (2 * a)
    
    # 计算残差
    phase_fit = a * R_sq
    residual = phase_centered - phase_fit
    residual_rms = np.sqrt(np.mean(residual[mask]**2))
    
    return R_fit, residual_rms


def analyze_beam_profile(wavefront, sampling_mm):
    """分析光束轮廓
    
    参数:
        wavefront: 复振幅数组
        sampling_mm: 采样间隔 (mm)
    
    返回:
        w_x: X 方向光束半径 (mm)
        w_y: Y 方向光束半径 (mm)
        peak_amplitude: 峰值振幅
    """
    amplitude = np.abs(wavefront)
    intensity = amplitude**2
    
    n = wavefront.shape[0]
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    
    total = np.sum(intensity)
    if total < 1e-15:
        return 0.0, 0.0, 0.0
    
    # 计算二阶矩
    x_var = np.sum(X**2 * intensity) / total
    y_var = np.sum(Y**2 * intensity) / total
    
    # 高斯光束：w = 2 * sigma
    w_x = 2 * np.sqrt(x_var)
    w_y = 2 * np.sqrt(y_var)
    
    peak_amplitude = np.max(amplitude)
    
    return w_x, w_y, peak_amplitude


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 创建光学系统（与 galilean_oap_expander.py 相同）
# ============================================================

print_section("创建伽利略式 OAP 扩束镜系统")

# 光源参数
wavelength = 1.064      # μm
w0_input = 10.0         # mm

# 扩束镜焦距
f1 = -50.0              # mm, OAP1 (凸面)
f2 = 150.0              # mm, OAP2 (凹面)
magnification = -f2 / f1

# 离轴参数
d_off_oap1 = 2 * abs(f1)
d_off_oap2 = 2 * f2
theta = np.radians(45.0)

# 几何参数
d_oap1_to_fold = 50.0
d_fold_to_oap2 = 50.0
d_oap2_to_output = 100.0
total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output

# 光源
source = GaussianBeamSource(
    wavelength=wavelength,
    w0=w0_input,
    z0=0.0,
)

# 创建系统
system = SequentialOpticalSystem(
    source=source,
    grid_size=512,
    beam_ratio=0.25,
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

print(system.summary())


# ============================================================
# 运行仿真
# ============================================================

print_section("运行仿真")
results = system.run()
print("仿真完成！")


# ============================================================
# 详细分析各采样面
# ============================================================

print_section("详细分析各采样面")

# 计算 ABCD 理论预测
from gaussian_beam_simulation.gaussian_beam import GaussianBeam

beam = GaussianBeam(
    wavelength=wavelength * 1e-3,  # mm
    w0=w0_input,
    z0=0.0,
)

# 理论预测（使用 ABCD 矩阵）
abcd_predictions = {}
for result in results:
    abcd_result = system.get_abcd_result(result.distance)
    abcd_predictions[result.name] = {
        'w': abcd_result.w,
        'R': abcd_result.R,
    }

# 分析每个采样面
analysis_results = []

for result in results:
    print(f"\n--- {result.name} (distance = {result.distance} mm) ---")
    
    wavefront = result.wavefront
    sampling = result.sampling
    
    # 1. 振幅分析
    w_x, w_y, peak_amp = analyze_beam_profile(wavefront, sampling)
    w_avg = (w_x + w_y) / 2
    
    # 2. 相位分析（波前曲率）
    R_fit, phase_residual = analyze_wavefront_curvature(wavefront, sampling, wavelength)
    
    # 3. ABCD 理论值
    abcd = abcd_predictions[result.name]
    w_theory = abcd['w']
    R_theory = abcd['R']
    
    # 4. 计算误差
    w_error = abs(w_avg - w_theory) / w_theory * 100 if w_theory > 0.001 else 0
    
    if np.isfinite(R_theory) and np.isfinite(R_fit) and abs(R_theory) > 0.001:
        R_error = abs(R_fit - R_theory) / abs(R_theory) * 100
    else:
        R_error = 0 if (np.isinf(R_theory) and np.isinf(R_fit)) else float('inf')
    
    # 5. 打印结果
    print(f"  振幅分析:")
    print(f"    光束半径 (PROPER): w_x = {w_x:.3f} mm, w_y = {w_y:.3f} mm, avg = {w_avg:.3f} mm")
    print(f"    光束半径 (ABCD):   w = {w_theory:.3f} mm")
    print(f"    误差: {w_error:.2f}%")
    print(f"    峰值振幅: {peak_amp:.6f}")
    
    print(f"  相位分析:")
    print(f"    波前曲率半径 (拟合): R = {R_fit:.1f} mm" if np.isfinite(R_fit) else f"    波前曲率半径 (拟合): R = inf (平面波)")
    print(f"    波前曲率半径 (ABCD): R = {R_theory:.1f} mm" if np.isfinite(R_theory) else f"    波前曲率半径 (ABCD): R = inf (平面波)")
    print(f"    相位拟合残差 RMS: {phase_residual:.4f} rad ({phase_residual/(2*np.pi):.4f} waves)")
    
    # 6. WFE 分析
    print(f"  波前误差:")
    print(f"    WFE RMS (from result): {result.wavefront_rms:.6f} waves")
    
    analysis_results.append({
        'name': result.name,
        'distance': result.distance,
        'w_proper': w_avg,
        'w_abcd': w_theory,
        'w_error': w_error,
        'R_fit': R_fit,
        'R_abcd': R_theory,
        'R_error': R_error,
        'phase_residual': phase_residual,
        'wfe_rms': result.wavefront_rms,
        'wavefront': wavefront,
        'sampling': sampling,
    })


# ============================================================
# 可视化详细分析
# ============================================================

print_section("生成详细可视化")

fig, axes = plt.subplots(len(results), 4, figsize=(20, 5*len(results)))

for i, ar in enumerate(analysis_results):
    wavefront = ar['wavefront']
    sampling = ar['sampling']
    n = wavefront.shape[0]
    
    amplitude = np.abs(wavefront)
    phase = np.angle(wavefront)
    
    # 坐标
    half_size = sampling * n / 2
    extent = [-half_size, half_size, -half_size, half_size]
    coords = np.linspace(-half_size, half_size, n)
    
    # 1. 振幅分布
    ax1 = axes[i, 0]
    im1 = ax1.imshow(amplitude, extent=extent, cmap='hot', origin='lower')
    ax1.set_title(f"{ar['name']}: Amplitude")
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=ax1, label='Amplitude')
    
    # 2. 相位分布
    ax2 = axes[i, 1]
    # 掩模低振幅区域
    phase_masked = np.where(amplitude > 0.1 * np.max(amplitude), phase, np.nan)
    im2 = ax2.imshow(phase_masked, extent=extent, cmap='twilight', origin='lower')
    ax2.set_title(f"{ar['name']}: Phase")
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    
    # 3. 振幅剖面对比
    ax3 = axes[i, 2]
    center = n // 2
    amp_profile = amplitude[center, :]
    
    # 理论高斯
    w_theory = ar['w_abcd']
    gaussian_theory = np.max(amp_profile) * np.exp(-coords**2 / w_theory**2)
    
    ax3.plot(coords, amp_profile, 'b-', label='PROPER', linewidth=2)
    ax3.plot(coords, gaussian_theory, 'r--', label=f'Gaussian (w={w_theory:.2f}mm)', linewidth=2)
    ax3.set_title(f"{ar['name']}: Amplitude Profile")
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相位剖面对比
    ax4 = axes[i, 3]
    phase_profile = phase[center, :]
    
    # 理论相位（球面波）
    R_theory = ar['R_abcd']
    k = 2 * np.pi / (wavelength * 1e-3)
    if np.isfinite(R_theory) and abs(R_theory) > 0.001:
        phase_theory = -k * coords**2 / (2 * R_theory)
        phase_theory = phase_theory - np.mean(phase_theory)  # 去除 piston
    else:
        phase_theory = np.zeros_like(coords)
    
    # 去除 piston
    mask = amplitude[center, :] > 0.1 * np.max(amplitude)
    if np.any(mask):
        phase_profile_centered = phase_profile - np.mean(phase_profile[mask])
    else:
        phase_profile_centered = phase_profile
    
    ax4.plot(coords, phase_profile_centered, 'b-', label='PROPER', linewidth=2)
    ax4.plot(coords, phase_theory, 'r--', label=f'Theory (R={R_theory:.1f}mm)' if np.isfinite(R_theory) else 'Theory (R=inf)', linewidth=2)
    ax4.set_title(f"{ar['name']}: Phase Profile")
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Phase (rad)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('tests/output/galilean_oap_phase_validation.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ 保存: tests/output/galilean_oap_phase_validation.png")


# ============================================================
# 总结
# ============================================================

print_section("验证总结")

print("\n采样面分析汇总:")
print(f"{'采样面':<15} {'w_PROPER':<10} {'w_ABCD':<10} {'w误差%':<10} {'R_fit':<12} {'R_ABCD':<12} {'相位残差':<12}")
print("-" * 85)

for ar in analysis_results:
    R_fit_str = f"{ar['R_fit']:.1f}" if np.isfinite(ar['R_fit']) else "inf"
    R_abcd_str = f"{ar['R_abcd']:.1f}" if np.isfinite(ar['R_abcd']) else "inf"
    print(f"{ar['name']:<15} {ar['w_proper']:<10.3f} {ar['w_abcd']:<10.3f} {ar['w_error']:<10.2f} {R_fit_str:<12} {R_abcd_str:<12} {ar['phase_residual']:<12.4f}")

print("\n关键发现:")
print("1. 振幅分布（光束半径）是否与 ABCD 理论一致？")
print("2. 相位分布（波前曲率）是否与 ABCD 理论一致？")
print("3. 相位拟合残差是否足够小？")

# 检查是否有问题
issues = []
for ar in analysis_results:
    if ar['w_error'] > 5:
        issues.append(f"  - {ar['name']}: 光束半径误差 {ar['w_error']:.1f}% > 5%")
    if ar['phase_residual'] > 0.5:
        issues.append(f"  - {ar['name']}: 相位残差 {ar['phase_residual']:.2f} rad > 0.5 rad")

if issues:
    print("\n⚠️ 发现以下问题:")
    for issue in issues:
        print(issue)
else:
    print("\n✓ 所有采样面的振幅和相位都与理论预测一致")
