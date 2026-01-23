"""
入射面光线采样误差标准测试文件

测试目标：
1. 测试不同初始束腰位置（近场和远场）下的光线采样精度
2. 比较光线初始化后（通过薄相位元件后，尚未传输至元件面）的相位值
   与根据光线位置从 Pilot Beam 计算的理论相位的差异
3. 分析误差来源

测试场景：
- 近场条件：z << z_R（束腰位置在采样面附近）
- 远场条件：z >> z_R（束腰位置远离采样面）
- 过渡区域：z ≈ z_R

测量指标：
- 相位误差 RMS（milli-waves）
- 相位误差 PV（waves）

注意：此测试仅分析误差，不进行代码修改。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import warnings
warnings.filterwarnings('ignore')


def compute_pilot_beam_phase(x: np.ndarray, y: np.ndarray, 
                              z_mm: float, z_w0_mm: float, 
                              w0_mm: float, wavelength_um: float) -> np.ndarray:
    """计算 Pilot Beam 理论相位
    
    参数:
        x, y: 光线位置（mm）
        z_mm: 当前位置（mm）
        z_w0_mm: 束腰位置（mm）
        w0_mm: 束腰半径（mm）
        wavelength_um: 波长（μm）
    
    返回:
        理论相位（rad），相对于主光线
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm  # 瑞利长度
    
    z_rel = z_mm - z_w0_mm  # 相对于束腰的距离
    
    if abs(z_rel) < 1e-10:
        # 在束腰位置，曲率半径无穷大，相位为 0
        return np.zeros_like(x)
    
    # 严格高斯光束曲率半径公式
    R = z_rel * (1 + (z_R / z_rel)**2)
    
    # Pilot Beam 相位（相对于主光线的相位延迟）
    k = 2 * np.pi / wavelength_mm
    r_sq = x**2 + y**2
    phase = k * r_sq / (2 * R)
    
    return phase


def test_ray_sampling_accuracy(z_ratio: float, verbose: bool = True) -> dict:
    """测试特定 z/z_R 比值下的光线采样精度
    
    参数:
        z_ratio: z/z_R 比值（0 = 束腰位置，1 = 瑞利距离，>1 = 远场）
        verbose: 是否打印详细信息
    
    返回:
        包含测试结果的字典
    """
    from wavefront_to_rays import WavefrontToRaysSampler
    
    # 参数设置
    wavelength_um = 0.633
    w0_mm = 5.0
    grid_size = 256
    physical_size_mm = 40.0
    num_rays = 500
    
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm  # 瑞利长度
    
    # 计算采样位置
    z_mm = z_ratio * z_R
    z_w0_mm = 0.0  # 束腰在原点
    
    if verbose:
        print(f"\n--- z/z_R = {z_ratio:.2f} ---")
        print(f"  瑞利长度 z_R = {z_R:.2f} mm")
        print(f"  采样位置 z = {z_mm:.2f} mm")
    
    # 计算高斯光束参数
    z_rel = z_mm - z_w0_mm
    w_z = w0_mm * np.sqrt(1 + (z_rel / z_R)**2) if z_R > 0 else w0_mm
    
    if abs(z_rel) < 1e-10:
        R = np.inf
    else:
        R = z_rel * (1 + (z_R / z_rel)**2)
    
    if verbose:
        print(f"  光斑半径 w(z) = {w_z:.4f} mm")
        print(f"  曲率半径 R(z) = {R:.2f} mm" if not np.isinf(R) else f"  曲率半径 R(z) = ∞")
    
    # 创建输入波前
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    # 高斯振幅
    amplitude = np.exp(-r_sq / w_z**2)
    
    # 高斯相位（Pilot Beam 相位）
    k = 2 * np.pi / wavelength_mm
    if np.isinf(R):
        phase = np.zeros_like(r_sq)
    else:
        phase = k * r_sq / (2 * R)
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        amplitude=amplitude,
        phase=phase,
        physical_size=physical_size_mm,
        wavelength=wavelength_um,
        num_rays=num_rays,
        distribution="hexapolar",
    )
    
    # 获取输出光线
    output_rays = sampler.get_output_rays()
    
    # 获取光线位置和 OPD
    ray_x = np.asarray(output_rays.x)
    ray_y = np.asarray(output_rays.y)
    ray_opd = np.asarray(output_rays.opd)  # 单位：mm
    
    # 计算光线相位（从 OPD 转换）
    ray_phase = k * ray_opd  # OPD (mm) -> phase (rad)
    
    # 计算 Pilot Beam 理论相位
    pilot_phase = compute_pilot_beam_phase(
        ray_x, ray_y, z_mm, z_w0_mm, w0_mm, wavelength_um
    )
    
    # 计算相位误差
    phase_error = ray_phase - pilot_phase
    
    # 转换为波长数
    phase_error_waves = phase_error / (2 * np.pi)
    
    # 计算统计量
    rms_waves = np.std(phase_error_waves)
    pv_waves = np.max(phase_error_waves) - np.min(phase_error_waves)
    mean_waves = np.mean(phase_error_waves)
    
    if verbose:
        print(f"\n  光线采样结果:")
        print(f"    光线数量: {len(ray_x)}")
        print(f"    光线位置范围: x=[{np.min(ray_x):.2f}, {np.max(ray_x):.2f}] mm")
        print(f"    光线位置范围: y=[{np.min(ray_y):.2f}, {np.max(ray_y):.2f}] mm")
        print(f"\n  相位误差（相对于 Pilot Beam）:")
        print(f"    RMS: {rms_waves * 1000:.4f} milli-waves")
        print(f"    PV:  {pv_waves:.6f} waves")
        print(f"    Mean: {mean_waves:.6f} waves")
    
    return {
        'z_ratio': z_ratio,
        'z_mm': z_mm,
        'z_R': z_R,
        'R': R,
        'w_z': w_z,
        'rms_milli_waves': rms_waves * 1000,
        'pv_waves': pv_waves,
        'mean_waves': mean_waves,
        'ray_x': ray_x,
        'ray_y': ray_y,
        'ray_phase': ray_phase,
        'pilot_phase': pilot_phase,
        'phase_error': phase_error,
    }


def analyze_error_sources(result: dict, verbose: bool = True) -> dict:
    """分析误差来源
    
    参数:
        result: test_ray_sampling_accuracy 的返回结果
        verbose: 是否打印详细信息
    
    返回:
        误差分析结果
    """
    ray_x = result['ray_x']
    ray_y = result['ray_y']
    phase_error = result['phase_error']
    R = result['R']
    z_ratio = result['z_ratio']
    
    # 计算径向距离
    r = np.sqrt(ray_x**2 + ray_y**2)
    
    # 分析误差与径向距离的关系
    # 如果误差与 r² 成正比，说明是曲率相关的误差
    # 如果误差与 r 成正比，说明是倾斜相关的误差
    
    # 拟合 phase_error = a * r² + b * r + c
    if len(r) > 3:
        # 使用最小二乘拟合
        A = np.column_stack([r**2, r, np.ones_like(r)])
        coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_error, rcond=None)
        a, b, c = coeffs
        
        # 计算拟合误差
        fitted = a * r**2 + b * r + c
        fit_residual = phase_error - fitted
        fit_rms = np.std(fit_residual)
    else:
        a, b, c = 0, 0, 0
        fit_rms = np.std(phase_error)
    
    analysis = {
        'quadratic_coeff': a,  # r² 系数（曲率相关）
        'linear_coeff': b,     # r 系数（倾斜相关）
        'constant_coeff': c,   # 常数项（整体偏移）
        'fit_residual_rms': fit_rms,
    }
    
    if verbose:
        print(f"\n  误差来源分析:")
        print(f"    r² 系数（曲率相关）: {a:.6e} rad/mm²")
        print(f"    r 系数（倾斜相关）: {b:.6e} rad/mm")
        print(f"    常数项（整体偏移）: {c:.6e} rad")
        print(f"    拟合残差 RMS: {fit_rms / (2*np.pi) * 1000:.4f} milli-waves")
        
        # 解释误差来源
        print(f"\n  误差来源解释:")
        
        # 曲率误差分析
        if abs(a) > 1e-10:
            # 曲率误差对应的等效曲率半径偏差
            # phase = k * r² / (2R)
            # a = k / (2R_eff) - k / (2R_theory)
            # 1/R_eff - 1/R_theory = 2a/k
            wavelength_mm = 0.633e-3
            k = 2 * np.pi / wavelength_mm
            if not np.isinf(R):
                R_theory_inv = 1 / R
                R_eff_inv = R_theory_inv + 2 * a / k
                if abs(R_eff_inv) > 1e-10:
                    R_eff = 1 / R_eff_inv
                    R_error_percent = (R_eff - R) / R * 100 if R != 0 else 0
                    print(f"    曲率半径偏差: {R_error_percent:.4f}%")
                    print(f"    理论曲率半径: {R:.2f} mm")
                    print(f"    等效曲率半径: {R_eff:.2f} mm")
        
        # 倾斜误差分析
        if abs(b) > 1e-10:
            # 倾斜误差对应的等效倾斜角
            # phase = k * r * sin(theta)
            # b = k * sin(theta)
            wavelength_mm = 0.633e-3
            k = 2 * np.pi / wavelength_mm
            sin_theta = b / k
            if abs(sin_theta) <= 1:
                theta_rad = np.arcsin(sin_theta)
                theta_deg = np.degrees(theta_rad)
                print(f"    等效倾斜角: {theta_deg:.6f}°")
        
        # 常数偏移分析
        if abs(c) > 1e-10:
            c_waves = c / (2 * np.pi)
            print(f"    常数相位偏移: {c_waves:.6f} waves")
    
    return analysis


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 70)
    print("入射面光线采样误差标准测试")
    print("=" * 70)
    
    # 测试不同的 z/z_R 比值
    z_ratios = [
        0.0,    # 束腰位置
        0.001,  # 极近场
        0.01,   # 近场
        0.1,    # 近场
        0.5,    # 近场
        1.0,    # 瑞利距离
        2.0,    # 远场
        5.0,    # 远场
        10.0,   # 远场
    ]
    
    results = []
    
    for z_ratio in z_ratios:
        try:
            result = test_ray_sampling_accuracy(z_ratio, verbose=True)
            analysis = analyze_error_sources(result, verbose=True)
            result['analysis'] = analysis
            results.append(result)
        except Exception as e:
            print(f"\n  测试失败: {e}")
            results.append({
                'z_ratio': z_ratio,
                'error': str(e),
            })
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    print(f"\n{'z/z_R':>8} | {'z (mm)':>12} | {'RMS (milli-waves)':>18} | {'PV (waves)':>12} | {'状态':>6}")
    print("-" * 70)
    
    for r in results:
        if 'error' in r:
            print(f"{r['z_ratio']:>8.3f} | {'N/A':>12} | {'N/A':>18} | {'N/A':>12} | {'失败':>6}")
        else:
            print(f"{r['z_ratio']:>8.3f} | {r['z_mm']:>12.2f} | {r['rms_milli_waves']:>18.4f} | {r['pv_waves']:>12.6f} | {'成功':>6}")
    
    # 分析总结
    print("\n" + "=" * 70)
    print("误差分析总结")
    print("=" * 70)
    
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        rms_values = [r['rms_milli_waves'] for r in successful_results]
        pv_values = [r['pv_waves'] for r in successful_results]
        
        print(f"\n统计信息:")
        print(f"  RMS 范围: {min(rms_values):.4f} - {max(rms_values):.4f} milli-waves")
        print(f"  RMS 平均: {np.mean(rms_values):.4f} milli-waves")
        print(f"  PV 范围:  {min(pv_values):.6f} - {max(pv_values):.6f} waves")
        print(f"  PV 平均:  {np.mean(pv_values):.6f} waves")
    
    return results


def plot_results(results: list, save_path: str = None):
    """绘制测试结果图表"""
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("没有成功的测试结果可绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMS 误差 vs z/z_R
    ax1 = axes[0, 0]
    z_ratios = [r['z_ratio'] for r in successful_results]
    rms_values = [r['rms_milli_waves'] for r in successful_results]
    ax1.semilogy(z_ratios, rms_values, 'bo-', markersize=8)
    ax1.axvline(x=1.0, color='r', linestyle='--', label='z = z_R')
    ax1.set_xlabel('z / z_R')
    ax1.set_ylabel('RMS 误差 (milli-waves)')
    ax1.set_title('光线采样相位误差 RMS vs 传播距离')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. PV 误差 vs z/z_R
    ax2 = axes[0, 1]
    pv_values = [r['pv_waves'] for r in successful_results]
    ax2.semilogy(z_ratios, pv_values, 'ro-', markersize=8)
    ax2.axvline(x=1.0, color='b', linestyle='--', label='z = z_R')
    ax2.set_xlabel('z / z_R')
    ax2.set_ylabel('PV 误差 (waves)')
    ax2.set_title('光线采样相位误差 PV vs 传播距离')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差系数分析
    ax3 = axes[1, 0]
    quad_coeffs = [abs(r['analysis']['quadratic_coeff']) for r in successful_results if 'analysis' in r]
    linear_coeffs = [abs(r['analysis']['linear_coeff']) for r in successful_results if 'analysis' in r]
    z_ratios_with_analysis = [r['z_ratio'] for r in successful_results if 'analysis' in r]
    
    if quad_coeffs and linear_coeffs:
        ax3.semilogy(z_ratios_with_analysis, quad_coeffs, 'b^-', label='|r² 系数| (曲率)', markersize=8)
        ax3.semilogy(z_ratios_with_analysis, linear_coeffs, 'gs-', label='|r 系数| (倾斜)', markersize=8)
        ax3.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('z / z_R')
        ax3.set_ylabel('系数绝对值')
        ax3.set_title('误差成分分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 选择一个典型结果绘制误差分布
    ax4 = axes[1, 1]
    # 选择 z/z_R = 1.0 或最接近的结果
    target_ratio = 1.0
    closest_result = min(successful_results, key=lambda r: abs(r['z_ratio'] - target_ratio))
    
    ray_x = closest_result['ray_x']
    ray_y = closest_result['ray_y']
    phase_error_waves = closest_result['phase_error'] / (2 * np.pi)
    
    scatter = ax4.scatter(ray_x, ray_y, c=phase_error_waves * 1000, 
                          cmap='RdBu_r', s=20, alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='相位误差 (milli-waves)')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    ax4.set_title(f'相位误差分布 (z/z_R = {closest_result["z_ratio"]:.2f})')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")
    
    plt.close()


def analyze_error_mechanism():
    """分析误差机制"""
    print("\n" + "=" * 70)
    print("误差机制分析")
    print("=" * 70)
    
    print("""
光线采样误差的主要来源分析：

1. 相位插值误差
   - WavefrontToRaysSampler 使用双线性插值从相位网格获取光线位置的相位
   - 当相位变化剧烈（远场，大曲率）时，插值误差增大
   - 网格分辨率有限导致的离散化误差

2. 相位梯度计算误差
   - 光线方向由相位梯度决定
   - 数值微分的精度受网格分辨率限制
   - 边缘区域的梯度计算可能不准确

3. 相位折叠（Wrapping）问题
   - 当相位超过 [-π, π] 范围时会发生折叠
   - 远场条件下相位变化大，更容易发生折叠
   - 折叠边界处的插值会产生大误差

4. 坐标系统精度
   - 光线位置的数值精度
   - 浮点数运算误差的累积

5. Pilot Beam 近似误差
   - Pilot Beam 假设理想高斯光束
   - 实际波前可能有像差
   - 近场和远场的 Pilot Beam 参数计算方式不同

预期行为：
- 近场（z << z_R）：相位变化小，误差应该较小
- 远场（z >> z_R）：相位变化大，插值误差增大
- 瑞利距离附近（z ≈ z_R）：过渡区域，误差可能有变化
""")


def main():
    """主函数"""
    # 运行综合测试
    results = run_comprehensive_test()
    
    # 绘制结果
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    plot_results(results, str(output_dir / 'entrance_ray_sampling_accuracy.png'))
    
    # 分析误差机制
    analyze_error_mechanism()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
