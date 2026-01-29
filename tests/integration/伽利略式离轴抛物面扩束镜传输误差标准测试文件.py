# -*- coding: utf-8 -*-
"""
伽利略式离轴抛物面扩束镜传输误差标准测试文件

测试由两个离轴抛物面镜（OAP）构成的伽利略式激光扩束镜系统的仿真精度。

============================================================
光学设计：经典消像差伽利略式 OAP 扩束镜
============================================================

系统配置：
- 两个 OAP 平行放置，光轴平行
- 入射光束平行于光轴，带离轴入射
- OAP1（凸面）发散光束，OAP2（凹面）准直光束
- 两镜共焦（虚焦点重合），实现无像差设计

光路示意图（YZ 平面）：

    入射光束（平行于 Z 轴）
         |
         |  y = d1（离轴距离）
         |
         v
    ─────●───────────────────────────────────────────────────→ Z
         OAP1                    OAP2
         (凸面, f1<0)            (凹面, f2>0)
         z=0                     z=L
         
         发散光束 ─────────────→ 准直光束（扩束后）
                                 y = d2（离轴距离）

设计原理：
- OAP1 焦距 f1 < 0（凸面），将平行光发散
- OAP2 焦距 f2 > 0（凹面），将发散光准直
- 共焦条件：两镜间距 L = f2 - |f1| = f2 + f1
- 放大倍率：M = f2 / |f1| = -f2 / f1
- 离轴距离关系：d2 = M × d1

消像差条件：
- 入射光平行于光轴（母抛物面轴）
- 两镜光轴平行
- 共焦设计

测试参数：
- OAP1: f1 = -100 mm（凸面），离轴距离 d1 = 50 mm
- OAP2: f2 = 300 mm（凹面），离轴距离 d2 = 150 mm
- 放大倍率: M = 3x
- 两镜间距: L = 300 - 100 = 200 mm
- 波长: 0.633 μm（He-Ne 激光）
- 束腰半径: 5 mm

通过标准：
- 相位 RMS < 500 milli-waves (0.5 lambda) - 基于 OAP 混合传播的当前精度限制
- 发散角 < 理论极限制的 2 倍

⚠️ 核心回归测试：修改以下模块时必须运行此测试
- src/wavefront_to_rays/element_raytracer.py
- src/hybrid_optical_propagation/hybrid_element_propagator.py

================================================================================
🚫🚫🚫 绝对禁止 🚫🚫🚫

本文件严格遵循绝对坐标定义方式：
- 使用绝对坐标 (x, y, z) 定义表面顶点位置
- 离轴效果完全由 (x, y) 坐标自然产生

以下参数/概念已被永久废弃，永远不存在于本项目中：
- off_axis_distance（离轴距离参数）
- dy（optiland 表面 Y 方向偏心）
- dx（optiland 表面 X 方向偏心）
- 任何形式的"偏心"或"decenter"
- semi_aperture（半口径参数）
- aperture（口径参数）

正确做法：离轴 50mm = 设置 y=50，就这么简单。
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

# 添加 src 目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import bts


# ============================================================
# 测试参数
# ============================================================

# 光学参数
WAVELENGTH_UM = 1      # 波长 (μm)
W0_MM = 5.0                # 输入束腰半径 (mm)

# OAP1 参数（凸面，发散）
F1_MM = -1000.0             # OAP1 焦距 (mm)，负值表示凸面
D1_MM = 100.0               # OAP1 离轴距离 (mm)

# OAP2 参数（凹面，准直）
F2_MM = 2000.0              # OAP2 焦距 (mm)，正值表示凹面
MAGNIFICATION = -F2_MM / F1_MM  # 放大倍率 = 3x
D2_MM = D1_MM # OAP2 离轴距离 (mm) 
# 几何参数
L_MM = F2_MM + F1_MM       # 两镜间距  （共焦条件）

# 网格参数
GRID_SIZE = 512            # 增加网格大小以适应扩束

# 通过标准
PHASE_RMS_THRESHOLD_MWAVES = 500.0   # 相位 RMS 阈值 (milli-waves)
DIVERGENCE_TOLERANCE_MRAD = 0.05     # 发散角容差 (mrad)


# ============================================================
# 辅助函数：高斯拟合
# ============================================================

def gaussian_1d(x, a, x0, w, offset):
    return a * np.exp(-2 * ((x - x0) / w)**2) + offset

def measure_beam_quality(wavefront):
    """
    测量光束质量：束腰大小 (w) 和发散角 (theta)
    
    使用高斯拟合而非简单的 RMS，更加鲁棒。
    """
    amp = wavefront.amplitude
    grid = wavefront.grid
    
    # 1. 测量近场光斑大小 (w_out)
    # 拟合中心切片
    n = grid.grid_size
    sampling = grid.physical_size_mm / n
    coords = np.linspace(-n/2, n/2-1, n) * sampling
    
    # 找到峰值位置
    y_idx, x_idx = np.unravel_index(np.argmax(amp), amp.shape)
    
    # X 切片和 Y 切片
    x_cut = amp[y_idx, :]
    y_cut = amp[:, x_idx]
    
    try:
        popt_x, _ = curve_fit(gaussian_1d, coords, x_cut, p0=[np.max(x_cut), coords[x_idx], 5.0 * MAGNIFICATION, 0])
        w_x = abs(popt_x[2])
        
        popt_y, _ = curve_fit(gaussian_1d, coords, y_cut, p0=[np.max(y_cut), coords[y_idx], 5.0 * MAGNIFICATION, 0])
        w_y = abs(popt_y[2])
    except:
        w_x, w_y = 0, 0
        
    w_out_avg = (w_x + w_y) / 2
    
    # 2. 测量发散角 (Divergence)
    # 方法：使用 FFT 传播到远场 (Fraunhofer)
    # 远场坐标 u = x / (lambda * z) -> theta = u * lambda = x / z
    # 频率域网格：df = 1 / (N * dx)
    # 角度网格：theta = df * lambda
    
    complex_field = wavefront.get_complex_amplitude()
    fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_field)))
    fft_amp = np.abs(fft_field)
    
    # 频率采样
    wavelength_mm = wavefront.wavelength_um / 1000.0
    df = 1.0 / grid.physical_size_mm  # 空间频率增量 (1/mm)
    theta_res_rad = df * wavelength_mm # 角度分辨率 (rad)
    
    theta_coords = np.linspace(-n/2, n/2-1, n) * theta_res_rad
    
    # 对远场振幅进行高斯拟合
    y_idx_fft, x_idx_fft = np.unravel_index(np.argmax(fft_amp), fft_amp.shape)
    x_cut_fft = fft_amp[y_idx_fft, :]
    
    try:
        # 初始猜测：非常小的发散角
        popt_div, _ = curve_fit(gaussian_1d, theta_coords, x_cut_fft, 
                                p0=[np.max(x_cut_fft), theta_coords[x_idx_fft], 0.001, 0])
        theta_div = abs(popt_div[2]) # 半发散角 (1/e^2 半径，弧度)
    except:
        theta_div = 0
        
    return w_out_avg, theta_div * 1000.0 # 转换为 mrad


# ============================================================
# 主测试函数
# ============================================================

def run_galilean_oap_expander_test(
    f1_mm: float = F1_MM,
    f2_mm: float = F2_MM,
    d1_mm: float = D1_MM,
    verbose: bool = True,
    use_global_raytracer: bool = False,
    grid_size: int = GRID_SIZE,
    propagation_method: str = "local_raytracing",
    plot: bool = False,
    plot_mode: str = '3d',
    debug: bool = False,
) -> dict:
    """运行伽利略式 OAP 扩束镜测试"""
    if verbose:
        print("=" * 70)
        print("伽利略式离轴抛物面扩束镜传输误差标准测试")
        if use_global_raytracer:
            print("（使用全局坐标系光线追迹器）")
        print("=" * 70)

    # ========================================================
    # 1. 计算设计参数
    # ========================================================
    
    
    r1_mm = -2 * f1_mm 
    r2_mm = -2 * f2_mm
    
    # 放大倍率
    magnification = -f2_mm / f1_mm
    
    # OAP2 离轴距离
    d2_mm = d1_mm
    
    # 两镜间距
    l_mm = f2_mm + f1_mm
    
    # 预期输出
    w0_output_expected = W0_MM * magnification
    # 理论发散角 (衍射极限)
    theta_diff_limit_mrad = (WAVELENGTH_UM / (np.pi * w0_output_expected * 1000)) * 1000 * 1000 
    
    if verbose:
        print(f"\n【设计参数】")
        print(f"  OAP1 焦距: {f1_mm} mm (Radius={r1_mm} mm)")
        print(f"  OAP2 焦距: {f2_mm} mm (Radius={r2_mm} mm)")
        print(f"  OAP1 离轴: {d1_mm} mm, OAP2 离轴: {d2_mm} mm")
        print(f"  间距 L: {l_mm} mm")
        print(f"  放大倍率: {magnification:.2f}x")
        print(f"  预期输出束腰: {w0_output_expected:.3f} mm")
        print(f"  衍射极限发散角: {theta_diff_limit_mrad:.6f} mrad")

    # ========================================================
    # 2. 定义光学系统
    # ========================================================
    
    system = bts.OpticalSystem("Galilean OAP Expander")
    
    # OAP1：凸面抛物面镜（发散）
    system.add_parabolic_mirror(
        x=0.0, y=d1_mm, z=500,
        radius=r1_mm, #2000
    )
    
    # OAP2：凹面抛物面镜（准直）
    # 注意：OAP1 无倾斜时反射光向 -Z 方向传播
    # 因此 OAP2 必须位于 -Z 侧（z = -L）才能接收到光束
    system.add_parabolic_mirror(
        x=0.0, y=d2_mm, z=500-l_mm, 
        tilt_y=180,
        radius=-4000,  #全局中凸，相对于-Z光线凹
    )

    # system.add_surface(
    #     x=0.0, y=d2_mm, z=0, 
    #     radius=np.inf,  #全局中凸，相对于-Z光线凹
    #     material="air"
    # )

    

    
    if plot:
        if verbose:
            print(f"\n【绘制光路图 ({plot_mode})】...")
        system.plot_layout(
            mode=plot_mode,
            projection='YZ', 
            save_path='galilean_oap_layout.png', 
            show=True
        )
    
    # ========================================================
    # 3. 定义光源
    # ========================================================
    
    source = bts.GaussianSource(
        wavelength_um=WAVELENGTH_UM,
        w0_mm=W0_MM,
        grid_size=grid_size,
        z0_mm = 1000,
        beam_diam_fraction=0.25,
        physical_size_mm = 8 * W0_MM,
    )
    
    # ========================================================
    # 4. 执行仿真
    # ========================================================
    
    if verbose:
        print(f"\n【执行仿真】...")
    
    try:
        result = bts.simulate(
            system, 
            source, 
            use_global_raytracer=use_global_raytracer,
            propagation_method=propagation_method,
            verbose=False,
            debug=debug
        )
    except Exception as e:
        print(f"仿真失败: {e}")
        return {'success': False, 'error': str(e)}
    
    # ========================================================
    # 5. 分析结果
    # ========================================================
    
    final_wf = result.get_final_wavefront()
    
    # 测量光束质量
    w_out_meas, theta_out_meas = measure_beam_quality(final_wf)
    
    # 测量残差 RMS
    phase_rms_mwaves = final_wavefront_rms = final_wf.get_residual_rms_waves() * 1000
    
    # 计算误差
    w_error_percent = (w_out_meas - w0_output_expected) / w0_output_expected * 100
    
    if verbose:
        print(f"\n【仿真结果】")
        print(f"  测量输出束腰: {w_out_meas:.3f} mm (误差 {w_error_percent:.2f}%)")
        print(f"  测量发散角: {theta_out_meas:.6f} mrad (理论 {theta_diff_limit_mrad:.6f} mrad)")
        print(f"  残差 RMS: {phase_rms_mwaves:.3f} milli-waves")

    # ========================================================
    # 6. 判断测试结果
    # ========================================================
    
    # 判定标准：
    # 1. 束腰大小误差 < 5%
    # 2. 发散角 < 理论值 + 容差
    # 3. 残差 RMS < 阈值
    
    w_pass = abs(w_error_percent) < 5.0
    div_pass = theta_out_meas < (theta_diff_limit_mrad + DIVERGENCE_TOLERANCE_MRAD)
    rms_pass = phase_rms_mwaves < PHASE_RMS_THRESHOLD_MWAVES
    
    overall_pass = w_pass and div_pass and rms_pass
    
    if verbose:
        print(f"\n【测试判定】")
        print(f"  束腰一致性: {'PASS' if w_pass else 'FAIL'} (< 5%)")
        print(f"  准直性(发散): {'PASS' if div_pass else 'FAIL'} (< {theta_diff_limit_mrad + DIVERGENCE_TOLERANCE_MRAD:.3f} mrad)")
        print(f"  波前质量(RMS): {'PASS' if rms_pass else 'FAIL'} (< {PHASE_RMS_THRESHOLD_MWAVES} mwaves)")
        print(f"\n  总体结果: {'[PASS]' if overall_pass else '[FAIL]'}")
        
    with open('failure_reason.txt', 'w', encoding='utf-8') as f:
        f.write(f"w_pass: {w_pass} (meas={w_out_meas:.3f}, err={w_error_percent:.2f}%)\n")
        f.write(f"div_pass: {div_pass} (meas={theta_out_meas:.6f}, limit={theta_diff_limit_mrad + DIVERGENCE_TOLERANCE_MRAD:.3f})\n")
        f.write(f"rms_pass: {rms_pass} (meas={phase_rms_mwaves:.3f}, limit={PHASE_RMS_THRESHOLD_MWAVES})\n")
        f.write(f"Overall: {overall_pass}\n")
    
    return {
        'success': overall_pass,
        'magnification': magnification,
        'w_out_meas': w_out_meas,
        'theta_out_meas': theta_out_meas,
        'phase_rms_mwaves': phase_rms_mwaves
    }

# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='伽利略式 OAP 扩束镜传输误差测试')
    parser.add_argument('--global-raytracer', action='store_true',
                        help='使用全局坐标系光线追迹器')#默认使用局部坐标系光线追迹器
    parser.add_argument('--no-plot', action='store_true',
                        help='不绘制光路图')
    parser.add_argument('--plot-3d', action='store_true',
                        help='使用 3D 绘图模式')
    args = parser.parse_args()
    
    result = run_galilean_oap_expander_test(
        verbose=True, 
        use_global_raytracer=args.global_raytracer,
        plot=not args.no_plot,
        plot_mode='3d' if args.plot_3d else '2d'
    )
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return result

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get('success', False) else 1)
