# -*- coding: utf-8 -*-
"""
离轴抛物面镜传输误差标准测试文件

测试离轴抛物面镜（OAP）的仿真精度，包括振幅和相位的 PV 与 RMS。

测试参数：
- 曲率半径：200 mm（焦距 100 mm）
- 离轴距离：100 mm（对应 45° 出射角）
- 波长：0.633 μm（He-Ne 激光）
- 束腰半径：3 mm

通过标准：
- 相位 RMS < 10 milli-waves
- 振幅 RMS < 1%

⚠️ 核心回归测试：修改以下模块时必须运行此测试
- src/wavefront_to_rays/element_raytracer.py
- src/hybrid_optical_propagation/hybrid_element_propagator.py

================================================================================
🚫🚫🚫 绝对禁止 🚫🚫🚫

本文件严格遵循 ZMX 文件加载后的坐标处理方式：
- 使用绝对坐标 (x, y, z) 定义表面顶点位置
- 使用姿态角 (rx, ry, rz) 定义表面方向
- 离轴效果完全由 (x, y) 坐标自然产生

以下参数/概念已被永久废弃，永远不存在于本项目中：
- off_axis_distance（离轴距离参数）
- dy（optiland 表面 Y 方向偏心）
- dx（optiland 表面 X 方向偏心）
- 任何形式的"偏心"或"decenter"

正确做法：离轴 100mm = 设置 y=100，就这么简单。
================================================================================
"""

import sys
from pathlib import Path
import numpy as np

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
WAVELENGTH_UM = 0.633      # 波长 (μm)
W0_MM = 10.0               # 束腰半径 (mm)
RADIUS_MM = 2000.0         # 曲率半径 (mm)，R = 2f，焦距 1000mm（凹面镜 R < 0）

# 表面顶点 Y 坐标（离轴效果由此产生）
SURFACE_Y_POSITION_MM = 200.0  # 离轴 100mm

# 网格参数
GRID_SIZE = 256

# 通过标准
PHASE_RMS_THRESHOLD_MWAVES = 10.0   # 相位 RMS 阈值 (milli-waves)
AMPLITUDE_RMS_THRESHOLD_PERCENT = 1.0  # 振幅 RMS 阈值 (%)


# ============================================================
# 主测试函数
# ============================================================

def run_oap_test(
    radius_mm: float = RADIUS_MM,
    surface_y_mm: float = SURFACE_Y_POSITION_MM,
    verbose: bool = True,
    use_global_raytracer: bool = False,
    grid_size: int = GRID_SIZE,
    z_mm: float = 1000.0,
    propagation_method: str = "local_raytracing",
    debug: bool = True,
) -> dict:
    """运行离轴抛物面镜测试
    
    参数:
        radius_mm: 曲率半径 (mm)，R = 2f
        surface_y_mm: 表面顶点 Y 坐标 (mm)，这就是"离轴量"
        verbose: 是否输出详细信息
        use_global_raytracer: 是否使用全局坐标系光线追迹器
        z_mm: 表面 Z 坐标 (mm)
    
    返回:
        测试结果字典
    """
    if verbose:
        print("=" * 70)
        print("离轴抛物面镜传输误差标准测试")
        if use_global_raytracer:
            print("（使用全局坐标系光线追迹器）")
        print("=" * 70)

    # ========================================================
    # 1. 定义光学系统
    # ========================================================
    
    focal_length = radius_mm / 2
    
    if verbose:
        print(f"\n【光学系统参数】")
        print(f"  曲率半径: {radius_mm} mm (焦距 {focal_length} mm)")
        print(f"  表面顶点 Y 坐标: {surface_y_mm} mm")
        print(f"  表面 Z 坐标: {z_mm} mm")
        
        # 计算理论出射角
        exit_angle_deg = 2 * np.degrees(np.arctan(surface_y_mm / (2 * focal_length)))
        print(f"  理论出射角: {exit_angle_deg:.1f}°")
    
    system = bts.OpticalSystem("OAP Test")
    
    # 添加离轴抛物面镜（使用绝对坐标）
    system.add_parabolic_mirror(
        x=0.0,
        y=surface_y_mm,
        z=z_mm,
        radius=radius_mm,
    )
    
    if verbose:
        system.print_info()
    
    # ========================================================
    # 2. 定义光源
    # ========================================================
    
    source = bts.GaussianSource(
        wavelength_um=WAVELENGTH_UM,
        w0_mm=W0_MM,
        grid_size=grid_size,
    )
    
    if verbose:
        print(f"\n【光源参数】")
        print(f"  波长: {WAVELENGTH_UM} μm")
        print(f"  束腰半径: {W0_MM} mm")
        print(f"  网格大小: {GRID_SIZE}")
        print(f"  物理尺寸: {4 * W0_MM} mm (4 × w0)")
    
    # ========================================================
    # 3. 执行仿真
    # ========================================================
    
    if verbose:
        print(f"\n【执行仿真】(Method: {propagation_method})")
    
    try:
        result = bts.simulate(
            system, 
            source, 
            use_global_raytracer=use_global_raytracer,
            propagation_method=propagation_method,
            debug=debug,
        )
    except Exception as e:
        print(f"Simulation Failed: {e}")
        import traceback
        with open(r'd:\BTS\error_info.txt', 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
            
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
        }
    
    # ========================================================
    # 4. 分析结果
    # ========================================================
    
    if verbose:
        print(f"\n[Simulation Results]")
    
    # Get final wavefront data
    final_wavefront = result.get_final_wavefront()
    output_amplitude = final_wavefront.amplitude
    output_phase = final_wavefront.phase
    
    if verbose:
        print(f"  Non-zero elements: {np.sum(output_amplitude > 0)}")
    
    # Create valid region mask
    amp_max = np.max(output_amplitude)
    valid_mask = output_amplitude > 0.01 * amp_max
    
    if verbose:
        print(f"  Valid region pixel count: {np.sum(valid_mask)}")
    
    if np.sum(valid_mask) < 100:
        print("Warning: Valid region is too small")
        return {
            'success': False,
            'error': 'Valid region is too small',
        }

    # --------------------------------------------------------
    # 4.1 Amplitude Analysis
    # --------------------------------------------------------
    
    sampling = final_wavefront.grid.physical_size_mm / final_wavefront.grid.grid_size
    n = output_amplitude.shape[0]
    x = np.linspace(-n/2, n/2-1, n) * sampling
    y = np.linspace(-n/2, n/2-1, n) * sampling
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # Theoretical Gaussian amplitude
    theoretical_amplitude = np.exp(-rr**2 / W0_MM**2)
    theoretical_amplitude = theoretical_amplitude / np.max(theoretical_amplitude) * amp_max
    
    # 振幅误差
    amp_error = output_amplitude - theoretical_amplitude
    amp_error_valid = amp_error[valid_mask]
    
    amp_rms_percent = np.std(amp_error_valid) / amp_max * 100
    amp_pv_percent = np.ptp(amp_error_valid) / amp_max * 100
    
    if verbose:
        print(f"\n  Amplitude Analysis:")
        print(f"    RMS Error: {amp_rms_percent:.3f}%")
        print(f"    PV Error: {amp_pv_percent:.3f}%")
    
    # --------------------------------------------------------
    # 4.2 Phase Analysis
    # --------------------------------------------------------
    
    phase_valid = output_phase[valid_mask]
    x_valid = xx[valid_mask]
    y_valid = yy[valid_mask]
    
    # Calculate residual relative to Pilot Beam
    phase_rms_waves = final_wavefront.get_residual_rms_waves()
    phase_pv_waves = final_wavefront.get_residual_pv_waves()
    
    # Convert to milli-waves
    phase_rms_mwaves = phase_rms_waves * 1000
    phase_pv_mwaves = phase_pv_waves * 1000
    
    if verbose:
        print(f"\n  Phase Analysis:")
        print(f"    Phase RMS (vs Pilot): {phase_rms_mwaves:.3f} milli-waves")
        print(f"    Phase PV (vs Pilot): {phase_pv_mwaves:.3f} milli-waves")
        
        # Save residual image
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            error_waves = final_wavefront.get_residual_phase()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(error_waves, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
            plt.colorbar(label='Residual Phase (waves)')
            plt.title(f'Residual Phase Error\nRMS={phase_rms_mwaves:.3f}mw, PV={phase_pv_mwaves:.3f}mw')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oap_residual_phase_{timestamp}.png"
            plt.savefig(filename)
            print(f"  Saved residual image: {filename}")
            plt.close()
        except ImportError:
            print("  matplotlib not installed, skipping plot")
        except Exception as e:
            print(f"  Plotting failed: {e}")

    # ========================================================
    # 5. Determine Result
    # ========================================================
    
    phase_pass = phase_rms_mwaves < PHASE_RMS_THRESHOLD_MWAVES
    amp_pass = amp_rms_percent < AMPLITUDE_RMS_THRESHOLD_PERCENT
    overall_pass = phase_pass and amp_pass
    
    if verbose:
        print(f"\n[Test Results]")
        print(f"  Phase RMS: {phase_rms_mwaves:.3f} milli-waves " +
              f"({'PASS' if phase_pass else 'FAIL'}, Threshold < {PHASE_RMS_THRESHOLD_MWAVES})")
        print(f"  Amplitude RMS: {amp_rms_percent:.3f}% " +
              f"({'PASS' if amp_pass else 'FAIL'}, Threshold < {AMPLITUDE_RMS_THRESHOLD_PERCENT}%)")
        print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    result_dict = {
        'success': overall_pass,
        'phase_rms_mwaves': phase_rms_mwaves,
        'phase_pv_mwaves': phase_pv_mwaves,
        'amplitude_rms_percent': amp_rms_percent,
        'amplitude_pv_percent': amp_pv_percent,
        'phase_pass': phase_pass,
        'amp_pass': amp_pass,
    }
    
    with open(r'd:\BTS\test_results.txt', 'w', encoding='utf-8') as f:
        f.write(str(result_dict))
        
    return result_dict


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='离轴抛物面镜传输误差测试')
    parser.add_argument('--global-raytracer', action='store_true',
                        help='使用全局坐标系光线追迹器')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    args = parser.parse_args()
    
    result = run_oap_test(
        verbose=True, 
        use_global_raytracer=args.global_raytracer,
        # debug 使用默认值 (True)
    )
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get('success', False) else 1)
