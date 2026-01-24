"""
不同倾斜角度平面镜传输误差标准测试文件

测试目标：
1. 系统性测试多个倾斜角度（0° 到 60°）
2. 记录每个角度的相位误差 RMS 和 PV
3. 验证所有角度的精度一致性
4. 作为回归测试确保倾斜角度处理的正确性

测试场景：
- 近场高斯光束（z << z_R）
- 平面镜反射
- 不同倾斜角度（0°, 5°, 10°, ..., 60°）

预期结果：
- 所有角度的 RMS 误差应一致（~0.343 milli-waves）
- 所有角度的 PV 误差应一致（~0.0006 waves）
"""

import sys
from pathlib import Path
import numpy as np

# 设置路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import bts


def test_single_angle(tilt_deg: float, verbose: bool = False) -> dict:
    """测试单个倾斜角度
    
    返回:
        包含测试结果的字典，如果失败则返回 None
    """
    try:
        # 创建光学系统
        system = bts.OpticalSystem(f"Tilted Mirror {tilt_deg}°")
        
        # 添加倾斜平面镜
        # 🚫 禁止设置 semi_aperture 参数
        system.add_flat_mirror(
            z=50.0,
            tilt_x=tilt_deg,  # 角度，单位度
        )
        
        # 定义光源参数（近场条件）
        # 注意：不指定 physical_size_mm，让其默认为 4 × w0 = 20 mm（PROPER 固定用法）
        source = bts.GaussianSource(
            wavelength_um=0.633,
            w0_mm=5.0,
            grid_size=256,
        )
        
        # 运行仿真
        result = bts.simulate(system, source, verbose=verbose)
        
        if not result.success:
            return {
                'angle_deg': tilt_deg,
                'success': False,
                'error': result.error_message if hasattr(result, 'error_message') else 'Simulation failed',
                'rms_milli_waves': None,
                'pv_waves': None,
                'exit_amplitude_max': None,
                'exit_amplitude_center': None,
            }
        
        # 获取出射面数据
        exit_wf = None
        for surface in result.surfaces:
            if surface.exit is not None:
                exit_wf = surface.exit
                break
        
        if exit_wf is None:
            return {
                'angle_deg': tilt_deg,
                'success': False,
                'error': 'No exit state found',
                'rms_milli_waves': None,
                'pv_waves': None,
                'exit_amplitude_max': None,
                'exit_amplitude_center': None,
            }
        
        # 获取振幅和相位
        amplitude = exit_wf.amplitude
        n = amplitude.shape[0]
        
        # 检查振幅是否有效
        if np.max(amplitude) < 1e-10:
            return {
                'angle_deg': tilt_deg,
                'success': False,
                'error': 'Zero amplitude at exit',
                'rms_milli_waves': None,
                'pv_waves': None,
                'exit_amplitude_max': np.max(amplitude),
                'exit_amplitude_center': amplitude[n//2, n//2],
            }
        
        # 使用内置方法计算残差
        rms_waves = exit_wf.get_residual_rms_waves()
        pv_waves = exit_wf.get_residual_pv_waves()
        
        return {
            'angle_deg': tilt_deg,
            'success': True,
            'error': None,
            'rms_milli_waves': rms_waves * 1000,
            'pv_waves': pv_waves,
            'exit_amplitude_max': np.max(amplitude),
            'exit_amplitude_center': amplitude[n//2, n//2],
        }
        
    except Exception as e:
        return {
            'angle_deg': tilt_deg,
            'success': False,
            'error': str(e),
            'rms_milli_waves': None,
            'pv_waves': None,
            'exit_amplitude_max': None,
            'exit_amplitude_center': None,
        }


def main():
    print("=" * 70)
    print("倾斜角度对仿真精度影响的系统性分析")
    print("=" * 70)
    print()
    
    # 测试角度列表
    # 包含常见角度和一些特殊角度
    test_angles = [
        0.0,      # 正入射
        5.0,      # 小角度
        10.0,
        15.0,
        20.0,
        22.5,     # 特殊角度（之前报告有问题）
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,     # 45° 折叠
        50.0,
        55.0,
        60.0,     # 大角度
    ]
    
    results = []
    
    print(f"测试 {len(test_angles)} 个角度...")
    print("-" * 70)
    
    for angle in test_angles:
        print(f"测试角度: {angle:5.1f}° ... ", end="", flush=True)
        result = test_single_angle(angle)
        results.append(result)
        
        if result['success']:
            print(f"成功  RMS={result['rms_milli_waves']:.3f} milli-waves, "
                  f"PV={result['pv_waves']:.4f} waves")
        else:
            print(f"失败  错误: {result['error']}")
    
    print()
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print()

    # 打印表格
    print(f"{'角度 (°)':>10} | {'状态':>6} | {'RMS (milli-waves)':>18} | "
          f"{'PV (waves)':>12} | {'振幅最大值':>12}")
    print("-" * 70)
    
    for r in results:
        if r['success']:
            print(f"{r['angle_deg']:>10.1f} | {'成功':>6} | "
                  f"{r['rms_milli_waves']:>18.3f} | "
                  f"{r['pv_waves']:>12.4f} | "
                  f"{r['exit_amplitude_max']:>12.6f}")
        else:
            print(f"{r['angle_deg']:>10.1f} | {'失败':>6} | "
                  f"{'N/A':>18} | {'N/A':>12} | "
                  f"{r['exit_amplitude_max'] if r['exit_amplitude_max'] else 'N/A':>12}")
    
    print()
    
    # 统计分析
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("=" * 70)
    print("统计分析")
    print("=" * 70)
    print()
    
    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    
    if failed:
        print()
        print("失败的角度:")
        for r in failed:
            print(f"  {r['angle_deg']:.1f}°: {r['error']}")
    
    if successful:
        rms_values = [r['rms_milli_waves'] for r in successful]
        pv_values = [r['pv_waves'] for r in successful]
        
        print()
        print("成功测试的统计:")
        print(f"  RMS 范围: {min(rms_values):.3f} - {max(rms_values):.3f} milli-waves")
        print(f"  RMS 平均: {np.mean(rms_values):.3f} milli-waves")
        print(f"  PV 范围:  {min(pv_values):.4f} - {max(pv_values):.4f} waves")
        print(f"  PV 平均:  {np.mean(pv_values):.4f} waves")
        
        # 找出精度最差的角度
        worst_idx = np.argmax(rms_values)
        worst_result = successful[worst_idx]
        print()
        print(f"精度最差的角度: {worst_result['angle_deg']:.1f}° "
              f"(RMS={worst_result['rms_milli_waves']:.3f} milli-waves)")
        
        # 找出精度最好的角度
        best_idx = np.argmin(rms_values)
        best_result = successful[best_idx]
        print(f"精度最好的角度: {best_result['angle_deg']:.1f}° "
              f"(RMS={best_result['rms_milli_waves']:.3f} milli-waves)")
    
    # 绘制结果图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. RMS vs 角度
        ax1 = axes[0, 0]
        angles_success = [r['angle_deg'] for r in successful]
        rms_success = [r['rms_milli_waves'] for r in successful]
        ax1.plot(angles_success, rms_success, 'bo-', markersize=8)
        ax1.set_xlabel('Tilt Angle (degrees)')
        ax1.set_ylabel('Phase RMS (milli-waves)')
        ax1.set_title('Phase Error RMS vs Tilt Angle')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='r', linestyle='--', label='1 milli-wave threshold')
        ax1.legend()
        
        # 标记失败的角度
        for r in failed:
            ax1.axvline(x=r['angle_deg'], color='r', linestyle=':', alpha=0.5)
        
        # 2. PV vs 角度
        ax2 = axes[0, 1]
        pv_success = [r['pv_waves'] for r in successful]
        ax2.plot(angles_success, pv_success, 'go-', markersize=8)
        ax2.set_xlabel('Tilt Angle (degrees)')
        ax2.set_ylabel('Phase PV (waves)')
        ax2.set_title('Phase Error PV vs Tilt Angle')
        ax2.grid(True, alpha=0.3)
        
        # 3. 振幅最大值 vs 角度
        ax3 = axes[1, 0]
        amp_success = [r['exit_amplitude_max'] for r in successful]
        ax3.plot(angles_success, amp_success, 'mo-', markersize=8)
        ax3.set_xlabel('Tilt Angle (degrees)')
        ax3.set_ylabel('Exit Amplitude Max')
        ax3.set_title('Exit Amplitude vs Tilt Angle')
        ax3.grid(True, alpha=0.3)
        
        # 4. 成功/失败状态
        ax4 = axes[1, 1]
        all_angles = [r['angle_deg'] for r in results]
        status = [1 if r['success'] else 0 for r in results]
        colors = ['green' if s else 'red' for s in status]
        ax4.bar(all_angles, [1]*len(all_angles), color=colors, width=2)
        ax4.set_xlabel('Tilt Angle (degrees)')
        ax4.set_ylabel('Status')
        ax4.set_title('Success/Failure Status (Green=Success, Red=Failure)')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Fail', 'Success'])
        
        plt.tight_layout()
        
        output_path = project_root / 'output' / 'tilt_angle_accuracy_analysis.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print()
        print(f"分析图已保存: {output_path}")
        
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print()
    print("=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
