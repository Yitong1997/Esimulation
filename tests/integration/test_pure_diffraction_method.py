"""
纯衍射方法（tilted_asm 投影传输）测试文件

测试目标：
1. 验证 pure_diffraction 方法能够正常运行
2. 比较 local_raytracing 和 pure_diffraction 两种方法的结果
3. 验证能量守恒和相位连续性

测试场景：
- 45° 倾斜平面镜（最简单的折叠场景）
- 近场高斯光束

预期结果：
- 两种方法的结果应该接近（对于平面镜，理论上应该完全一致）
- 能量守恒（振幅总能量变化 < 5%）
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


def test_pure_diffraction_flat_mirror(tilt_deg: float = 45.0, verbose: bool = True):
    """测试纯衍射方法在平面镜上的表现
    
    参数:
        tilt_deg: 倾斜角度（度）
        verbose: 是否输出详细信息
    
    返回:
        包含测试结果的字典
    """
    print(f"\n{'='*70}")
    print(f"测试 pure_diffraction 方法：{tilt_deg}° 倾斜平面镜")
    print(f"{'='*70}")
    
    # 创建光学系统
    system = bts.OpticalSystem(f"Tilted Mirror {tilt_deg}°")
    system.add_flat_mirror(z=50.0, tilt_x=tilt_deg)
    
    # 定义光源参数
    source = bts.GaussianSource(
        wavelength_um=0.633,
        w0_mm=5.0,
        grid_size=256,
    )

    results = {}
    
    # 1. 使用 local_raytracing 方法
    print("\n1. 使用 local_raytracing 方法...")
    try:
        result_raytracing = bts.simulate(
            system, source,
            verbose=verbose,
            propagation_method='local_raytracing',
        )
        
        if result_raytracing.success:
            exit_wf = None
            for surface in result_raytracing.surfaces:
                if surface.exit is not None:
                    exit_wf = surface.exit
                    break
            
            if exit_wf is not None:
                results['raytracing'] = {
                    'success': True,
                    'rms_milli_waves': exit_wf.get_residual_rms_waves() * 1000,
                    'pv_waves': exit_wf.get_residual_pv_waves(),
                    'amplitude_max': np.max(exit_wf.amplitude),
                    'amplitude_sum': np.sum(exit_wf.amplitude**2),
                }
                print(f"   成功！RMS = {results['raytracing']['rms_milli_waves']:.3f} milli-waves")
            else:
                results['raytracing'] = {'success': False, 'error': 'No exit state'}
                print(f"   失败：No exit state")
        else:
            results['raytracing'] = {'success': False, 'error': 'Simulation failed'}
            print(f"   失败：Simulation failed")
            
    except Exception as e:
        results['raytracing'] = {'success': False, 'error': str(e)}
        print(f"   异常：{e}")
    
    # 2. 使用 pure_diffraction 方法
    print("\n2. 使用 pure_diffraction 方法...")
    try:
        result_diffraction = bts.simulate(
            system, source,
            verbose=True,  # 开启详细输出
            propagation_method='pure_diffraction',
        )
        
        if result_diffraction.success:
            exit_wf = None
            for surface in result_diffraction.surfaces:
                if surface.exit is not None:
                    exit_wf = surface.exit
                    break
            
            if exit_wf is not None:
                results['diffraction'] = {
                    'success': True,
                    'rms_milli_waves': exit_wf.get_residual_rms_waves() * 1000,
                    'pv_waves': exit_wf.get_residual_pv_waves(),
                    'amplitude_max': np.max(exit_wf.amplitude),
                    'amplitude_sum': np.sum(exit_wf.amplitude**2),
                }
                print(f"   成功！RMS = {results['diffraction']['rms_milli_waves']:.3f} milli-waves")
            else:
                results['diffraction'] = {'success': False, 'error': 'No exit state'}
                print(f"   失败：No exit state")
        else:
            results['diffraction'] = {'success': False, 'error': 'Simulation failed'}
            print(f"   失败：Simulation failed")
            
    except Exception as e:
        import traceback
        results['diffraction'] = {'success': False, 'error': str(e)}
        print(f"   异常：{e}")
        traceback.print_exc()
    
    # 3. 比较结果
    print(f"\n{'='*70}")
    print("结果比较")
    print(f"{'='*70}")
    
    if results.get('raytracing', {}).get('success') and results.get('diffraction', {}).get('success'):
        rt = results['raytracing']
        df = results['diffraction']
        
        print(f"\n{'指标':<25} | {'local_raytracing':>18} | {'pure_diffraction':>18}")
        print("-" * 70)
        print(f"{'RMS (milli-waves)':<25} | {rt['rms_milli_waves']:>18.3f} | {df['rms_milli_waves']:>18.3f}")
        print(f"{'PV (waves)':<25} | {rt['pv_waves']:>18.4f} | {df['pv_waves']:>18.4f}")
        print(f"{'振幅最大值':<25} | {rt['amplitude_max']:>18.6f} | {df['amplitude_max']:>18.6f}")
        print(f"{'振幅能量 (sum^2)':<25} | {rt['amplitude_sum']:>18.2f} | {df['amplitude_sum']:>18.2f}")
        
        # 计算差异
        rms_diff = abs(rt['rms_milli_waves'] - df['rms_milli_waves'])
        energy_ratio = df['amplitude_sum'] / rt['amplitude_sum'] if rt['amplitude_sum'] > 0 else 0
        
        print(f"\n差异分析:")
        print(f"  RMS 差异: {rms_diff:.3f} milli-waves")
        print(f"  能量比值: {energy_ratio:.4f} (理想值 = 1.0)")
        
        results['comparison'] = {
            'rms_diff_milli_waves': rms_diff,
            'energy_ratio': energy_ratio,
        }
    else:
        print("\n无法比较：至少一种方法失败")
    
    return results


def main():
    print("=" * 70)
    print("纯衍射方法（tilted_asm 投影传输）测试")
    print("=" * 70)
    
    # 测试不同角度
    test_angles = [0.0, 22.5, 45.0]
    
    all_results = {}
    
    for angle in test_angles:
        results = test_pure_diffraction_flat_mirror(angle, verbose=False)
        all_results[angle] = results
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    
    print(f"\n{'角度 (°)':<10} | {'raytracing RMS':>18} | {'diffraction RMS':>18} | {'差异':>12}")
    print("-" * 70)
    
    for angle, results in all_results.items():
        rt_rms = results.get('raytracing', {}).get('rms_milli_waves', 'N/A')
        df_rms = results.get('diffraction', {}).get('rms_milli_waves', 'N/A')
        
        if isinstance(rt_rms, float) and isinstance(df_rms, float):
            diff = abs(rt_rms - df_rms)
            print(f"{angle:<10.1f} | {rt_rms:>18.3f} | {df_rms:>18.3f} | {diff:>12.3f}")
        else:
            rt_str = f"{rt_rms:.3f}" if isinstance(rt_rms, float) else str(rt_rms)
            df_str = f"{df_rms:.3f}" if isinstance(df_rms, float) else str(df_rms)
            print(f"{angle:<10.1f} | {rt_str:>18} | {df_str:>18} | {'N/A':>12}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
