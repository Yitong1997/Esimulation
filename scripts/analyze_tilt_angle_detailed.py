"""
详细分析倾斜角度对仿真精度的影响

基于初步分析的发现：
1. 0° 和 45° 精度极高（~0.343 milli-waves）
2. 其他角度精度较低（~288 milli-waves）
3. 22.5° 导致零振幅（失败）
4. 误差呈现关于 22.5° 的对称性

本脚本进一步分析：
1. 更细粒度的角度测试（特别是 0°、22.5°、45° 附近）
2. 分析误差与角度的数学关系
3. 定位问题的物理根源
"""

import sys
from pathlib import Path
import numpy as np

# 设置路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

from hybrid_simulation import HybridSimulator
import warnings
warnings.filterwarnings('ignore')


def test_single_angle(tilt_deg: float) -> dict:
    """测试单个倾斜角度"""
    try:
        sim = HybridSimulator(verbose=False)
        sim.add_flat_mirror(z=50.0, tilt_x=tilt_deg, aperture=30.0)
        sim.set_source(wavelength_um=0.633, w0_mm=5.0, grid_size=256, physical_size_mm=40.0)
        result = sim.run()
        
        if not result.success:
            return {'angle_deg': tilt_deg, 'success': False, 'rms_milli_waves': None}
        
        exit_wf = None
        for surface in result.surfaces:
            if surface.exit is not None:
                exit_wf = surface.exit
                break
        
        if exit_wf is None or np.max(exit_wf.amplitude) < 1e-10:
            return {'angle_deg': tilt_deg, 'success': False, 'rms_milli_waves': None}
        
        rms_waves = exit_wf.get_residual_rms_waves()
        return {'angle_deg': tilt_deg, 'success': True, 'rms_milli_waves': rms_waves * 1000}
        
    except Exception as e:
        return {'angle_deg': tilt_deg, 'success': False, 'rms_milli_waves': None, 'error': str(e)}


def main():
    print("=" * 70)
    print("倾斜角度对仿真精度的详细分析")
    print("=" * 70)
    print()
    
    # 1. 细粒度测试：0° 附近
    print("1. 测试 0° 附近的角度...")
    angles_near_0 = np.arange(0, 6, 0.5)
    results_near_0 = []
    for angle in angles_near_0:
        r = test_single_angle(angle)
        results_near_0.append(r)
        status = f"{r['rms_milli_waves']:.3f}" if r['success'] else "FAIL"
        print(f"  {angle:5.1f}°: {status}")
    
    # 2. 细粒度测试：22.5° 附近
    print("\n2. 测试 22.5° 附近的角度...")
    angles_near_22_5 = np.arange(20, 26, 0.5)
    results_near_22_5 = []
    for angle in angles_near_22_5:
        r = test_single_angle(angle)
        results_near_22_5.append(r)
        status = f"{r['rms_milli_waves']:.3f}" if r['success'] else "FAIL"
        print(f"  {angle:5.1f}°: {status}")
    
    # 3. 细粒度测试：45° 附近
    print("\n3. 测试 45° 附近的角度...")
    angles_near_45 = np.arange(42, 49, 0.5)
    results_near_45 = []
    for angle in angles_near_45:
        r = test_single_angle(angle)
        results_near_45.append(r)
        status = f"{r['rms_milli_waves']:.3f}" if r['success'] else "FAIL"
        print(f"  {angle:5.1f}°: {status}")
    
    # 4. 分析对称性
    print("\n" + "=" * 70)
    print("对称性分析")
    print("=" * 70)
    
    # 测试关于 22.5° 对称的角度对
    symmetric_pairs = [
        (0, 45),
        (5, 40),
        (10, 35),
        (15, 30),
        (20, 25),
    ]
    
    print("\n关于 22.5° 对称的角度对比较:")
    print(f"{'角度1':>8} | {'RMS1':>12} | {'角度2':>8} | {'RMS2':>12} | {'差异':>12}")
    print("-" * 60)
    
    for a1, a2 in symmetric_pairs:
        r1 = test_single_angle(a1)
        r2 = test_single_angle(a2)
        
        rms1 = f"{r1['rms_milli_waves']:.3f}" if r1['success'] else "FAIL"
        rms2 = f"{r2['rms_milli_waves']:.3f}" if r2['success'] else "FAIL"
        
        if r1['success'] and r2['success']:
            diff = abs(r1['rms_milli_waves'] - r2['rms_milli_waves'])
            diff_str = f"{diff:.3f}"
        else:
            diff_str = "N/A"
        
        print(f"{a1:>8.1f}° | {rms1:>12} | {a2:>8.1f}° | {rms2:>12} | {diff_str:>12}")
    
    # 5. 分析误差与角度的关系
    print("\n" + "=" * 70)
    print("误差与角度关系分析")
    print("=" * 70)
    
    # 计算 |sin(2θ)| 和 |cos(2θ)|
    print("\n角度与三角函数值的关系:")
    print(f"{'角度':>8} | {'sin(2θ)':>10} | {'cos(2θ)':>10} | {'|sin(2θ)|':>10} | {'RMS':>12}")
    print("-" * 60)
    
    test_angles = [0, 5, 10, 15, 20, 22.5, 25, 30, 35, 40, 45]
    for angle in test_angles:
        r = test_single_angle(angle)
        theta_rad = np.radians(angle)
        sin_2theta = np.sin(2 * theta_rad)
        cos_2theta = np.cos(2 * theta_rad)
        
        rms_str = f"{r['rms_milli_waves']:.3f}" if r['success'] else "FAIL"
        print(f"{angle:>8.1f}° | {sin_2theta:>10.4f} | {cos_2theta:>10.4f} | {abs(sin_2theta):>10.4f} | {rms_str:>12}")
    
    # 6. 关键发现总结
    print("\n" + "=" * 70)
    print("关键发现总结")
    print("=" * 70)
    
    print("""
1. 高精度角度：
   - 0° (正入射): RMS ≈ 0.343 milli-waves
   - 45° (标准折叠): RMS ≈ 0.343 milli-waves
   - 这两个角度对应 sin(2θ) = 0 或 cos(2θ) = ±1

2. 低精度角度：
   - 所有非 0°/45° 的角度: RMS ≈ 288 milli-waves
   - 误差几乎恒定，与具体角度无关

3. 失败角度：
   - 22.5°: 出射振幅为零
   - 这个角度对应 sin(2θ) = sin(45°) = 0.707, cos(2θ) = 0

4. 对称性：
   - 误差关于 22.5° 对称
   - 例如：5° 和 40° 的误差相同，10° 和 35° 的误差相同

5. 物理解释：
   - 22.5° 是入射角和反射角之和为 45° 的情况
   - 此时入射面和出射面的坐标变换可能存在奇异性
   - 0° 和 45° 是特殊情况，入射/出射面重合或正交

6. 误差来源推测：
   - 非 0°/45° 角度时，入射面和出射面之间存在坐标变换
   - 这个坐标变换在光线追迹或波前重建过程中引入误差
   - 误差大小（~288 milli-waves）与角度无关，说明是系统性问题
""")
    
    # 7. 绘图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 收集所有测试数据
        all_angles = np.arange(0, 61, 1)
        all_results = []
        print("\n正在收集完整数据用于绘图...")
        for angle in all_angles:
            r = test_single_angle(angle)
            all_results.append(r)
        
        angles_success = [r['angle_deg'] for r in all_results if r['success']]
        rms_success = [r['rms_milli_waves'] for r in all_results if r['success']]
        angles_fail = [r['angle_deg'] for r in all_results if not r['success']]
        
        # 1. RMS vs 角度
        ax1 = axes[0, 0]
        ax1.plot(angles_success, rms_success, 'b.-', markersize=4, linewidth=0.5)
        for af in angles_fail:
            ax1.axvline(x=af, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax1.set_xlabel('Tilt Angle (degrees)')
        ax1.set_ylabel('Phase RMS (milli-waves)')
        ax1.set_title('Phase Error RMS vs Tilt Angle')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 2. RMS vs sin(2θ)
        ax2 = axes[0, 1]
        sin_2theta = [np.sin(2 * np.radians(a)) for a in angles_success]
        ax2.scatter(sin_2theta, rms_success, c='blue', s=20, alpha=0.7)
        ax2.set_xlabel('sin(2θ)')
        ax2.set_ylabel('Phase RMS (milli-waves)')
        ax2.set_title('Phase Error RMS vs sin(2θ)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='sin(2θ)=0')
        ax2.legend()
        
        # 3. RMS vs |cos(2θ)|
        ax3 = axes[1, 0]
        cos_2theta = [np.cos(2 * np.radians(a)) for a in angles_success]
        ax3.scatter(cos_2theta, rms_success, c='green', s=20, alpha=0.7)
        ax3.set_xlabel('cos(2θ)')
        ax3.set_ylabel('Phase RMS (milli-waves)')
        ax3.set_title('Phase Error RMS vs cos(2θ)')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='cos(2θ)=1 (θ=0°)')
        ax3.axvline(x=-1, color='r', linestyle='--', alpha=0.5, label='cos(2θ)=-1 (θ=45°)')
        ax3.axvline(x=0, color='orange', linestyle='--', alpha=0.5, label='cos(2θ)=0 (θ=22.5°)')
        ax3.legend(fontsize=8)
        
        # 4. 对称性验证
        ax4 = axes[1, 1]
        # 绘制关于 22.5° 的对称性
        angles_left = [a for a in angles_success if a < 22.5]
        angles_right = [a for a in angles_success if a > 22.5]
        rms_left = [r['rms_milli_waves'] for r in all_results if r['success'] and r['angle_deg'] < 22.5]
        rms_right = [r['rms_milli_waves'] for r in all_results if r['success'] and r['angle_deg'] > 22.5]
        
        # 将右侧角度映射到左侧（关于 22.5° 对称）
        angles_right_mapped = [45 - a for a in angles_right]
        
        ax4.plot(angles_left, rms_left, 'bo-', label='θ < 22.5°', markersize=4)
        ax4.plot(angles_right_mapped, rms_right, 'r^-', label='45° - θ (θ > 22.5°)', markersize=4)
        ax4.set_xlabel('Angle from 0° (or 45° - θ)')
        ax4.set_ylabel('Phase RMS (milli-waves)')
        ax4.set_title('Symmetry about 22.5°')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = project_root / 'output' / 'tilt_angle_detailed_analysis.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\n详细分析图已保存: {output_path}")
        
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
