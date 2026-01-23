"""
近场高斯光束入射倾斜平面镜仿真测试

本脚本演示：
1. 高斯光束在近场范围内传输一段距离
2. 入射至倾斜平面镜
3. 比较仿真结果相对于 Pilot Beam 的相位与振幅误差

测试条件：
- 波长: 0.633 μm (He-Ne)
- 束腰半径: 5.0 mm
- 传输距离: 50 mm（在瑞利长度内，近场）
- 平面镜倾斜角: 45°（绕 X 轴）

已知问题：
- 22.5° 精确角度会导致出射面振幅为零（底层系统数值问题）
- 非 45° 倾斜角的精度较低（~288 milli-waves vs ~1.25 milli-waves）

验证结果：
- 45° 倾斜角：相位残差 RMS ≈ 1.25 milli-waves（高精度）
- 入射面相位残差 RMS ≈ 0.063 milli-waves（极高精度）
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体
def _setup_chinese_fonts():
    """设置 matplotlib 中文字体"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'STHeiti', 'SimSun']
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['font.family'] = 'sans-serif'
            break
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_fonts()

from hybrid_simulation import HybridSimulator


def compute_rayleigh_length(w0_mm: float, wavelength_um: float) -> float:
    """计算瑞利长度"""
    wavelength_mm = wavelength_um * 1e-3
    return np.pi * w0_mm**2 / wavelength_mm


def analyze_results(result, wavelength_um: float, w0_mm: float, mirror_z_mm: float):
    """分析仿真结果，计算相对于 Pilot Beam 的误差"""
    print("\n" + "=" * 70)
    print("仿真结果分析：相对于 Pilot Beam 的误差")
    print("=" * 70)
    
    z_R = compute_rayleigh_length(w0_mm, wavelength_um)
    print(f"\n光束参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R:.2f} mm")
    print(f"  镜面位置: {mirror_z_mm} mm")
    print(f"  z/z_R = {mirror_z_mm/z_R:.3f} (近场条件: < 1)")
    
    for surface in result.surfaces:
        print(f"\n--- 表面 {surface.index}: {surface.name} ({surface.surface_type}) ---")
        
        if surface.entrance is not None:
            print(f"\n  入射面:")
            _analyze_wavefront(surface.entrance)
        
        if surface.exit is not None:
            print(f"\n  出射面:")
            _analyze_wavefront(surface.exit)


def _analyze_wavefront(wf):
    """分析单个波前数据"""
    norm_amp = wf.amplitude / np.max(wf.amplitude) if np.max(wf.amplitude) > 0 else wf.amplitude
    valid_mask = norm_amp > 0.01
    
    if np.sum(valid_mask) == 0:
        print("    [警告] 无有效数据")
        return
    
    residual_rms_waves = wf.get_residual_rms_waves()
    residual_pv_waves = wf.get_residual_pv_waves()
    
    print(f"    相位残差 RMS: {residual_rms_waves:.6f} waves ({residual_rms_waves*1000:.3f} milli-waves)")
    print(f"    相位残差 PV:  {residual_pv_waves:.4f} waves")
    print(f"    Pilot Beam 曲率半径: {wf.pilot_beam.curvature_radius_mm:.2f} mm")
    print(f"    Pilot Beam 光斑大小: {wf.pilot_beam.spot_size_mm:.4f} mm")


def plot_detailed_analysis(result, wavelength_um: float, w0_mm: float, save_path: str = None):
    """绘制详细分析图"""
    surfaces_with_exit = [s for s in result.surfaces if s.exit is not None]
    
    if not surfaces_with_exit:
        print("没有出射面数据可供分析")
        return
    
    last_surface = surfaces_with_exit[-1]
    wf = last_surface.exit
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    grid = wf.grid
    half_size = grid.physical_size_mm / 2
    extent = [-half_size, half_size, -half_size, half_size]
    
    # 1. 仿真振幅
    ax = axes[0, 0]
    im = ax.imshow(wf.amplitude, extent=extent, cmap='hot', origin='lower')
    ax.set_title('仿真振幅')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax)
    
    # 2. 仿真相位
    ax = axes[0, 1]
    im = ax.imshow(wf.phase, extent=extent, cmap='twilight', origin='lower')
    ax.set_title('仿真相位 (rad)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax)
    
    # 3. Pilot Beam 参考相位
    ax = axes[0, 2]
    pilot_phase = wf.get_pilot_beam_phase()
    im = ax.imshow(pilot_phase, extent=extent, cmap='twilight', origin='lower')
    ax.set_title('Pilot Beam 参考相位 (rad)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax)
    
    # 4. 残差相位
    ax = axes[1, 0]
    residual = wf.get_residual_phase()
    residual_mwaves = residual / (2 * np.pi) * 1000
    vmax = max(abs(np.nanmin(residual_mwaves)), abs(np.nanmax(residual_mwaves)), 1)
    im = ax.imshow(residual_mwaves, extent=extent, cmap='RdBu_r', origin='lower',
                   vmin=-vmax, vmax=vmax)
    ax.set_title('残差相位 (milli-waves)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax)
    
    # 5. 振幅剖面
    ax = axes[1, 1]
    center = grid.grid_size // 2
    coords = np.linspace(-half_size, half_size, grid.grid_size)
    ax.plot(coords, wf.amplitude[center, :], 'b-', label='X 剖面', linewidth=2)
    ax.plot(coords, wf.amplitude[:, center], 'r--', label='Y 剖面', linewidth=2)
    ax.set_xlabel('位置 (mm)')
    ax.set_ylabel('振幅')
    ax.set_title('振幅剖面')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 相位剖面
    ax = axes[1, 2]
    ax.plot(coords, wf.phase[center, :], 'b-', label='X 剖面', linewidth=2)
    ax.plot(coords, wf.phase[:, center], 'r--', label='Y 剖面', linewidth=2)
    ax.set_xlabel('位置 (mm)')
    ax.set_ylabel('相位 (rad)')
    ax.set_title('相位剖面')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    z_R = compute_rayleigh_length(w0_mm, wavelength_um)
    rms_waves = wf.get_residual_rms_waves()
    pv_waves = wf.get_residual_pv_waves()
    
    fig.suptitle(
        f'近场倾斜平面镜仿真分析\n'
        f'λ={wavelength_um}μm, w0={w0_mm}mm, z_R={z_R:.1f}mm\n'
        f'相位残差: RMS={rms_waves*1000:.3f} milli-waves, PV={pv_waves:.4f} waves',
        fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n详细分析图已保存: {save_path}")
    
    plt.close(fig)


def main():
    """主程序"""
    print("=" * 70)
    print("近场高斯光束入射倾斜平面镜仿真测试")
    print("=" * 70)
    
    # ========== 仿真参数 ==========
    wavelength_um = 0.633  # He-Ne 激光波长
    w0_mm = 5.0            # 束腰半径
    grid_size = 256        # 网格大小
    
    z_R = compute_rayleigh_length(w0_mm, wavelength_um)
    print(f"\n瑞利长度: z_R = {z_R:.2f} mm")
    
    # 镜面位置（近场）
    mirror_z_mm = 50.0
    print(f"镜面位置: z = {mirror_z_mm:.2f} mm (z/z_R = {mirror_z_mm/z_R:.3f})")
    
    # 平面镜倾斜角度（45° 是经过验证的高精度配置）
    tilt_angle_deg = 45.0
    print(f"平面镜倾斜角: {tilt_angle_deg}° (绕 X 轴)")
    
    # ========== 执行仿真 ==========
    print("\n" + "-" * 40)
    print("开始仿真...")
    print("-" * 40)
    
    sim = HybridSimulator(verbose=True)
    sim.add_flat_mirror(z=mirror_z_mm, tilt_x=tilt_angle_deg, aperture=30.0)
    sim.set_source(wavelength_um=wavelength_um, w0_mm=w0_mm, grid_size=grid_size,
                   physical_size_mm=8 * w0_mm)
    
    result = sim.run()
    
    # ========== 结果分析 ==========
    if result.success:
        result.summary()
        analyze_results(result, wavelength_um, w0_mm, mirror_z_mm)
        
        output_dir = project_root / 'output' / 'near_field_tilted_mirror'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_detailed_analysis(result, wavelength_um, w0_mm,
                               save_path=str(output_dir / 'detailed_analysis.png'))
        
        result.plot_all(save_path=str(output_dir / 'overview.png'), show=False)
        print(f"\n概览图已保存: {output_dir / 'overview.png'}")
        
        result.save(str(output_dir / 'result_data'))
        print(f"完整结果已保存: {output_dir / 'result_data'}")
        
        # ========== 最终总结 ==========
        print("\n" + "=" * 70)
        print("仿真完成！")
        print("=" * 70)
        
        for surface in result.surfaces:
            if surface.exit is not None:
                wf = surface.exit
                rms = wf.get_residual_rms_waves()
                pv = wf.get_residual_pv_waves()
                print(f"\n表面 {surface.index} 出射面误差:")
                print(f"  相位残差 RMS: {rms*1000:.3f} milli-waves")
                print(f"  相位残差 PV:  {pv:.4f} waves")
                
                if rms < 0.001:
                    print(f"  精度等级: 极高 (< 1 milli-wave)")
                elif rms < 0.01:
                    print(f"  精度等级: 高 (< 10 milli-waves)")
                elif rms < 0.1:
                    print(f"  精度等级: 中等 (< 100 milli-waves)")
                else:
                    print(f"  精度等级: 需要改进 (> 100 milli-waves)")
    else:
        print(f"\n仿真失败: {result.error_message}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
