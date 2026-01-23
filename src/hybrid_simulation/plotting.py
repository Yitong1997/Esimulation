"""
可视化模块

提供仿真结果的可视化功能。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体支持
def _setup_chinese_fonts():
    """设置 matplotlib 中文字体"""
    # 按优先级尝试的中文字体列表
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'Noto Sans SC',     # Google Noto 简体中文
        'STHeiti',          # 华文黑体
        'SimSun',           # 宋体
        'FangSong',         # 仿宋
    ]
    
    # 获取系统可用字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    
    # 修复负号显示
    plt.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
_setup_chinese_fonts()

if TYPE_CHECKING:
    from .result import SimulationResult, SurfaceRecord, WavefrontData


def plot_surface_detail(
    surface: "SurfaceRecord",
    wavelength_um: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """绘制单个表面的详细图表
    
    包含：振幅、相位、Pilot Beam 相位、残差相位
    
    参数:
        surface: 表面记录
        wavelength_um: 波长 (μm)
        save_path: 保存路径（可选）
        show: 是否显示图表
    """
    # 确定要绘制的数据
    wavefront = surface.exit if surface.exit is not None else surface.entrance
    if wavefront is None:
        print(f"表面 {surface.index} 无波前数据")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Surface {surface.index}: {surface.name} ({surface.surface_type})",
        fontsize=14,
    )
    
    # 坐标范围
    half_size = wavefront.grid.physical_size_mm / 2
    extent = [-half_size, half_size, -half_size, half_size]
    
    # 1. 振幅
    ax = axes[0, 0]
    im = ax.imshow(
        wavefront.amplitude,
        extent=extent,
        origin='lower',
        cmap='hot',
    )
    ax.set_title('振幅')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax, label='Amplitude')

    # 2. 相位
    ax = axes[0, 1]
    phase_waves = wavefront.phase / (2 * np.pi)
    im = ax.imshow(
        phase_waves,
        extent=extent,
        origin='lower',
        cmap='twilight',
    )
    ax.set_title('相位')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax, label='Phase (waves)')
    
    # 3. Pilot Beam 相位
    ax = axes[1, 0]
    pilot_phase = wavefront.get_pilot_beam_phase()
    pilot_waves = pilot_phase / (2 * np.pi)
    im = ax.imshow(
        pilot_waves,
        extent=extent,
        origin='lower',
        cmap='twilight',
    )
    ax.set_title(f'Pilot Beam 相位 (R={wavefront.pilot_beam.curvature_radius_mm:.1f}mm)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax, label='Phase (waves)')
    
    # 4. 残差相位
    ax = axes[1, 1]
    residual = wavefront.get_residual_phase()
    residual_waves = residual / (2 * np.pi)
    
    # 有效区域掩模
    norm_amp = wavefront.amplitude / np.max(wavefront.amplitude) if np.max(wavefront.amplitude) > 0 else wavefront.amplitude
    valid_mask = norm_amp > 0.01
    residual_masked = np.where(valid_mask, residual_waves, np.nan)
    
    im = ax.imshow(
        residual_masked,
        extent=extent,
        origin='lower',
        cmap='RdBu_r',
    )
    
    rms = wavefront.get_residual_rms_waves()
    pv = wavefront.get_residual_pv_waves()
    ax.set_title(f'残差相位 (RMS={rms:.4f}, PV={pv:.4f} waves)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(im, ax=ax, label='Residual (waves)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_surfaces(
    result: "SimulationResult",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """绘制所有表面的概览图
    
    每个表面显示振幅和残差相位。
    
    参数:
        result: 仿真结果
        save_path: 保存路径（可选）
        show: 是否显示图表
    """
    # 过滤有波前数据的表面
    surfaces_with_data = [
        s for s in result.surfaces
        if s.exit is not None or s.entrance is not None
    ]
    
    n_surfaces = len(surfaces_with_data)
    if n_surfaces == 0:
        print("无波前数据可绘制")
        return
    
    # 创建图表
    fig, axes = plt.subplots(n_surfaces, 2, figsize=(10, 4 * n_surfaces))
    fig.suptitle('混合光学仿真结果概览', fontsize=14)
    
    if n_surfaces == 1:
        axes = axes.reshape(1, -1)
    
    for i, surface in enumerate(surfaces_with_data):
        wavefront = surface.exit if surface.exit is not None else surface.entrance
        
        half_size = wavefront.grid.physical_size_mm / 2
        extent = [-half_size, half_size, -half_size, half_size]
        
        # 振幅
        ax = axes[i, 0]
        im = ax.imshow(
            wavefront.amplitude,
            extent=extent,
            origin='lower',
            cmap='hot',
        )
        ax.set_title(f'[{surface.index}] {surface.name} - 振幅')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)
        
        # 残差相位
        ax = axes[i, 1]
        residual = wavefront.get_residual_phase()
        residual_waves = residual / (2 * np.pi)
        
        norm_amp = wavefront.amplitude / np.max(wavefront.amplitude) if np.max(wavefront.amplitude) > 0 else wavefront.amplitude
        valid_mask = norm_amp > 0.01
        residual_masked = np.where(valid_mask, residual_waves, np.nan)
        
        im = ax.imshow(
            residual_masked,
            extent=extent,
            origin='lower',
            cmap='RdBu_r',
        )
        
        rms = wavefront.get_residual_rms_waves()
        ax.set_title(f'残差相位 (RMS={rms:.4f} waves)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='waves')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
