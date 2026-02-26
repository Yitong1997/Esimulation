"""
可视化模块

提供仿真结果的可视化功能，包括 2D 和 3D 绘图。
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# 配置中文字体支持
def _setup_chinese_fonts():
    """设置 matplotlib 中文字体"""
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'Noto Sans SC',     # Google Noto 简体中文
        'STHeiti',          # 华文黑体
        'SimSun',           # 宋体
        'FangSong',         # 仿宋
    ]
    
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_fonts()

if TYPE_CHECKING:
    from .result import SimulationResult, SurfaceRecord, WavefrontData


def _get_valid_mask(amplitude: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """获取有效区域掩模
    
    参数:
        amplitude: 振幅数组
        threshold: 相对阈值（相对于最大值）
    
    返回:
        布尔掩模数组
    """
    max_amp = np.max(amplitude)
    if max_amp > 0:
        norm_amp = amplitude / max_amp
    else:
        norm_amp = amplitude
    return norm_amp > threshold


def _compute_pilot_beam_amplitude(
    wavefront: "WavefrontData",
) -> np.ndarray:
    """计算 Pilot Beam 参考振幅（高斯分布）
    
    参数:
        wavefront: 波前数据
    
    返回:
        Pilot Beam 振幅网格
    """
    n = wavefront.grid.grid_size
    half_size = wavefront.grid.physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    # 使用 Pilot Beam 的光斑大小
    w = wavefront.pilot_beam.spot_size_mm
    
    # 高斯振幅分布
    pilot_amplitude = np.exp(-r_sq / w**2)
    
    # 归一化到与实际振幅相同的峰值
    max_actual = np.max(wavefront.amplitude)
    if max_actual > 0:
        pilot_amplitude = pilot_amplitude * max_actual
    
    return pilot_amplitude


def _compute_residual_amplitude(
    wavefront: "WavefrontData",
) -> np.ndarray:
    """计算振幅残差（实际振幅 - Pilot Beam 振幅）
    
    参数:
        wavefront: 波前数据
    
    返回:
        振幅残差网格
    """
    pilot_amplitude = _compute_pilot_beam_amplitude(wavefront)
    return wavefront.amplitude - pilot_amplitude



def plot_wavefront(
    wavefront: "WavefrontData",
    title: str = "Wavefront Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """绘制波前详细分析图（振幅、相位、Pilot Beam、残差、截面）
    
    采用了 2x3 的布局，包含横截面比较，提供更深入的物理洞察。
    
    布局:
        Row 1: 振幅, Pilot Beam 振幅, 振幅截面比较 (y=0)
        Row 2: 相位, Pilot Beam 相位, 相位残差
    
    参数:
        wavefront: WavefrontData 对象
        title: 图表标题
        save_path: 保存路径（可选）
        show: 是否显示图表
    
    返回:
        matplotlib Figure 对象（如果 show=False）
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # 准备数据
    grid_size = wavefront.grid.grid_size
    physical_size = wavefront.grid.physical_size_mm
    half_size = physical_size / 2
    
    # 1. 仿真波前
    sim_amp = wavefront.amplitude
    sim_phase = wavefront.phase
    
    # 2. Pilot Beam 理论值
    pilot_amp = _compute_pilot_beam_amplitude(wavefront)
    pilot_phase = wavefront.get_pilot_beam_phase()
    
    # 3. 相位残差
    phase_residual = wavefront.get_residual_phase()
    
    # 计算截取范围 (80%) - 聚焦中心细节
    margin = int(grid_size * 0.1)
    sl = slice(margin, grid_size - margin)
    
    # 更新 extent
    crop_half_size = half_size * 0.8
    extent = [-crop_half_size, crop_half_size, -crop_half_size, crop_half_size]
    
    # 绘图配置 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Info String
    pilot_info = (
        f"Pilot Beam Params: "
        f"w(z)={wavefront.pilot_beam.spot_size_mm:.4f}mm, "
        f"R(z)={wavefront.pilot_beam.curvature_radius_mm:.4e}mm, "
        f"z_waist={wavefront.pilot_beam.waist_position_mm:.4f}mm"
    )
    
    fig.suptitle(f"{title}\n{pilot_info}", fontsize=12)
    
    # Helper for colorbar
    def plot_im(ax, data, title, cmap='viridis', has_cbar=True):
        im = ax.imshow(data[sl, sl], extent=extent, origin='lower', cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        if has_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        return im

    # --- Row 1: Amplitude ---
    
    # 1.1 Simulation Amplitude
    max_val = np.max(sim_amp)
    if max_val > 0:
        intensity = sim_amp**2
        total_power = np.sum(intensity)
        dx = wavefront.grid.sampling_mm
        w_eff = np.sqrt(2 * total_power * dx**2 / (np.pi * np.max(intensity)))
        title_amp = f"Sim Amplitude\nCalc w ≈ {w_eff:.4f} mm\n(Max={max_val:.2e})"
    else:
        title_amp = "Sim Amplitude (Zero)"
    
    plot_im(axes[0, 0], sim_amp, title_amp, cmap='inferno')
    
    # 1.2 Pilot Beam Amplitude
    plot_im(axes[0, 1], pilot_amp, "Pilot Beam Amplitude\n(Theoretical Gaussian)", cmap='inferno')
    
    # 1.3 Cross Section Comparison
    ax_slice = axes[0, 2]
    y_mid = grid_size // 2
    x_axis = np.linspace(-half_size, half_size, grid_size)
    x_crop = x_axis[sl]
    
    # Normalize Sim Amp for comparison if max > 0
    sim_slice = sim_amp[y_mid, sl]
    if max_val > 0:
        sim_slice_norm = sim_slice / max_val
        ax_slice.plot(x_crop, sim_slice_norm, label='Sim (Norm)', color='blue')
    else:
        ax_slice.plot(x_crop, sim_slice, label='Sim', color='blue')
        
    ax_slice.plot(x_crop, pilot_amp[y_mid, sl], '--', label='Pilot (Ref)', color='orange')
    ax_slice.set_title("Amplitude Cross-section (y=0)", fontsize=10)
    ax_slice.set_xlabel('x (mm)')
    ax_slice.legend()
    ax_slice.grid(True, alpha=0.3)
    ax_slice.set_xlim(extent[0], extent[1])
    
    # --- Row 2: Phase ---
    
    # 2.1 Simulation Phase (Unwrapped raw phase)
    # Note: Phase is already unwrapped in WavefrontData
    phase_waves = sim_phase / (2 * np.pi)
    plot_im(axes[1, 0], phase_waves, "Simulation Phase\n(Unwrapped, waves)", cmap='RdBu')
    
    # 2.2 Pilot Phase
    pilot_phase_waves = pilot_phase / (2 * np.pi)
    plot_im(axes[1, 1], pilot_phase_waves, "Pilot Beam Phase\n(Analytic, waves)", cmap='RdBu')
    
    # 2.3 Phase Residual
    res_crop = phase_residual[sl, sl]
    rms = np.std(res_crop)
    pv = np.max(res_crop) - np.min(res_crop)
    
    # Convert to waves
    res_crop_waves = res_crop / (2 * np.pi)
    rms_waves = rms / (2 * np.pi)
    pv_waves = pv / (2 * np.pi)
    
    plot_im(axes[1, 2], phase_residual / (2 * np.pi), 
            f"Phase Residual\n(Sim - Pilot)\nRMS={rms_waves:.4f} waves, PV={pv_waves:.4f} waves", 
            cmap='RdBu_r')
    
    # Annotations
    fig.text(0.5, 0.02, 
                f"Wavelength: {wavefront.wavelength_um} um", 
                ha='center', fontsize=11, bbox=dict(facecolor='#f0f0f0', alpha=0.9, pad=5))
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def plot_surface_detail(
    surface: "SurfaceRecord",
    wavelength_um: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """绘制单个表面的详细图表（2D）
    
    使用 plot_wavefront 进行绘制。
    
    参数:
        surface: 表面记录
        wavelength_um: 波长 (μm)
        save_path: 保存路径（可选）
        show: 是否显示图表
    """
    wavefront = surface.exit if surface.exit is not None else surface.entrance
    if wavefront is None:
        print(f"表面 {surface.index} 无波前数据")
        return None
    
    title = f"Surface {surface.index}: {surface.name} ({surface.surface_type})"
    return plot_wavefront(wavefront, title, save_path, show)



def plot_surface_3d(
    surface: "SurfaceRecord",
    wavelength_um: float,
    plot_type: str = "phase",
    save_path: Optional[str] = None,
    show: bool = True,
    elevation: float = 30,
    azimuth: float = -60,
) -> Optional[plt.Figure]:
    """绘制单个表面的 3D 图表
    
    参数:
        surface: 表面记录
        wavelength_um: 波长 (μm)
        plot_type: 绘图类型，可选：
            - "amplitude": 振幅
            - "phase": 相位
            - "residual_phase": 残差相位
            - "pilot_amplitude": Pilot Beam 振幅
            - "residual_amplitude": 振幅残差
        save_path: 保存路径（可选）
        show: 是否显示图表
        elevation: 3D 视角仰角（度）
        azimuth: 3D 视角方位角（度）
    
    返回:
        matplotlib Figure 对象（如果 show=False）
    """
    wavefront = surface.exit if surface.exit is not None else surface.entrance
    if wavefront is None:
        print(f"表面 {surface.index} 无波前数据")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建坐标网格
    n = wavefront.grid.grid_size
    half_size = wavefront.grid.physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    
    valid_mask = _get_valid_mask(wavefront.amplitude)
    
    # 根据类型选择数据
    if plot_type == "amplitude":
        Z = wavefront.amplitude.copy()
        title = "振幅"
        cmap = 'hot'
        zlabel = 'Amplitude'
    elif plot_type == "phase":
        Z = wavefront.phase / (2 * np.pi)
        title = "相位"
        cmap = 'twilight'
        zlabel = 'Phase (waves)'
    elif plot_type == "residual_phase":
        residual = wavefront.get_residual_phase()
        Z = residual / (2 * np.pi)
        Z = np.where(valid_mask, Z, np.nan)
        rms = wavefront.get_residual_rms_waves()
        title = f"残差相位 (RMS={rms:.4f} waves)"
        cmap = 'RdBu_r'
        zlabel = 'Residual (waves)'
    elif plot_type == "pilot_amplitude":
        Z = _compute_pilot_beam_amplitude(wavefront)
        title = f"Pilot Beam 振幅 (w={wavefront.pilot_beam.spot_size_mm:.2f}mm)"
        cmap = 'hot'
        zlabel = 'Amplitude'
    elif plot_type == "residual_amplitude":
        Z = _compute_residual_amplitude(wavefront)
        Z = np.where(valid_mask, Z, np.nan)
        title = "振幅残差"
        cmap = 'RdBu_r'
        zlabel = 'Residual'
    else:
        raise ValueError(f"未知的绘图类型: {plot_type}")
    
    # 降采样以提高绘图性能
    stride = max(1, n // 100)
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]
    Z_ds = Z[::stride, ::stride]
    
    # 绘制 3D 表面
    surf = ax.plot_surface(
        X_ds, Y_ds, Z_ds,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel(zlabel)
    ax.set_title(f"Surface {surface.index}: {surface.name} - {title}")
    ax.view_init(elev=elevation, azim=azimuth)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def plot_surface_detail_3d(
    surface: "SurfaceRecord",
    wavelength_um: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """绘制单个表面的完整 3D 详细图表
    
    包含 6 个子图：振幅、相位、Pilot Beam 相位、残差相位、
                  Pilot Beam 振幅、振幅残差
    
    参数:
        surface: 表面记录
        wavelength_um: 波长 (μm)
        save_path: 保存路径（可选）
        show: 是否显示图表
    
    返回:
        matplotlib Figure 对象（如果 show=False）
    """
    wavefront = surface.exit if surface.exit is not None else surface.entrance
    if wavefront is None:
        print(f"表面 {surface.index} 无波前数据")
        return None
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Surface {surface.index}: {surface.name} ({surface.surface_type}) - 3D 视图",
        fontsize=14,
    )
    
    # 创建坐标网格
    n = wavefront.grid.grid_size
    half_size = wavefront.grid.physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    
    valid_mask = _get_valid_mask(wavefront.amplitude)
    
    # 降采样
    stride = max(1, n // 80)
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]
    
    plot_configs = [
        ("amplitude", wavefront.amplitude, "振幅", "hot", "Amplitude"),
        ("phase", wavefront.phase / (2 * np.pi), "相位", "twilight", "Phase (waves)"),
        ("pilot_phase", wavefront.get_pilot_beam_phase() / (2 * np.pi), 
         f"Pilot Beam 相位 (R={wavefront.pilot_beam.curvature_radius_mm:.1f}mm)", 
         "twilight", "Phase (waves)"),
        ("residual_phase", 
         np.where(valid_mask, wavefront.get_residual_phase() / (2 * np.pi), np.nan),
         f"残差相位 (RMS={wavefront.get_residual_rms_waves():.4f} waves)", 
         "RdBu_r", "Residual (waves)"),
        ("pilot_amplitude", _compute_pilot_beam_amplitude(wavefront),
         f"Pilot Beam 振幅 (w={wavefront.pilot_beam.spot_size_mm:.2f}mm)",
         "hot", "Amplitude"),
        ("residual_amplitude", 
         np.where(valid_mask, _compute_residual_amplitude(wavefront), np.nan),
         "振幅残差", "RdBu_r", "Residual"),
    ]
    
    for i, (name, data, title, cmap, zlabel) in enumerate(plot_configs):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        
        Z_ds = data[::stride, ::stride]
        
        surf = ax.plot_surface(
            X_ds, Y_ds, Z_ds,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=30, azim=-60)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def plot_all_surfaces(
    result: "SimulationResult",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """绘制所有表面的概览图（2D）
    
    每个表面显示振幅和残差相位。
    
    参数:
        result: 仿真结果
        save_path: 保存路径（可选）
        show: 是否显示图表
    
    返回:
        matplotlib Figure 对象（如果 show=False）
    """
    surfaces_with_data = [
        s for s in result.surfaces
        if s.exit is not None or s.entrance is not None
    ]
    
    n_surfaces = len(surfaces_with_data)
    if n_surfaces == 0:
        print("无波前数据可绘制")
        return None
    
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
        
        valid_mask = _get_valid_mask(wavefront.amplitude)
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
        return None
    else:
        plt.close()
        return fig


def plot_all_surfaces_extended(
    result: "SimulationResult",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """绘制所有表面的扩展概览图（2D）
    
    每个表面显示：振幅、残差相位、Pilot Beam 振幅、振幅残差
    
    参数:
        result: 仿真结果
        save_path: 保存路径（可选）
        show: 是否显示图表
    
    返回:
        matplotlib Figure 对象（如果 show=False）
    """
    surfaces_with_data = [
        s for s in result.surfaces
        if s.exit is not None or s.entrance is not None
    ]
    
    n_surfaces = len(surfaces_with_data)
    if n_surfaces == 0:
        print("无波前数据可绘制")
        return None
    
    fig, axes = plt.subplots(n_surfaces, 4, figsize=(16, 3.5 * n_surfaces))
    fig.suptitle('混合光学仿真结果扩展概览', fontsize=14)
    
    if n_surfaces == 1:
        axes = axes.reshape(1, -1)
    
    for i, surface in enumerate(surfaces_with_data):
        wavefront = surface.exit if surface.exit is not None else surface.entrance
        
        half_size = wavefront.grid.physical_size_mm / 2
        extent = [-half_size, half_size, -half_size, half_size]
        valid_mask = _get_valid_mask(wavefront.amplitude)
        
        # 1. 振幅
        ax = axes[i, 0]
        im = ax.imshow(wavefront.amplitude, extent=extent, origin='lower', cmap='hot')
        ax.set_title(f'[{surface.index}] {surface.name} - 振幅')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)
        
        # 2. 残差相位
        ax = axes[i, 1]
        residual = wavefront.get_residual_phase() / (2 * np.pi)
        residual_masked = np.where(valid_mask, residual, np.nan)
        im = ax.imshow(residual_masked, extent=extent, origin='lower', cmap='RdBu_r')
        rms = wavefront.get_residual_rms_waves()
        ax.set_title(f'残差相位 (RMS={rms:.4f} waves)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='waves')
        
        # 3. Pilot Beam 振幅
        ax = axes[i, 2]
        pilot_amp = _compute_pilot_beam_amplitude(wavefront)
        im = ax.imshow(pilot_amp, extent=extent, origin='lower', cmap='hot')
        ax.set_title(f'Pilot Beam 振幅 (w={wavefront.pilot_beam.spot_size_mm:.2f}mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)
        
        # 4. 振幅残差
        ax = axes[i, 3]
        amp_residual = _compute_residual_amplitude(wavefront)
        amp_residual_masked = np.where(valid_mask, amp_residual, np.nan)
        im = ax.imshow(amp_residual_masked, extent=extent, origin='lower', cmap='RdBu_r')
        if np.any(valid_mask):
            amp_rms = np.sqrt(np.mean(amp_residual[valid_mask]**2))
        else:
            amp_rms = np.nan
        ax.set_title(f'振幅残差 (RMS={amp_rms:.4f})')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig
