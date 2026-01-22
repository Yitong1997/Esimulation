"""
ZMX 文件综合精度验证示例

============================================================
示例说明
============================================================

本示例演示如何对任意 ZMX 文件进行完整的端到端精度验证：

1. 载入 ZMX 文件并生成光学系统
2. 定义入射光复振幅（理想高斯光束）
3. 光路绘图（可选）
4. 主光线追迹全部系统（基于 optiland）
5. 完整的物理光学仿真与验证结果记录

验证内容包括：
- 所有折反射面的入射/出射面仿真复振幅（振幅与相位单独绘制）
- 出入面 Pilot Beam 高斯光束参数
- 出入面仿真复振幅相对于 Pilot Beam 的残差（振幅与相位单独绘制）
- 面形参数

精度要求（严格，不得放松）：
- 初始波前：相位 RMS 误差 < 0.001 waves
- 自由空间传播后：相位 RMS 误差 < 0.01 waves
- 经过光学元件后：取决于元件像差

作者：混合光学仿真项目
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# 第一部分：导入模块
# ============================================================================

def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


print_section("第一部分：导入模块")

try:
    from hybrid_optical_propagation import (
        SourceDefinition,
        HybridOpticalPropagator,
        PilotBeamParams,
        GridSampling,
        PropagationState,
        load_optical_system_from_zmx,
        ZmxOpticalSystem,
    )
    from sequential_system.zmx_visualization import (
        ZmxOpticLoader,
        view_2d,
    )
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    print("[OK] 成功导入所有模块")
except ImportError as e:
    print(f"[FAIL] 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# 第二部分：精度分析数据类
# ============================================================================

print_section("第二部分：定义精度分析数据类")


@dataclass
class SurfaceInfo:
    """表面信息"""
    index: int
    surface_type: str
    is_mirror: bool
    radius: float
    conic: float
    vertex_position: np.ndarray
    comment: str
    material: str


@dataclass
class PilotBeamInfo:
    """Pilot Beam 参数信息"""
    wavelength_um: float
    waist_radius_mm: float
    waist_position_mm: float
    curvature_radius_mm: float
    spot_size_mm: float
    rayleigh_length_mm: float


@dataclass
class AccuracyMetrics:
    """精度指标"""
    surface_name: str
    position: str  # 'entrance', 'exit', 'source'
    
    # 振幅误差
    amplitude_rms_error: float
    amplitude_max_error: float
    amplitude_pv_error: float
    
    # 相位误差（相对于 Pilot Beam）
    phase_rms_error_rad: float
    phase_max_error_rad: float
    phase_pv_error_rad: float
    
    # Pilot Beam 参数
    pilot_info: PilotBeamInfo
    
    # 表面信息（如果有）
    surface_info: Optional[SurfaceInfo] = None
    
    @property
    def phase_rms_error_waves(self) -> float:
        return self.phase_rms_error_rad / (2 * np.pi)
    
    @property
    def phase_pv_error_waves(self) -> float:
        return self.phase_pv_error_rad / (2 * np.pi)
    
    @property
    def phase_max_error_waves(self) -> float:
        return self.phase_max_error_rad / (2 * np.pi)


@dataclass
class AnalysisData:
    """分析数据"""
    sim_amplitude: np.ndarray
    sim_phase: np.ndarray
    ideal_amplitude: np.ndarray
    pilot_phase: np.ndarray
    amp_diff: np.ndarray
    phase_diff: np.ndarray
    valid_mask: np.ndarray
    physical_size_mm: float


@dataclass
class ValidationResult:
    """验证结果"""
    zmx_file: str
    timestamp: str
    source_params: Dict[str, Any]
    metrics_list: List[AccuracyMetrics]
    analysis_data_list: List[AnalysisData]
    first_error_surface: Optional[str] = None
    error_message: Optional[str] = None
    passed: bool = True


print("[OK] 精度分析数据类定义完成")


# ============================================================================
# 第三部分：精度分析函数
# ============================================================================

print_section("第三部分：定义精度分析函数")


def extract_pilot_beam_info(pilot_params: PilotBeamParams) -> PilotBeamInfo:
    """提取 Pilot Beam 参数信息"""
    return PilotBeamInfo(
        wavelength_um=pilot_params.wavelength_um,
        waist_radius_mm=pilot_params.waist_radius_mm,
        waist_position_mm=pilot_params.waist_position_mm,
        curvature_radius_mm=pilot_params.curvature_radius_mm,
        spot_size_mm=pilot_params.spot_size_mm,
        rayleigh_length_mm=pilot_params.rayleigh_length_mm,
    )


def extract_surface_info(
    surface: GlobalSurfaceDefinition
) -> SurfaceInfo:
    """提取表面信息"""
    return SurfaceInfo(
        index=surface.index,
        surface_type=surface.surface_type,
        is_mirror=surface.is_mirror,
        radius=surface.radius,
        conic=surface.conic,
        vertex_position=surface.vertex_position.copy(),
        comment=surface.comment,
        material=surface.material,
    )


def compute_ideal_gaussian_amplitude(
    pilot_params: PilotBeamParams,
    grid_size: int,
    physical_size_mm: float,
) -> np.ndarray:
    """计算理想高斯光束振幅分布"""
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    w = pilot_params.spot_size_mm
    amplitude = np.exp(-r_sq / w**2)
    
    return amplitude




def analyze_state_accuracy(
    state: PropagationState,
    grid_size: int,
    physical_size_mm: float,
    surface_name: str,
    surface_info: Optional[SurfaceInfo] = None,
    valid_threshold: float = 0.01,
) -> Tuple[AccuracyMetrics, AnalysisData]:
    """分析传播状态的精度
    
    参数:
        state: PropagationState 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        surface_name: 表面名称
        surface_info: 表面信息（可选）
        valid_threshold: 有效区域阈值
    
    返回:
        (AccuracyMetrics, AnalysisData)
    """
    # 提取仿真振幅和相位
    sim_amplitude = state.amplitude
    sim_phase = state.phase
    pilot_params = state.pilot_beam_params
    
    # 计算理想高斯振幅
    ideal_amplitude = compute_ideal_gaussian_amplitude(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 定义有效区域
    max_amp = np.max(sim_amplitude)
    if max_amp > 0:
        valid_mask = sim_amplitude > valid_threshold * max_amp
    else:
        valid_mask = np.zeros_like(sim_amplitude, dtype=bool)
    
    # 提取 Pilot Beam 信息
    pilot_info = extract_pilot_beam_info(pilot_params)
    
    # 处理无有效数据的情况
    if np.sum(valid_mask) == 0:
        metrics = AccuracyMetrics(
            surface_name=surface_name,
            position=state.position,
            amplitude_rms_error=np.nan,
            amplitude_max_error=np.nan,
            amplitude_pv_error=np.nan,
            phase_rms_error_rad=np.nan,
            phase_max_error_rad=np.nan,
            phase_pv_error_rad=np.nan,
            pilot_info=pilot_info,
            surface_info=surface_info,
        )
        
        analysis_data = AnalysisData(
            sim_amplitude=sim_amplitude,
            sim_phase=sim_phase,
            ideal_amplitude=ideal_amplitude,
            pilot_phase=pilot_phase,
            amp_diff=np.zeros_like(sim_amplitude),
            phase_diff=np.zeros_like(sim_phase),
            valid_mask=valid_mask,
            physical_size_mm=physical_size_mm,
        )
        
        return metrics, analysis_data
    
    # 归一化振幅
    sim_amp_norm = sim_amplitude / max_amp
    ideal_amp_norm = ideal_amplitude / np.max(ideal_amplitude)
    
    # 振幅误差
    amp_diff = sim_amp_norm - ideal_amp_norm
    amplitude_rms_error = np.sqrt(np.mean(amp_diff[valid_mask]**2))
    amplitude_max_error = np.max(np.abs(amp_diff[valid_mask]))
    amplitude_pv_error = np.max(amp_diff[valid_mask]) - np.min(amp_diff[valid_mask])
    
    # 相位误差（使用 angle(exp(1j * diff)) 处理 2π 周期性）
    phase_diff = np.angle(np.exp(1j * (sim_phase - pilot_phase)))
    phase_rms_error = np.sqrt(np.mean(phase_diff[valid_mask]**2))
    phase_max_error = np.max(np.abs(phase_diff[valid_mask]))
    phase_pv_error = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
    
    metrics = AccuracyMetrics(
        surface_name=surface_name,
        position=state.position,
        amplitude_rms_error=amplitude_rms_error,
        amplitude_max_error=amplitude_max_error,
        amplitude_pv_error=amplitude_pv_error,
        phase_rms_error_rad=phase_rms_error,
        phase_max_error_rad=phase_max_error,
        phase_pv_error_rad=phase_pv_error,
        pilot_info=pilot_info,
        surface_info=surface_info,
    )
    
    analysis_data = AnalysisData(
        sim_amplitude=sim_amplitude,
        sim_phase=sim_phase,
        ideal_amplitude=ideal_amplitude,
        pilot_phase=pilot_phase,
        amp_diff=amp_diff,
        phase_diff=phase_diff,
        valid_mask=valid_mask,
        physical_size_mm=physical_size_mm,
    )
    
    return metrics, analysis_data


print("[OK] 精度分析函数定义完成")


# ============================================================================
# 第四部分：可视化函数
# ============================================================================

print_section("第四部分：定义可视化函数")


def plot_surface_analysis(
    metrics: AccuracyMetrics,
    data: AnalysisData,
    output_path: str,
) -> None:
    """绘制单个表面的详细分析图表
    
    包含：
    - 仿真振幅、理想振幅、振幅误差
    - 仿真相位、Pilot Beam 相位、相位误差
    - 误差直方图
    - Pilot Beam 参数信息
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    extent = [-data.physical_size_mm/2, data.physical_size_mm/2,
              -data.physical_size_mm/2, data.physical_size_mm/2]
    
    # 第一行：振幅分析
    # 1. 仿真振幅
    ax1 = fig.add_subplot(gs[0, 0])
    max_amp = np.max(data.sim_amplitude)
    sim_amp_norm = data.sim_amplitude / max_amp if max_amp > 0 else data.sim_amplitude
    im1 = ax1.imshow(sim_amp_norm, extent=extent, cmap='viridis', origin='lower')
    ax1.set_title('仿真振幅（归一化）')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 理想高斯振幅
    ax2 = fig.add_subplot(gs[0, 1])
    ideal_amp_norm = data.ideal_amplitude / np.max(data.ideal_amplitude)
    im2 = ax2.imshow(ideal_amp_norm, extent=extent, cmap='viridis', origin='lower')
    ax2.set_title('理想高斯振幅')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 振幅误差
    ax3 = fig.add_subplot(gs[0, 2])
    amp_diff_masked = np.where(data.valid_mask, data.amp_diff, np.nan)
    vmax_amp = max(0.05, np.nanmax(np.abs(amp_diff_masked)))
    im3 = ax3.imshow(amp_diff_masked, extent=extent, cmap='RdBu_r',
                     vmin=-vmax_amp, vmax=vmax_amp, origin='lower')
    ax3.set_title(f'振幅误差\nRMS={metrics.amplitude_rms_error:.6f}')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 振幅误差直方图
    ax4 = fig.add_subplot(gs[0, 3])
    amp_diff_valid = data.amp_diff[data.valid_mask]
    if len(amp_diff_valid) > 0:
        ax4.hist(amp_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_title(f'振幅误差分布\nPV={metrics.amplitude_pv_error:.6f}')
    ax4.set_xlabel('误差')
    ax4.set_ylabel('计数')
    
    # 第二行：相位分析
    # 5. 仿真相位
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(data.sim_phase, extent=extent, cmap='twilight', origin='lower')
    ax5.set_title('仿真相位 (rad)')
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Pilot Beam 参考相位
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(data.pilot_phase, extent=extent, cmap='twilight', origin='lower')
    ax6.set_title('Pilot Beam 参考相位 (rad)')
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('Y (mm)')
    plt.colorbar(im6, ax=ax6)
    
    # 7. 相位误差
    ax7 = fig.add_subplot(gs[1, 2])
    phase_diff_masked = np.where(data.valid_mask, data.phase_diff, np.nan)
    vmax_phase = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
    im7 = ax7.imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                     vmin=-vmax_phase, vmax=vmax_phase, origin='lower')
    ax7.set_title(f'相位误差 (rad)\nRMS={metrics.phase_rms_error_waves:.6f} waves')
    ax7.set_xlabel('X (mm)')
    ax7.set_ylabel('Y (mm)')
    plt.colorbar(im7, ax=ax7)
    
    # 8. 相位误差直方图
    ax8 = fig.add_subplot(gs[1, 3])
    phase_diff_valid = data.phase_diff[data.valid_mask]
    if len(phase_diff_valid) > 0:
        ax8.hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
        ax8.axvline(0, color='red', linestyle='--', linewidth=2)
    ax8.set_title(f'相位误差分布\nPV={metrics.phase_pv_error_waves:.6f} waves')
    ax8.set_xlabel('误差 (rad)')
    ax8.set_ylabel('计数')
    
    # 第三行：参数信息
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    # 构建信息文本
    info_lines = []
    info_lines.append(f"表面: {metrics.surface_name} ({metrics.position})")
    info_lines.append("")
    info_lines.append("Pilot Beam 参数:")
    info_lines.append(f"  波长: {metrics.pilot_info.wavelength_um:.4f} μm")
    info_lines.append(f"  束腰半径: {metrics.pilot_info.waist_radius_mm:.4f} mm")
    info_lines.append(f"  束腰位置: {metrics.pilot_info.waist_position_mm:.4f} mm")
    
    if np.isinf(metrics.pilot_info.curvature_radius_mm):
        info_lines.append(f"  曲率半径: ∞ (平面波)")
    else:
        info_lines.append(f"  曲率半径: {metrics.pilot_info.curvature_radius_mm:.4f} mm")
    
    info_lines.append(f"  光斑大小: {metrics.pilot_info.spot_size_mm:.4f} mm")
    info_lines.append(f"  瑞利长度: {metrics.pilot_info.rayleigh_length_mm:.4f} mm")
    
    if metrics.surface_info is not None:
        info_lines.append("")
        info_lines.append("表面参数:")
        info_lines.append(f"  类型: {metrics.surface_info.surface_type}")
        info_lines.append(f"  是否反射镜: {metrics.surface_info.is_mirror}")
        
        if np.isinf(metrics.surface_info.radius):
            info_lines.append(f"  曲率半径: ∞ (平面)")
        else:
            info_lines.append(f"  曲率半径: {metrics.surface_info.radius:.4f} mm")
        
        info_lines.append(f"  圆锥常数: {metrics.surface_info.conic:.4f}")
        info_lines.append(f"  材料: {metrics.surface_info.material}")
        info_lines.append(f"  顶点位置: {metrics.surface_info.vertex_position}")
        if metrics.surface_info.comment:
            info_lines.append(f"  注释: {metrics.surface_info.comment}")
    
    info_text = "\n".join(info_lines)
    ax_info.text(0.02, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'精度分析: {metrics.surface_name} ({metrics.position})', 
                 fontsize=14, y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 保存: {output_path}")



def plot_summary_table(
    metrics_list: List[AccuracyMetrics],
    output_path: str,
    title: str = "精度验证汇总",
) -> None:
    """绘制精度汇总表格"""
    fig, ax = plt.subplots(figsize=(16, max(6, len(metrics_list) * 0.5 + 2)))
    ax.axis('off')
    
    # 准备表格数据
    headers = ['表面', '位置', '振幅RMS', '振幅PV', '相位RMS\n(waves)', 
               '相位PV\n(waves)', '曲率半径\n(mm)', '光斑大小\n(mm)']
    
    data = []
    for m in metrics_list:
        curvature_str = "∞" if np.isinf(m.pilot_info.curvature_radius_mm) else f"{m.pilot_info.curvature_radius_mm:.2f}"
        
        row = [
            m.surface_name,
            m.position,
            f"{m.amplitude_rms_error:.6f}" if not np.isnan(m.amplitude_rms_error) else "N/A",
            f"{m.amplitude_pv_error:.6f}" if not np.isnan(m.amplitude_pv_error) else "N/A",
            f"{m.phase_rms_error_waves:.6f}" if not np.isnan(m.phase_rms_error_waves) else "N/A",
            f"{m.phase_pv_error_waves:.6f}" if not np.isnan(m.phase_pv_error_waves) else "N/A",
            curvature_str,
            f"{m.pilot_info.spot_size_mm:.4f}",
        ]
        data.append(row)
    
    # 创建表格
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 根据误差大小设置单元格颜色
    for row_idx, m in enumerate(metrics_list):
        row = row_idx + 1
        
        # 相位 RMS 误差着色
        if not np.isnan(m.phase_rms_error_waves):
            if m.phase_rms_error_waves < 0.001:
                table[(row, 4)].set_facecolor('#C6EFCE')  # 绿色
            elif m.phase_rms_error_waves < 0.01:
                table[(row, 4)].set_facecolor('#FFEB9C')  # 黄色
            else:
                table[(row, 4)].set_facecolor('#FFC7CE')  # 红色
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 保存汇总表格: {output_path}")


def plot_optical_layout(
    zmx_file: str,
    output_path: str,
) -> None:
    """绘制光学系统布局图"""
    try:
        loader = ZmxOpticLoader(zmx_file)
        optic = loader.load()
        
        fig, ax, _ = view_2d(optic, projection='YZ', num_rays=5)
        ax.set_title(f'光学系统布局\n{Path(zmx_file).name}')
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] 保存光路图: {output_path}")
    except Exception as e:
        print(f"[WARN] 无法绘制光路图: {e}")


print("[OK] 可视化函数定义完成")


# ============================================================================
# 第五部分：主验证函数
# ============================================================================

print_section("第五部分：定义主验证函数")


# 精度要求（严格，不得放松）
INITIAL_PHASE_RMS_TOLERANCE_WAVES = 0.001
PROPAGATION_PHASE_RMS_TOLERANCE_WAVES = 0.01


def validate_zmx_accuracy(
    zmx_file: str,
    wavelength_um: float = 0.55,
    w0_mm: float = 5.0,
    grid_size: int = 256,
    physical_size_mm: float = 40.0,
    num_rays: int = 150,
    output_dir: str = ".",
    generate_plots: bool = True,
) -> ValidationResult:
    """对 ZMX 文件进行完整的精度验证
    
    参数:
        zmx_file: ZMX 文件路径
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        num_rays: 光线采样数量
        output_dir: 输出目录
        generate_plots: 是否生成图表
    
    返回:
        ValidationResult 对象
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zmx_name = Path(zmx_file).stem
    
    print(f"\n开始验证: {zmx_file}")
    print(f"时间戳: {timestamp}")
    
    # 记录源参数
    source_params = {
        'wavelength_um': wavelength_um,
        'w0_mm': w0_mm,
        'grid_size': grid_size,
        'physical_size_mm': physical_size_mm,
        'num_rays': num_rays,
    }
    
    # 计算瑞利长度
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    source_params['rayleigh_length_mm'] = z_R
    
    print(f"\n光源参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰半径: {w0_mm} mm")
    print(f"  瑞利长度: {z_R:.1f} mm")
    print(f"  网格大小: {grid_size} × {grid_size}")
    print(f"  物理尺寸: {physical_size_mm} mm")
    
    # 加载光学系统
    print(f"\n加载光学系统...")
    try:
        optical_system = load_optical_system_from_zmx(zmx_file)
        print(f"  表面数量: {len(optical_system)}")
        
        for surface in optical_system:
            mirror_str = " [MIRROR]" if surface.is_mirror else ""
            radius_str = f"R={surface.radius:.2f}" if not np.isinf(surface.radius) else "R=∞"
            print(f"  - 表面 {surface.index}: {surface.surface_type}, "
                  f"{radius_str}{mirror_str}, comment='{surface.comment}'")
    except Exception as e:
        return ValidationResult(
            zmx_file=zmx_file,
            timestamp=timestamp,
            source_params=source_params,
            metrics_list=[],
            analysis_data_list=[],
            first_error_surface=None,
            error_message=f"加载光学系统失败: {e}",
            passed=False,
        )
    
    # 创建光源定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    # 创建传播器
    print(f"\n创建传播器...")
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        num_rays=num_rays,
    )
    
    # 执行传播
    print(f"执行传播仿真...")
    result = propagator.propagate()
    
    if not result.success:
        return ValidationResult(
            zmx_file=zmx_file,
            timestamp=timestamp,
            source_params=source_params,
            metrics_list=[],
            analysis_data_list=[],
            first_error_surface=None,
            error_message=f"传播失败: {result.error_message}",
            passed=False,
        )
    
    print(f"  传播成功!")
    print(f"  表面状态数量: {len(result.surface_states)}")
    print(f"  总光程: {result.total_path_length:.1f} mm")
    
    # 分析每个表面的精度
    print(f"\n分析精度...")
    metrics_list = []
    analysis_data_list = []
    first_error_surface = None
    
    # 创建表面索引到表面定义的映射
    surface_map = {s.index: s for s in optical_system}
    
    for state in result.surface_states:
        if state.surface_index < 0:
            name = "Initial_source"
            surface_info = None
        else:
            surface = surface_map.get(state.surface_index)
            if surface is not None:
                name = f"Surface_{state.surface_index}"
                if surface.comment:
                    name += f"_{surface.comment}"
                surface_info = extract_surface_info(surface)
            else:
                name = f"Surface_{state.surface_index}"
                surface_info = None
        
        # 使用 PROPER 的实际采样
        grid_sampling = GridSampling.from_proper(state.proper_wfo)
        
        metrics, data = analyze_state_accuracy(
            state=state,
            grid_size=grid_size,
            physical_size_mm=grid_sampling.physical_size_mm,
            surface_name=name,
            surface_info=surface_info,
        )
        
        metrics_list.append(metrics)
        analysis_data_list.append(data)
        
        # 检查是否出现误差
        if first_error_surface is None and not np.isnan(metrics.phase_rms_error_waves):
            if state.surface_index < 0:
                # 初始波前
                if metrics.phase_rms_error_waves > INITIAL_PHASE_RMS_TOLERANCE_WAVES:
                    first_error_surface = name
            else:
                # 传播后
                if metrics.phase_rms_error_waves > PROPAGATION_PHASE_RMS_TOLERANCE_WAVES:
                    first_error_surface = name
    
    # 打印精度汇总
    print(f"\n精度汇总:")
    print("-" * 100)
    print(f"{'表面':<30} {'位置':<10} {'振幅RMS':<12} {'相位RMS':<15} {'相位PV':<15} {'曲率半径':<12}")
    print(f"{'':30} {'':10} {'误差':12} {'(waves)':15} {'(waves)':15} {'(mm)':12}")
    print("-" * 100)
    
    for m in metrics_list:
        curvature_str = "∞" if np.isinf(m.pilot_info.curvature_radius_mm) else f"{m.pilot_info.curvature_radius_mm:.1f}"
        
        amp_rms_str = f"{m.amplitude_rms_error:.6f}" if not np.isnan(m.amplitude_rms_error) else "N/A"
        phase_rms_str = f"{m.phase_rms_error_waves:.6f}" if not np.isnan(m.phase_rms_error_waves) else "N/A"
        phase_pv_str = f"{m.phase_pv_error_waves:.6f}" if not np.isnan(m.phase_pv_error_waves) else "N/A"
        
        print(f"{m.surface_name:<30} {m.position:<10} {amp_rms_str:<12} "
              f"{phase_rms_str:<15} {phase_pv_str:<15} {curvature_str:<12}")
    
    print("-" * 100)
    
    # 生成图表
    if generate_plots:
        print(f"\n生成图表...")
        
        # 光路图
        layout_path = Path(output_dir) / f"{zmx_name}_layout.png"
        plot_optical_layout(zmx_file, str(layout_path))
        
        # 汇总表格
        summary_path = Path(output_dir) / f"{zmx_name}_accuracy_summary.png"
        plot_summary_table(metrics_list, str(summary_path), 
                          title=f"精度验证汇总 - {zmx_name}")
        
        # 每个表面的详细分析
        for metrics, data in zip(metrics_list, analysis_data_list):
            safe_name = metrics.surface_name.replace(" ", "_").replace("/", "_")
            detail_path = Path(output_dir) / f"{zmx_name}_{safe_name}_{metrics.position}.png"
            plot_surface_analysis(metrics, data, str(detail_path))
    
    # 判断是否通过
    passed = first_error_surface is None
    
    return ValidationResult(
        zmx_file=zmx_file,
        timestamp=timestamp,
        source_params=source_params,
        metrics_list=metrics_list,
        analysis_data_list=analysis_data_list,
        first_error_surface=first_error_surface,
        error_message=None if passed else f"从 {first_error_surface} 开始出现误差",
        passed=passed,
    )


print("[OK] 主验证函数定义完成")



# ============================================================================
# 第六部分：主程序
# ============================================================================

print_section("第六部分：主程序")


def main():
    """主函数"""
    
    # 默认测试文件
    default_zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(
        description="ZMX 文件综合精度验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python zmx_comprehensive_accuracy_test.py
  python zmx_comprehensive_accuracy_test.py --zmx path/to/file.zmx
  python zmx_comprehensive_accuracy_test.py --wavelength 0.633 --w0 3.0
        """
    )
    parser.add_argument('--zmx', type=str, default=default_zmx_file,
                        help='ZMX 文件路径')
    parser.add_argument('--wavelength', type=float, default=0.55,
                        help='波长 (μm)')
    parser.add_argument('--w0', type=float, default=5.0,
                        help='束腰半径 (mm)')
    parser.add_argument('--grid-size', type=int, default=256,
                        help='网格大小')
    parser.add_argument('--physical-size', type=float, default=40.0,
                        help='物理尺寸 (mm)')
    parser.add_argument('--num-rays', type=int, default=150,
                        help='光线采样数量')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录')
    parser.add_argument('--no-plots', action='store_true',
                        help='不生成图表')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    zmx_path = Path(args.zmx)
    if not zmx_path.exists():
        print(f"[ERROR] ZMX 文件不存在: {args.zmx}")
        sys.exit(1)
    
    # 执行验证
    result = validate_zmx_accuracy(
        zmx_file=str(zmx_path),
        wavelength_um=args.wavelength,
        w0_mm=args.w0,
        grid_size=args.grid_size,
        physical_size_mm=args.physical_size,
        num_rays=args.num_rays,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )
    
    # 输出结果
    print_section("验证结果")
    
    print(f"""
ZMX 文件: {result.zmx_file}
时间戳: {result.timestamp}

光源参数:
  波长: {result.source_params['wavelength_um']} μm
  束腰半径: {result.source_params['w0_mm']} mm
  瑞利长度: {result.source_params['rayleigh_length_mm']:.1f} mm
  网格大小: {result.source_params['grid_size']} × {result.source_params['grid_size']}
  物理尺寸: {result.source_params['physical_size_mm']} mm

精度要求:
  初始波前相位 RMS 误差: < {INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves
  传播后相位 RMS 误差: < {PROPAGATION_PHASE_RMS_TOLERANCE_WAVES} waves
""")
    
    if result.passed:
        print("[OK] 验证通过！所有表面精度满足要求。")
    else:
        print(f"[FAIL] 验证失败！")
        if result.error_message:
            print(f"  错误信息: {result.error_message}")
        if result.first_error_surface:
            print(f"  首次出现误差的表面: {result.first_error_surface}")
            print(f"\n请检查该表面的详细分析图表，分析误差来源。")
    
    # 误差来源分析
    print(f"""
误差来源分析:

1. 初始波前误差
   - 理论上应为零（仿真复振幅直接从 Pilot Beam 参数构造）
   - 实际误差来源：数值精度、网格离散化

2. 自由空间传播误差
   - PROPER 使用远场近似曲率 R_ref = z - z_w0
   - Pilot Beam 使用严格公式 R = z × (1 + (z_R/z)²)
   - 在近场（z ≈ z_R）时，两者存在差异
   - 误差随传播距离增加而减小（趋近远场）

3. 光学元件引入的误差
   - 反射镜的像差（球差、彗差等）
   - 光线追迹的数值误差
   - 网格重采样误差

4. 相位解包裹误差
   - 当相位梯度过大时，解包裹可能失败
   - 使用 Pilot Beam 参考相位可以最小化残差相位
""")
    
    # 返回结果
    return result


if __name__ == '__main__':
    result = main()
    
    # 如果验证失败，等待用户指引
    if not result.passed:
        print("\n" + "=" * 80)
        print("验证未通过，等待用户指引...")
        print("=" * 80)
        print(f"\n请查看生成的图表文件，分析误差来源。")
        print(f"首次出现误差的表面: {result.first_error_surface}")
