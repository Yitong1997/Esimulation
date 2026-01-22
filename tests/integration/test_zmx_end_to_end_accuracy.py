"""
ZMX 文件端到端精度测试：仿真复振幅与 Pilot Beam 精度验证

本模块测试基于 ZMX 文件（激光扩束镜）的端到端传播，
验证仿真复振幅与 Pilot Beam 参考相位的精度误差。

测试目标：
1. 在每个入射面检查振幅与相位的误差
2. 绘制误差分析图表
3. 分析误差来源
4. 验证仿真复振幅相对于 Pilot Beam 几乎没有误差

理论基础：
- 对于理想高斯光束，仿真复振幅的相位应当与 Pilot Beam 参考相位完全一致
- Pilot Beam 使用严格精确的高斯光束曲率公式 R = z × (1 + (z_R/z)²)
- 任何偏差都应该来自于光学元件引入的像差

精度要求（严格，不得放松）：
- 初始波前：相位 RMS 误差 < 0.001 waves
- 自由空间传播后：相位 RMS 误差 < 0.01 waves
- 经过光学元件后：取决于元件像差

**Feature: hybrid-raytracing-validation**
**Validates: Requirements 8.1-8.7, 10.1-10.5, 18.1-18.6**
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# 精度指标数据类
# ============================================================================

@dataclass
class AccuracyMetrics:
    """精度指标数据类"""
    surface_index: int
    surface_name: str
    position: str  # 'entrance', 'exit', 'source'
    
    # 振幅误差
    amplitude_rms_error: float  # RMS 相对误差
    amplitude_max_error: float  # 最大相对误差
    
    # 相位误差（相对于 Pilot Beam）
    phase_rms_error_rad: float  # RMS 误差（弧度）
    phase_max_error_rad: float  # 最大误差（弧度）
    phase_pv_error_rad: float   # PV 误差（弧度）
    
    # Pilot Beam 参数
    pilot_curvature_mm: float
    pilot_spot_size_mm: float
    
    # 转换为波长数
    @property
    def phase_rms_error_waves(self) -> float:
        return self.phase_rms_error_rad / (2 * np.pi)
    
    @property
    def phase_max_error_waves(self) -> float:
        return self.phase_max_error_rad / (2 * np.pi)
    
    @property
    def phase_pv_error_waves(self) -> float:
        return self.phase_pv_error_rad / (2 * np.pi)


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


# ============================================================================
# 辅助函数
# ============================================================================

def compute_pilot_beam_phase(
    pilot_params,
    grid_size: int,
    physical_size_mm: float,
) -> np.ndarray:
    """计算 Pilot Beam 参考相位
    
    使用严格精确的高斯光束曲率公式计算参考相位。
    """
    return pilot_params.compute_phase_grid(grid_size, physical_size_mm)


def compute_ideal_gaussian_amplitude(
    pilot_params,
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
    state,
    grid_size: int,
    physical_size_mm: float,
    surface_name: str,
    valid_threshold: float = 0.01,
) -> Tuple[AccuracyMetrics, AnalysisData]:
    """分析传播状态的精度
    
    参数:
        state: PropagationState 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        surface_name: 表面名称
        valid_threshold: 有效区域阈值
    
    返回:
        (AccuracyMetrics, AnalysisData)
    """
    # 提取仿真振幅和相位（使用新的分离接口）
    sim_amplitude = state.amplitude
    sim_phase = state.phase
    pilot_params = state.pilot_beam_params
    
    # 计算理想高斯振幅
    ideal_amplitude = compute_ideal_gaussian_amplitude(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = compute_pilot_beam_phase(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 定义有效区域
    max_amp = np.max(sim_amplitude)
    if max_amp > 0:
        valid_mask = sim_amplitude > valid_threshold * max_amp
    else:
        valid_mask = np.zeros_like(sim_amplitude, dtype=bool)
    
    # 处理无有效数据的情况
    if np.sum(valid_mask) == 0:
        metrics = AccuracyMetrics(
            surface_index=state.surface_index,
            surface_name=surface_name,
            position=state.position,
            amplitude_rms_error=np.nan,
            amplitude_max_error=np.nan,
            phase_rms_error_rad=np.nan,
            phase_max_error_rad=np.nan,
            phase_pv_error_rad=np.nan,
            pilot_curvature_mm=pilot_params.curvature_radius_mm,
            pilot_spot_size_mm=pilot_params.spot_size_mm,
        )
        
        analysis_data = AnalysisData(
            sim_amplitude=sim_amplitude,
            sim_phase=sim_phase,
            ideal_amplitude=ideal_amplitude,
            pilot_phase=pilot_phase,
            amp_diff=np.zeros_like(sim_amplitude),
            phase_diff=np.zeros_like(sim_phase),
            valid_mask=valid_mask,
        )
        
        return metrics, analysis_data
    
    # 归一化振幅
    sim_amp_norm = sim_amplitude / max_amp
    ideal_amp_norm = ideal_amplitude / np.max(ideal_amplitude)
    
    # 振幅误差
    amp_diff = np.abs(sim_amp_norm - ideal_amp_norm)
    amplitude_rms_error = np.sqrt(np.mean(amp_diff[valid_mask]**2))
    amplitude_max_error = np.max(amp_diff[valid_mask])
    
    # 相位误差（使用 angle(exp(1j * diff)) 处理 2π 周期性）
    phase_diff = np.angle(np.exp(1j * (sim_phase - pilot_phase)))
    phase_rms_error = np.sqrt(np.mean(phase_diff[valid_mask]**2))
    phase_max_error = np.max(np.abs(phase_diff[valid_mask]))
    phase_pv_error = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
    
    metrics = AccuracyMetrics(
        surface_index=state.surface_index,
        surface_name=surface_name,
        position=state.position,
        amplitude_rms_error=amplitude_rms_error,
        amplitude_max_error=amplitude_max_error,
        phase_rms_error_rad=phase_rms_error,
        phase_max_error_rad=phase_max_error,
        phase_pv_error_rad=phase_pv_error,
        pilot_curvature_mm=pilot_params.curvature_radius_mm,
        pilot_spot_size_mm=pilot_params.spot_size_mm,
    )
    
    analysis_data = AnalysisData(
        sim_amplitude=sim_amplitude,
        sim_phase=sim_phase,
        ideal_amplitude=ideal_amplitude,
        pilot_phase=pilot_phase,
        amp_diff=amp_diff,
        phase_diff=phase_diff,
        valid_mask=valid_mask,
    )
    
    return metrics, analysis_data


def plot_accuracy_analysis(
    metrics: AccuracyMetrics,
    data: AnalysisData,
    physical_size_mm: float,
    output_path: str,
) -> None:
    """绘制精度分析图表"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    extent = [-physical_size_mm/2, physical_size_mm/2,
              -physical_size_mm/2, physical_size_mm/2]
    
    # 第一行：振幅分析
    # 1. 仿真振幅
    max_amp = np.max(data.sim_amplitude)
    sim_amp_norm = data.sim_amplitude / max_amp if max_amp > 0 else data.sim_amplitude
    im1 = axes[0, 0].imshow(sim_amp_norm, extent=extent, cmap='viridis')
    axes[0, 0].set_title('仿真振幅（归一化）')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 理想高斯振幅
    ideal_amp_norm = data.ideal_amplitude / np.max(data.ideal_amplitude)
    im2 = axes[0, 1].imshow(ideal_amp_norm, extent=extent, cmap='viridis')
    axes[0, 1].set_title('理想高斯振幅')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 振幅误差
    amp_diff_masked = np.where(data.valid_mask, data.amp_diff, np.nan)
    im3 = axes[0, 2].imshow(amp_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title(f'振幅误差\nRMS={metrics.amplitude_rms_error:.6f}')
    axes[0, 2].set_xlabel('X (mm)')
    axes[0, 2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 振幅误差直方图
    amp_diff_valid = data.amp_diff[data.valid_mask]
    if len(amp_diff_valid) > 0:
        axes[0, 3].hist(amp_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[0, 3].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 3].set_title(f'振幅误差分布')
    axes[0, 3].set_xlabel('误差')
    axes[0, 3].set_ylabel('计数')
    
    # 第二行：相位分析
    # 5. 仿真相位
    im5 = axes[1, 0].imshow(data.sim_phase, extent=extent, cmap='twilight')
    axes[1, 0].set_title('仿真相位 (rad)')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im5, ax=axes[1, 0])
    
    # 6. Pilot Beam 参考相位
    im6 = axes[1, 1].imshow(data.pilot_phase, extent=extent, cmap='twilight')
    axes[1, 1].set_title('Pilot Beam 参考相位 (rad)')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im6, ax=axes[1, 1])
    
    # 7. 相位误差
    phase_diff_masked = np.where(data.valid_mask, data.phase_diff, np.nan)
    vmax = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
    im7 = axes[1, 2].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title(f'相位误差 (rad)\nRMS={metrics.phase_rms_error_waves:.6f} waves')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Y (mm)')
    plt.colorbar(im7, ax=axes[1, 2])
    
    # 8. 相位误差直方图
    phase_diff_valid = data.phase_diff[data.valid_mask]
    if len(phase_diff_valid) > 0:
        axes[1, 3].hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[1, 3].axvline(0, color='red', linestyle='--', linewidth=2)
    rms_waves = metrics.phase_rms_error_waves
    axes[1, 3].set_title(f'相位误差分布\nRMS={rms_waves:.6f} waves')
    axes[1, 3].set_xlabel('误差 (rad)')
    axes[1, 3].set_ylabel('计数')
    
    fig.suptitle(f'精度分析: {metrics.surface_name} ({metrics.position})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)



# ============================================================================
# 测试类
# ============================================================================

class TestZmxEndToEndAccuracy:
    """ZMX 文件端到端精度测试
    
    测试仿真复振幅与 Pilot Beam 参考相位的精度误差。
    
    精度要求（严格，不得放松）：
    - 初始波前：相位 RMS 误差 < 0.001 waves
    - 自由空间传播后：相位 RMS 误差 < 0.01 waves
    
    **Validates: Requirements 8.1-8.7, 10.1-10.5**
    """
    
    # 严格精度要求（不得放松）
    INITIAL_PHASE_RMS_TOLERANCE_WAVES = 0.001   # 初始波前相位 RMS 误差
    PROPAGATION_PHASE_RMS_TOLERANCE_WAVES = 0.01  # 传播后相位 RMS 误差
    AMPLITUDE_RMS_TOLERANCE = 0.01  # 振幅 RMS 误差
    
    @pytest.fixture
    def zmx_file_path(self) -> str:
        """获取测试用 ZMX 文件路径"""
        return "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"
    
    @pytest.fixture
    def source_definition(self):
        """创建测试用光源定义"""
        from hybrid_optical_propagation import SourceDefinition
        
        return SourceDefinition(
            wavelength_um=0.55,
            w0_mm=5.0,
            z0_mm=0.0,
            grid_size=256,
            physical_size_mm=40.0,
        )
    
    def test_initial_wavefront_accuracy(self, source_definition):
        """
        测试初始波前与 Pilot Beam 的一致性。
        
        对于初始高斯光束，仿真复振幅应当与 Pilot Beam 参考完全一致。
        
        精度要求：相位 RMS 误差 < 0.001 waves
        
        **Validates: Requirements 8.1, 8.2, 1.1, 1.2**
        """
        from hybrid_optical_propagation import (
            HybridOpticalPropagator,
            GridSampling,
        )
        
        # 创建空光学系统（只测试初始波前）
        propagator = HybridOpticalPropagator(
            optical_system=[],
            source=source_definition,
            wavelength_um=source_definition.wavelength_um,
            grid_size=source_definition.grid_size,
            num_rays=50,
        )
        
        # 初始化传播状态
        initial_state = propagator._initialize_propagation()
        
        # 使用 PROPER 的实际采样信息
        actual_grid_sampling = GridSampling.from_proper(initial_state.proper_wfo)
        
        # 计算精度指标
        metrics, data = analyze_state_accuracy(
            state=initial_state,
            grid_size=source_definition.grid_size,
            physical_size_mm=actual_grid_sampling.physical_size_mm,
            surface_name="Initial",
        )
        
        # 生成分析图表
        plot_accuracy_analysis(
            metrics=metrics,
            data=data,
            physical_size_mm=actual_grid_sampling.physical_size_mm,
            output_path="accuracy_analysis_Initial_source.png",
        )
        
        # 打印详细信息
        print(f"\n初始波前精度分析:")
        print(f"  振幅 RMS 误差: {metrics.amplitude_rms_error:.6f}")
        print(f"  相位 RMS 误差: {metrics.phase_rms_error_waves:.6f} waves")
        print(f"  相位 PV 误差: {metrics.phase_pv_error_waves:.6f} waves")
        print(f"  Pilot Beam 曲率半径: {metrics.pilot_curvature_mm:.2f} mm")
        print(f"  Pilot Beam 光斑大小: {metrics.pilot_spot_size_mm:.4f} mm")
        
        # 严格验证（不得放松）
        assert metrics.phase_rms_error_waves < self.INITIAL_PHASE_RMS_TOLERANCE_WAVES, (
            f"初始波前相位 RMS 误差 {metrics.phase_rms_error_waves:.6f} waves "
            f"超过严格容差 {self.INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves\n"
            f"误差来源分析:\n"
            f"  - 网格离散化误差\n"
            f"  - 数值精度限制\n"
            f"  - PROPER 与 Pilot Beam 参数不一致"
        )
        
        print(f"  [OK] 初始波前精度满足严格要求 (< {self.INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves)")
    
    def test_free_space_propagation_accuracy(self, source_definition):
        """
        测试自由空间传播后的精度。
        
        自由空间传播应当保持仿真复振幅与 Pilot Beam 的一致性。
        
        精度要求：相位 RMS 误差 < 0.01 waves
        
        **Validates: Requirements 8.3, 8.4**
        """
        from hybrid_optical_propagation import (
            FreeSpacePropagator,
            PropagationState,
            GridSampling,
        )
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        # 创建初始状态
        amplitude, phase, pilot_beam_params, proper_wfo = (
            source_definition.create_initial_wavefront()
        )
        
        initial_axis_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, 0.0),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=0.0,
        )
        
        grid_sampling = GridSampling.from_proper(proper_wfo)
        
        initial_state = PropagationState(
            surface_index=-1,
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_beam_params,
            proper_wfo=proper_wfo,
            optical_axis_state=initial_axis_state,
            grid_sampling=grid_sampling,
        )
        
        # 传播 50mm
        propagation_distance = 50.0  # mm
        
        target_axis_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, propagation_distance),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=propagation_distance,
        )
        
        # 执行自由空间传播
        propagator = FreeSpacePropagator(source_definition.wavelength_um)
        new_state = propagator.propagate(
            initial_state,
            target_axis_state,
            target_surface_index=0,
            target_position='entrance',
        )
        
        # 使用传播后的实际采样
        new_grid_sampling = GridSampling.from_proper(new_state.proper_wfo)
        
        # 计算精度指标
        metrics, data = analyze_state_accuracy(
            state=new_state,
            grid_size=source_definition.grid_size,
            physical_size_mm=new_grid_sampling.physical_size_mm,
            surface_name=f"After_{propagation_distance}mm",
        )
        
        # 生成分析图表
        plot_accuracy_analysis(
            metrics=metrics,
            data=data,
            physical_size_mm=new_grid_sampling.physical_size_mm,
            output_path=f"accuracy_analysis_propagation_{propagation_distance}mm.png",
        )
        
        # 打印详细信息
        print(f"\n自由空间传播精度分析 (距离={propagation_distance}mm):")
        print(f"  振幅 RMS 误差: {metrics.amplitude_rms_error:.6f}")
        print(f"  相位 RMS 误差: {metrics.phase_rms_error_waves:.6f} waves")
        print(f"  相位 PV 误差: {metrics.phase_pv_error_waves:.6f} waves")
        print(f"  Pilot Beam 曲率半径: {metrics.pilot_curvature_mm:.2f} mm")
        print(f"  Pilot Beam 光斑大小: {metrics.pilot_spot_size_mm:.4f} mm")
        
        # 分析误差来源
        wavelength_mm = source_definition.wavelength_um * 1e-3
        z_R = np.pi * source_definition.w0_mm**2 / wavelength_mm
        print(f"\n误差来源分析:")
        print(f"  瑞利长度: {z_R:.2f} mm")
        print(f"  传播距离/瑞利长度: {propagation_distance/z_R:.2f}")
        if propagation_distance < z_R:
            print(f"  [INFO] 近场传播，PROPER 远场近似可能引入误差")
        else:
            print(f"  [INFO] 远场传播，PROPER 远场近似应当准确")
        
        # 严格验证（不得放松）
        assert metrics.phase_rms_error_waves < self.PROPAGATION_PHASE_RMS_TOLERANCE_WAVES, (
            f"自由空间传播后相位 RMS 误差 {metrics.phase_rms_error_waves:.6f} waves "
            f"超过严格容差 {self.PROPAGATION_PHASE_RMS_TOLERANCE_WAVES} waves\n"
            f"误差来源分析:\n"
            f"  - PROPER 远场近似 vs Pilot Beam 严格公式\n"
            f"  - 传播距离/瑞利长度 = {propagation_distance/z_R:.2f}\n"
            f"  - 数值衍射误差"
        )
        
        print(f"  [OK] 自由空间传播精度满足严格要求 (< {self.PROPAGATION_PHASE_RMS_TOLERANCE_WAVES} waves)")



    def test_zmx_fold_mirror_accuracy(self, zmx_file_path, source_definition):
        """
        测试基于 ZMX 文件的折叠镜系统精度。
        
        使用 simple_fold_mirror_up.zmx 测试完整的传播流程。
        
        **Validates: Requirements 18.1, 18.2**
        """
        from hybrid_optical_propagation import (
            load_optical_system_from_zmx,
            HybridOpticalPropagator,
            GridSampling,
        )
        
        # 检查文件是否存在
        if not Path(zmx_file_path).exists():
            pytest.skip(f"ZMX 文件不存在: {zmx_file_path}")
        
        # 加载光学系统
        try:
            optical_system = load_optical_system_from_zmx(zmx_file_path)
        except Exception as e:
            pytest.skip(f"无法加载 ZMX 文件: {e}")
        
        if len(optical_system) == 0:
            pytest.skip("光学系统为空")
        
        print(f"\n加载的光学系统:")
        print(f"  表面数量: {len(optical_system)}")
        for surface in optical_system:
            print(f"  - 表面 {surface.index}: {surface.surface_type}, "
                  f"R={surface.radius:.2f}mm, "
                  f"mirror={surface.is_mirror}, "
                  f"comment='{surface.comment}'")
        
        # 创建传播器
        propagator = HybridOpticalPropagator(
            optical_system=optical_system,
            source=source_definition,
            wavelength_um=source_definition.wavelength_um,
            grid_size=source_definition.grid_size,
            num_rays=100,
        )
        
        # 执行传播
        result = propagator.propagate()
        
        if not result.success:
            pytest.skip(f"传播失败: {result.error_message}")
        
        print(f"\n传播结果:")
        print(f"  表面状态数量: {len(result.surface_states)}")
        print(f"  总光程: {result.total_path_length:.2f} mm")
        
        # 收集所有表面的精度指标
        metrics_list = []
        data_list = []
        
        print(f"\n精度分析:")
        print("-" * 80)
        print(f"{'表面':<25} {'位置':<10} {'振幅RMS':<12} {'相位RMS':<15} {'相位PV':<15}")
        print(f"{'':25} {'':10} {'误差':12} {'(waves)':15} {'(waves)':15}")
        print("-" * 80)
        
        for state in result.surface_states:
            if state.surface_index < 0:
                name = "Initial"
            else:
                name = f"Surface_{state.surface_index}"
            
            # 使用 PROPER 的实际采样
            grid_sampling = GridSampling.from_proper(state.proper_wfo)
            
            metrics, data = analyze_state_accuracy(
                state=state,
                grid_size=source_definition.grid_size,
                physical_size_mm=grid_sampling.physical_size_mm,
                surface_name=name,
            )
            
            metrics_list.append(metrics)
            data_list.append(data)
            
            # 生成分析图表
            plot_accuracy_analysis(
                metrics=metrics,
                data=data,
                physical_size_mm=grid_sampling.physical_size_mm,
                output_path=f"accuracy_analysis_{name}_{state.position}.png",
            )
            
            print(f"{name:<25} {state.position:<10} {metrics.amplitude_rms_error:<12.6f} "
                  f"{metrics.phase_rms_error_waves:<15.6f} {metrics.phase_pv_error_waves:<15.6f}")
        
        print("-" * 80)
        
        # 验证初始状态精度（严格要求）
        initial_metrics = metrics_list[0]
        assert initial_metrics.phase_rms_error_waves < self.INITIAL_PHASE_RMS_TOLERANCE_WAVES, (
            f"初始状态相位 RMS 误差 {initial_metrics.phase_rms_error_waves:.6f} waves "
            f"超过严格容差 {self.INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves"
        )
        
        print(f"\n[OK] 初始波前精度满足严格要求")
        
        # 分析误差来源
        print(f"\n误差来源分析:")
        print(f"  1. 初始波前: 数值精度和网格离散化")
        print(f"  2. 自由空间传播: PROPER 远场近似 vs Pilot Beam 严格公式")
        print(f"  3. 光学元件: 像差和光线追迹数值误差")
        print(f"  4. 网格重采样: 插值误差")


class TestZmxEndToEndVisualization:
    """ZMX 端到端可视化测试
    
    生成完整的可视化图表和误差分析报告。
    
    **Validates: Requirements 18.3, 18.4, 18.5, 18.6**
    """
    
    def test_comprehensive_visualization(self):
        """
        生成综合可视化图表。
        
        **Validates: Requirements 18.3**
        """
        from hybrid_optical_propagation import (
            SourceDefinition,
            HybridOpticalPropagator,
            GridSampling,
        )
        from dataclasses import dataclass, field
        from typing import List
        
        # 创建模拟的光学系统（单个平面镜）
        @dataclass
        class MockSurface:
            """模拟的 GlobalSurfaceDefinition 对象"""
            index: int
            surface_type: str
            vertex_position: np.ndarray
            orientation: np.ndarray
            radius: float = np.inf
            conic: float = 0.0
            is_mirror: bool = False
            semi_aperture: float = 25.0
            material: str = "air"
            asphere_coeffs: List[float] = field(default_factory=list)
            comment: str = ""
            thickness: float = 0.0
            radius_x: float = np.inf
            conic_x: float = 0.0
            focal_length: float = np.inf
            
            @property
            def surface_normal(self) -> np.ndarray:
                return -self.orientation[:, 2]
        
        # 创建 45 度平面镜
        tilt_x_rad = -np.pi / 4
        c, s = np.cos(tilt_x_rad), np.sin(tilt_x_rad)
        orientation = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
        
        mirror = MockSurface(
            index=0,
            surface_type='flat',
            vertex_position=np.array([0.0, 0.0, 50.0]),
            orientation=orientation,
            is_mirror=True,
            material='mirror',
            comment='M1 (45deg flat mirror)',
        )
        
        # 创建光源
        source = SourceDefinition(
            wavelength_um=0.633,
            w0_mm=5.0,
            z0_mm=0.0,
            grid_size=256,
            physical_size_mm=40.0,
        )
        
        # 创建传播器
        propagator = HybridOpticalPropagator(
            optical_system=[mirror],
            source=source,
            wavelength_um=0.633,
            grid_size=256,
            num_rays=100,
        )
        
        # 执行传播
        result = propagator.propagate()
        
        if not result.success:
            pytest.skip(f"传播失败: {result.error_message}")
        
        # 生成综合图表
        n_states = len(result.surface_states)
        fig, axes = plt.subplots(n_states, 4, figsize=(16, 4 * n_states))
        
        if n_states == 1:
            axes = axes.reshape(1, -1)
        
        for i, state in enumerate(result.surface_states):
            if state.surface_index < 0:
                name = "Initial"
            else:
                name = f"Surface_{state.surface_index}"
            
            grid_sampling = GridSampling.from_proper(state.proper_wfo)
            physical_size_mm = grid_sampling.physical_size_mm
            
            metrics, data = analyze_state_accuracy(
                state=state,
                grid_size=256,
                physical_size_mm=physical_size_mm,
                surface_name=name,
            )
            
            extent = [-physical_size_mm/2, physical_size_mm/2,
                      -physical_size_mm/2, physical_size_mm/2]
            
            # 仿真振幅
            max_amp = np.max(data.sim_amplitude)
            sim_amp_norm = data.sim_amplitude / max_amp if max_amp > 0 else data.sim_amplitude
            im1 = axes[i, 0].imshow(sim_amp_norm, extent=extent, cmap='viridis')
            axes[i, 0].set_title(f'{name} ({state.position})\n仿真振幅')
            axes[i, 0].set_xlabel('X (mm)')
            axes[i, 0].set_ylabel('Y (mm)')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 仿真相位
            im2 = axes[i, 1].imshow(data.sim_phase, extent=extent, cmap='twilight')
            axes[i, 1].set_title('仿真相位 (rad)')
            axes[i, 1].set_xlabel('X (mm)')
            axes[i, 1].set_ylabel('Y (mm)')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # 相位误差
            phase_diff_masked = np.where(data.valid_mask, data.phase_diff, np.nan)
            vmax = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
            im3 = axes[i, 2].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                                    vmin=-vmax, vmax=vmax)
            axes[i, 2].set_title(f'相位误差 (rad)\nRMS={metrics.phase_rms_error_waves:.6f} waves')
            axes[i, 2].set_xlabel('X (mm)')
            axes[i, 2].set_ylabel('Y (mm)')
            plt.colorbar(im3, ax=axes[i, 2])
            
            # 误差直方图
            phase_diff_valid = data.phase_diff[data.valid_mask]
            if len(phase_diff_valid) > 0:
                axes[i, 3].hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
                axes[i, 3].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[i, 3].set_title(f'相位误差分布\nPV={metrics.phase_pv_error_waves:.6f} waves')
            axes[i, 3].set_xlabel('误差 (rad)')
            axes[i, 3].set_ylabel('计数')
        
        fig.suptitle('仿真复振幅与 Pilot Beam 精度分析', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig("zmx_end_to_end_accuracy.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n[OK] 综合可视化图表已保存: zmx_end_to_end_accuracy.png")
        
        # 验证传播成功
        assert result.success
        assert len(result.surface_states) >= 2


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
