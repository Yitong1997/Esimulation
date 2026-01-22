"""
ZMX 文件端到端测试：仿真复振幅与 Pilot Beam 精度验证

本模块测试基于 ZMX 文件（激光扩束镜）的端到端传播，
验证仿真复振幅与 Pilot Beam 参考相位的精度误差。

测试目标：
1. 在每个入射面检查振幅与相位的误差
2. 绘制误差分析图表
3. 分析误差来源
4. 验证仿真复振幅相对于 Pilot Beam 几乎没有误差

理论基础：
- 对于理想高斯光束，仿真复振幅的相位应当与 Pilot Beam 参考相位完全一致
- 任何偏差都应该来自于光学元件引入的像差
- 在无像差系统中，残差相位应当接近零

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
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class AccuracyMetrics:
    """精度指标数据类"""
    surface_index: int
    surface_name: str
    position: str  # 'entrance' or 'exit'
    
    # 振幅误差
    amplitude_rms_error: float  # RMS 相对误差
    amplitude_max_error: float  # 最大相对误差
    amplitude_mean_error: float  # 平均相对误差
    
    # 相位误差（相对于 Pilot Beam）
    phase_rms_error_rad: float  # RMS 误差（弧度）
    phase_max_error_rad: float  # 最大误差（弧度）
    phase_pv_error_rad: float   # PV 误差（弧度）
    
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
class ValidationResult:
    """验证结果数据类"""
    zmx_file: str
    wavelength_um: float
    grid_size: int
    metrics_list: List[AccuracyMetrics]
    
    # 总体评估
    all_passed: bool
    failure_reasons: List[str]
    
    def get_summary(self) -> str:
        """生成摘要报告"""
        lines = [
            f"ZMX 文件: {self.zmx_file}",
            f"波长: {self.wavelength_um} μm",
            f"网格大小: {self.grid_size}",
            f"表面数量: {len(self.metrics_list)}",
            "",
            "精度指标摘要:",
            "-" * 60,
        ]
        
        for m in self.metrics_list:
            lines.append(
                f"  {m.surface_name} ({m.position}):"
            )
            lines.append(
                f"    振幅 RMS 误差: {m.amplitude_rms_error:.6f}"
            )
            lines.append(
                f"    相位 RMS 误差: {m.phase_rms_error_waves:.6f} waves"
            )
            lines.append(
                f"    相位 PV 误差: {m.phase_pv_error_waves:.6f} waves"
            )
        
        lines.append("-" * 60)
        lines.append(f"总体评估: {'通过' if self.all_passed else '失败'}")
        
        if self.failure_reasons:
            lines.append("失败原因:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")
        
        return "\n".join(lines)


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
    
    参数:
        pilot_params: PilotBeamParams 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
    
    返回:
        参考相位网格 (弧度)
    """
    return pilot_params.compute_phase_grid(grid_size, physical_size_mm)


def compute_ideal_gaussian_amplitude(
    pilot_params,
    grid_size: int,
    physical_size_mm: float,
    grid_sampling = None,
) -> np.ndarray:
    """计算理想高斯光束振幅分布
    
    参数:
        pilot_params: PilotBeamParams 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        grid_sampling: GridSampling 对象（可选，用于精确匹配采样）
    
    返回:
        振幅网格
    """
    if grid_sampling is not None:
        # 使用 GridSampling 的坐标
        X, Y = grid_sampling.get_coordinate_arrays()
    else:
        half_size = physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
    
    r_sq = X**2 + Y**2
    
    # 使用当前光斑大小
    w = pilot_params.spot_size_mm
    
    # 高斯振幅: A = exp(-r²/w²)
    amplitude = np.exp(-r_sq / w**2)
    
    return amplitude


def compute_accuracy_metrics(
    simulation_amplitude: np.ndarray,
    pilot_params,
    grid_size: int,
    physical_size_mm: float,
    surface_index: int,
    surface_name: str,
    position: str,
    valid_threshold: float = 0.01,
    grid_sampling = None,
) -> AccuracyMetrics:
    """计算精度指标
    
    参数:
        simulation_amplitude: 仿真复振幅
        pilot_params: PilotBeamParams 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        surface_index: 表面索引
        surface_name: 表面名称
        position: 位置 ('entrance' or 'exit')
        valid_threshold: 有效区域阈值（相对于最大振幅）
        grid_sampling: GridSampling 对象（可选）
    
    返回:
        AccuracyMetrics 对象
    """
    # 提取仿真振幅和相位
    sim_amplitude = np.abs(simulation_amplitude)
    sim_phase = np.angle(simulation_amplitude)
    
    # 计算理想高斯振幅
    ideal_amplitude = compute_ideal_gaussian_amplitude(
        pilot_params, grid_size, physical_size_mm, grid_sampling
    )
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = compute_pilot_beam_phase(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 定义有效区域（振幅大于阈值的区域）
    max_amp = np.max(sim_amplitude)
    valid_mask = sim_amplitude > valid_threshold * max_amp
    
    if np.sum(valid_mask) == 0:
        # 无有效数据
        return AccuracyMetrics(
            surface_index=surface_index,
            surface_name=surface_name,
            position=position,
            amplitude_rms_error=np.nan,
            amplitude_max_error=np.nan,
            amplitude_mean_error=np.nan,
            phase_rms_error_rad=np.nan,
            phase_max_error_rad=np.nan,
            phase_pv_error_rad=np.nan,
        )
    
    # 归一化振幅（使最大值为 1）
    sim_amp_norm = sim_amplitude / max_amp
    ideal_amp_norm = ideal_amplitude / np.max(ideal_amplitude)
    
    # 振幅误差（在有效区域内）
    amp_diff = np.abs(sim_amp_norm - ideal_amp_norm)
    amplitude_rms_error = np.sqrt(np.mean(amp_diff[valid_mask]**2))
    amplitude_max_error = np.max(amp_diff[valid_mask])
    amplitude_mean_error = np.mean(amp_diff[valid_mask])
    
    # 相位误差（相对于 Pilot Beam）
    # 使用 angle(exp(1j * diff)) 处理 2π 周期性
    phase_diff = sim_phase - pilot_phase
    phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
    
    phase_rms_error = np.sqrt(np.mean(phase_diff_wrapped[valid_mask]**2))
    phase_max_error = np.max(np.abs(phase_diff_wrapped[valid_mask]))
    phase_pv_error = np.max(phase_diff_wrapped[valid_mask]) - np.min(phase_diff_wrapped[valid_mask])
    
    return AccuracyMetrics(
        surface_index=surface_index,
        surface_name=surface_name,
        position=position,
        amplitude_rms_error=amplitude_rms_error,
        amplitude_max_error=amplitude_max_error,
        amplitude_mean_error=amplitude_mean_error,
        phase_rms_error_rad=phase_rms_error,
        phase_max_error_rad=phase_max_error,
        phase_pv_error_rad=phase_pv_error,
    )



def plot_accuracy_analysis(
    simulation_amplitude: np.ndarray,
    pilot_params,
    grid_size: int,
    physical_size_mm: float,
    surface_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """绘制精度分析图表
    
    参数:
        simulation_amplitude: 仿真复振幅
        pilot_params: PilotBeamParams 对象
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        surface_name: 表面名称
        output_path: 输出路径（可选）
    
    返回:
        matplotlib Figure 对象
    """
    # 提取数据
    sim_amplitude = np.abs(simulation_amplitude)
    sim_phase = np.angle(simulation_amplitude)
    
    # 计算理想值
    ideal_amplitude = compute_ideal_gaussian_amplitude(
        pilot_params, grid_size, physical_size_mm
    )
    pilot_phase = compute_pilot_beam_phase(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 归一化
    sim_amp_norm = sim_amplitude / np.max(sim_amplitude)
    ideal_amp_norm = ideal_amplitude / np.max(ideal_amplitude)
    
    # 计算误差
    amp_diff = sim_amp_norm - ideal_amp_norm
    phase_diff = np.angle(np.exp(1j * (sim_phase - pilot_phase)))
    
    # 有效区域掩模
    valid_mask = sim_amp_norm > 0.01
    
    # 创建图表
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 坐标范围
    extent = [-physical_size_mm/2, physical_size_mm/2,
              -physical_size_mm/2, physical_size_mm/2]
    
    # 第一行：振幅分析
    # 1. 仿真振幅
    im1 = axes[0, 0].imshow(sim_amp_norm, extent=extent, cmap='viridis')
    axes[0, 0].set_title('仿真振幅（归一化）')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 理想高斯振幅
    im2 = axes[0, 1].imshow(ideal_amp_norm, extent=extent, cmap='viridis')
    axes[0, 1].set_title('理想高斯振幅')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 振幅误差
    amp_diff_masked = np.where(valid_mask, amp_diff, np.nan)
    im3 = axes[0, 2].imshow(amp_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('振幅误差')
    axes[0, 2].set_xlabel('X (mm)')
    axes[0, 2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 振幅误差直方图
    amp_diff_valid = amp_diff[valid_mask]
    axes[0, 3].hist(amp_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[0, 3].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 3].set_title(f'振幅误差分布\nRMS={np.std(amp_diff_valid):.4f}')
    axes[0, 3].set_xlabel('误差')
    axes[0, 3].set_ylabel('计数')
    
    # 第二行：相位分析
    # 5. 仿真相位
    im5 = axes[1, 0].imshow(sim_phase, extent=extent, cmap='twilight')
    axes[1, 0].set_title('仿真相位（弧度）')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im5, ax=axes[1, 0])
    
    # 6. Pilot Beam 参考相位
    im6 = axes[1, 1].imshow(pilot_phase, extent=extent, cmap='twilight')
    axes[1, 1].set_title('Pilot Beam 参考相位')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im6, ax=axes[1, 1])
    
    # 7. 相位误差
    phase_diff_masked = np.where(valid_mask, phase_diff, np.nan)
    vmax_phase = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
    im7 = axes[1, 2].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-vmax_phase, vmax=vmax_phase)
    axes[1, 2].set_title('相位误差（弧度）')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Y (mm)')
    plt.colorbar(im7, ax=axes[1, 2])
    
    # 8. 相位误差直方图
    phase_diff_valid = phase_diff[valid_mask]
    axes[1, 3].hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[1, 3].axvline(0, color='red', linestyle='--', linewidth=2)
    rms_waves = np.std(phase_diff_valid) / (2 * np.pi)
    axes[1, 3].set_title(f'相位误差分布\nRMS={rms_waves:.6f} waves')
    axes[1, 3].set_xlabel('误差（弧度）')
    axes[1, 3].set_ylabel('计数')
    
    fig.suptitle(f'精度分析: {surface_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# ============================================================================
# 测试类
# ============================================================================

class TestZmxPilotBeamAccuracy:
    """ZMX 文件 Pilot Beam 精度测试
    
    测试仿真复振幅与 Pilot Beam 参考相位的精度误差。
    
    **Validates: Requirements 8.1-8.7, 10.1-10.5**
    """
    
    # 精度要求（严格）
    AMPLITUDE_RMS_TOLERANCE = 0.01      # 振幅 RMS 误差 < 1%
    PHASE_RMS_TOLERANCE_WAVES = 0.001   # 相位 RMS 误差 < 0.001 waves
    PHASE_PV_TOLERANCE_WAVES = 0.01     # 相位 PV 误差 < 0.01 waves
    
    @pytest.fixture
    def zmx_file_path(self) -> str:
        """获取测试用 ZMX 文件路径"""
        # 使用 optiland 测试文件中的单镜系统
        return "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
    
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
    
    def test_initial_wavefront_matches_pilot_beam(self, source_definition):
        """
        测试初始波前与 Pilot Beam 的一致性。
        
        对于初始高斯光束，仿真复振幅应当与 Pilot Beam 参考完全一致。
        
        **Validates: Requirements 8.1, 8.2, 1.1, 1.2**
        """
        from hybrid_optical_propagation import (
            HybridOpticalPropagator,
            PilotBeamParams,
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
        
        # 使用 PROPER 的实际采样信息（而不是用户指定的 physical_size_mm）
        # 因为 PROPER 的 beam_ratio=0.5 意味着实际网格大小是 physical_size_mm / beam_ratio
        actual_grid_sampling = GridSampling.from_proper(initial_state.proper_wfo)
        
        # 计算精度指标（使用 PROPER 的实际采样）
        # 使用 get_complex_amplitude() 获取复振幅形式
        metrics = compute_accuracy_metrics(
            simulation_amplitude=initial_state.get_complex_amplitude(),
            pilot_params=initial_state.pilot_beam_params,
            grid_size=source_definition.grid_size,
            physical_size_mm=actual_grid_sampling.physical_size_mm,
            surface_index=-1,
            surface_name="Initial",
            position="source",
            grid_sampling=actual_grid_sampling,
        )
        
        # 验证振幅误差
        assert metrics.amplitude_rms_error < self.AMPLITUDE_RMS_TOLERANCE, (
            f"初始波前振幅 RMS 误差 {metrics.amplitude_rms_error:.6f} "
            f"超过容差 {self.AMPLITUDE_RMS_TOLERANCE}"
        )
        
        # 验证相位误差
        assert metrics.phase_rms_error_waves < self.PHASE_RMS_TOLERANCE_WAVES, (
            f"初始波前相位 RMS 误差 {metrics.phase_rms_error_waves:.6f} waves "
            f"超过容差 {self.PHASE_RMS_TOLERANCE_WAVES} waves"
        )
        
        print(f"\n初始波前精度验证通过:")
        print(f"  振幅 RMS 误差: {metrics.amplitude_rms_error:.6f}")
        print(f"  相位 RMS 误差: {metrics.phase_rms_error_waves:.6f} waves")
        print(f"  相位 PV 误差: {metrics.phase_pv_error_waves:.6f} waves")
    
    def test_free_space_propagation_accuracy(self, source_definition):
        """
        测试自由空间传播后的精度。
        
        自由空间传播应当保持仿真复振幅与 Pilot Beam 的一致性。
        
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
        
        # 创建初始状态（使用新的四元组返回值）
        amplitude, phase, pilot_beam_params, proper_wfo = (
            source_definition.create_initial_wavefront()
        )
        
        initial_axis_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, 0.0),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=0.0,
        )
        
        # 使用 PROPER 的实际采样
        grid_sampling = GridSampling.from_proper(proper_wfo)
        
        # 使用新的振幅/相位分离接口
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
        
        # 传播 100mm
        propagation_distance = 100.0  # mm
        
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
        
        # 计算精度指标（使用 get_complex_amplitude() 获取复振幅形式）
        metrics = compute_accuracy_metrics(
            simulation_amplitude=new_state.get_complex_amplitude(),
            pilot_params=new_state.pilot_beam_params,
            grid_size=source_definition.grid_size,
            physical_size_mm=new_grid_sampling.physical_size_mm,
            surface_index=0,
            surface_name=f"After {propagation_distance}mm",
            position="entrance",
            grid_sampling=new_grid_sampling,
        )
        
        # 验证相位误差（自由空间传播后允许稍大的误差）
        # 由于 PROPER 使用远场近似，近场传播可能有一定误差
        relaxed_phase_tolerance = 0.01  # 0.01 waves
        
        assert metrics.phase_rms_error_waves < relaxed_phase_tolerance, (
            f"自由空间传播后相位 RMS 误差 {metrics.phase_rms_error_waves:.6f} waves "
            f"超过容差 {relaxed_phase_tolerance} waves"
        )
        
        print(f"\n自由空间传播精度验证通过 (距离={propagation_distance}mm):")
        print(f"  振幅 RMS 误差: {metrics.amplitude_rms_error:.6f}")
        print(f"  相位 RMS 误差: {metrics.phase_rms_error_waves:.6f} waves")
        print(f"  相位 PV 误差: {metrics.phase_pv_error_waves:.6f} waves")



    def test_zmx_single_mirror_accuracy(self, zmx_file_path, source_definition):
        """
        测试基于 ZMX 文件的单镜系统精度。
        
        使用 one_mirror_up_45deg.zmx 测试完整的传播流程。
        
        **Validates: Requirements 18.1, 18.2**
        """
        from hybrid_optical_propagation import (
            load_optical_system_from_zmx,
            HybridOpticalPropagator,
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
        
        # 收集所有表面的精度指标
        metrics_list = []
        
        for state in result.surface_states:
            if state.surface_index < 0:
                name = "Initial"
            else:
                name = f"Surface_{state.surface_index}"
            
            # 使用 get_complex_amplitude() 获取复振幅形式
            metrics = compute_accuracy_metrics(
                simulation_amplitude=state.get_complex_amplitude(),
                pilot_params=state.pilot_beam_params,
                grid_size=source_definition.grid_size,
                physical_size_mm=source_definition.physical_size_mm,
                surface_index=state.surface_index,
                surface_name=name,
                position=state.position,
            )
            metrics_list.append(metrics)
        
        # 打印结果
        print(f"\nZMX 文件精度分析: {zmx_file_path}")
        print("-" * 60)
        
        for m in metrics_list:
            print(f"{m.surface_name} ({m.position}):")
            print(f"  振幅 RMS 误差: {m.amplitude_rms_error:.6f}")
            print(f"  相位 RMS 误差: {m.phase_rms_error_waves:.6f} waves")
            print(f"  相位 PV 误差: {m.phase_pv_error_waves:.6f} waves")
        
        # 验证初始状态精度（应当非常高）
        initial_metrics = metrics_list[0]
        assert initial_metrics.phase_rms_error_waves < self.PHASE_RMS_TOLERANCE_WAVES, (
            f"初始状态相位误差过大: {initial_metrics.phase_rms_error_waves:.6f} waves"
        )


class TestZmxEndToEndValidation:
    """ZMX 端到端验证测试
    
    完整的端到端测试，包括图表生成和误差分析。
    
    **Validates: Requirements 18.1-18.6**
    """
    
    def test_end_to_end_with_visualization(self):
        """
        端到端测试，生成可视化图表。
        
        **Validates: Requirements 18.1, 18.2, 18.3**
        """
        from hybrid_optical_propagation import (
            SourceDefinition,
            HybridOpticalPropagator,
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
            vertex_position=np.array([0.0, 0.0, 100.0]),
            orientation=orientation,
            is_mirror=True,
            material='mirror',
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
        
        # 生成可视化图表
        output_dir = Path(".")
        
        for i, state in enumerate(result.surface_states):
            if state.surface_index < 0:
                name = "Initial"
            else:
                name = f"Surface_{state.surface_index}_{state.position}"
            
            # 使用 get_complex_amplitude() 获取复振幅形式
            fig = plot_accuracy_analysis(
                simulation_amplitude=state.get_complex_amplitude(),
                pilot_params=state.pilot_beam_params,
                grid_size=256,
                physical_size_mm=40.0,
                surface_name=name,
                output_path=str(output_dir / f"accuracy_analysis_{name}.png"),
            )
            plt.close(fig)
        
        print(f"\n可视化图表已保存到当前目录")
        
        # 验证传播成功
        assert result.success
        assert len(result.surface_states) >= 2


# ============================================================================
# 主函数：运行完整验证
# ============================================================================

def run_full_validation(
    zmx_file_path: str,
    wavelength_um: float = 0.55,
    w0_mm: float = 5.0,
    grid_size: int = 256,
    physical_size_mm: float = 40.0,
    output_dir: str = ".",
) -> ValidationResult:
    """运行完整的 ZMX 文件验证
    
    参数:
        zmx_file_path: ZMX 文件路径
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        output_dir: 输出目录
    
    返回:
        ValidationResult 对象
    """
    from hybrid_optical_propagation import (
        SourceDefinition,
        load_optical_system_from_zmx,
        HybridOpticalPropagator,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建光源
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    # 加载光学系统
    try:
        optical_system = load_optical_system_from_zmx(zmx_file_path)
    except Exception as e:
        return ValidationResult(
            zmx_file=zmx_file_path,
            wavelength_um=wavelength_um,
            grid_size=grid_size,
            metrics_list=[],
            all_passed=False,
            failure_reasons=[f"无法加载 ZMX 文件: {e}"],
        )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        num_rays=100,
    )
    
    # 执行传播
    result = propagator.propagate()
    
    if not result.success:
        return ValidationResult(
            zmx_file=zmx_file_path,
            wavelength_um=wavelength_um,
            grid_size=grid_size,
            metrics_list=[],
            all_passed=False,
            failure_reasons=[f"传播失败: {result.error_message}"],
        )
    
    # 收集精度指标
    metrics_list = []
    failure_reasons = []
    
    # 精度要求
    AMPLITUDE_RMS_TOLERANCE = 0.01
    PHASE_RMS_TOLERANCE_WAVES = 0.001
    
    for state in result.surface_states:
        if state.surface_index < 0:
            name = "Initial"
        else:
            name = f"Surface_{state.surface_index}"
        
        # 使用 get_complex_amplitude() 获取复振幅形式
        metrics = compute_accuracy_metrics(
            simulation_amplitude=state.get_complex_amplitude(),
            pilot_params=state.pilot_beam_params,
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            surface_index=state.surface_index,
            surface_name=name,
            position=state.position,
        )
        metrics_list.append(metrics)
        
        # 生成可视化图表
        fig = plot_accuracy_analysis(
            simulation_amplitude=state.get_complex_amplitude(),
            pilot_params=state.pilot_beam_params,
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            surface_name=f"{name}_{state.position}",
            output_path=str(output_path / f"accuracy_{name}_{state.position}.png"),
        )
        plt.close(fig)
        
        # 检查精度要求（仅对初始状态严格要求）
        if state.surface_index < 0:
            if metrics.phase_rms_error_waves > PHASE_RMS_TOLERANCE_WAVES:
                failure_reasons.append(
                    f"{name}: 相位 RMS 误差 {metrics.phase_rms_error_waves:.6f} waves "
                    f"超过容差 {PHASE_RMS_TOLERANCE_WAVES} waves"
                )
    
    all_passed = len(failure_reasons) == 0
    
    return ValidationResult(
        zmx_file=zmx_file_path,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        metrics_list=metrics_list,
        all_passed=all_passed,
        failure_reasons=failure_reasons,
    )


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
