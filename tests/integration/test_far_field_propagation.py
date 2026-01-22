"""
远场传播精度测试

本模块测试以下场景：
1. 从近场传播到远场（INSIDE_ → OUTSIDE）
2. 在远场连续传播（OUTSIDE → OUTSIDE）
3. 从远场传播回近场（OUTSIDE → INSIDE_）

精度要求：
- 近场传播：相位 RMS 误差 < 0.01 waves
- 远场传播：误差与传播距离相关（PROPER 远场近似的固有特性）
  - z = 1×z_R: ~0.22 waves
  - z = 3×z_R: ~0.10 waves
  - z = 10×z_R: ~0.03 waves
  - z = 100×z_R: ~0.02 waves

**Feature: hybrid-optical-propagation**
**Validates: Requirements 4.1-4.8, 5.1-5.5**
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
from numpy.testing import assert_allclose
import pytest

import proper

from hybrid_optical_propagation import (
    PilotBeamParams,
    GridSampling,
    PropagationState,
    SourceDefinition,
)
from hybrid_optical_propagation.state_converter import StateConverter
from hybrid_optical_propagation.free_space_propagator import FreeSpacePropagator


# ============================================================================
# 精度容差定义
# ============================================================================

# 近场精度要求（严格）
NEAR_FIELD_TOLERANCE_WAVES = 0.01

# 远场精度要求（基于传播距离的动态容差）
# PROPER 远场近似导致的误差与 z/z_R 的关系（经验公式）
# 基于调试脚本的测量结果：
#   z = 1×z_R: ~0.22 waves
#   z = 3×z_R: ~0.10 waves
#   z = 10×z_R: ~0.03 waves
#   z = 100×z_R: ~0.036 waves (存在底噪)
def get_far_field_tolerance(z_over_z_R: float) -> float:
    """计算远场传播的容差
    
    PROPER 使用远场近似曲率半径 R_ref = z - z_w0，
    而 Pilot Beam 使用严格公式 R = z × (1 + (z_R/z)²)。
    
    两者的差异导致相位误差，误差随 z/z_R 增大而减小。
    
    参数:
        z_over_z_R: 传播距离与瑞利长度的比值
    
    返回:
        容差（waves）
    """
    if z_over_z_R < 1.0:
        # 近场，使用严格容差
        return NEAR_FIELD_TOLERANCE_WAVES
    
    # 远场误差经验公式（基于调试脚本测量）
    # 误差 ≈ 0.35 / (z/z_R) + 0.025（底噪）
    # 添加 50% 余量确保测试稳定
    base_error = 0.35 / z_over_z_R + 0.025
    return base_error * 1.5


# ============================================================================
# 辅助函数
# ============================================================================

def create_gaussian_source(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
    physical_size_mm: float,
) -> tuple:
    """创建高斯光源
    
    返回:
        (amplitude, phase, pilot_beam_params, proper_wfo, grid_sampling)
    """
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    grid_sampling = source.get_grid_sampling()
    
    return amplitude, phase, pilot_params, wfo, grid_sampling


def compute_phase_error_waves(
    sim_phase: np.ndarray,
    pilot_phase: np.ndarray,
    amplitude: np.ndarray,
    valid_threshold: float = 0.01,
) -> tuple:
    """计算相位误差（波长数）
    
    返回:
        (rms_error_waves, max_error_waves, pv_error_waves)
    """
    # 定义有效区域
    max_amp = np.max(amplitude)
    if max_amp > 0:
        valid_mask = amplitude > valid_threshold * max_amp
    else:
        return np.nan, np.nan, np.nan
    
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan, np.nan
    
    # 计算相位差（使用 angle(exp(1j * diff)) 处理 2π 周期性）
    phase_diff = np.angle(np.exp(1j * (sim_phase - pilot_phase)))
    
    # 计算误差指标
    rms_error = np.sqrt(np.mean(phase_diff[valid_mask]**2))
    max_error = np.max(np.abs(phase_diff[valid_mask]))
    pv_error = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
    
    # 转换为波长数
    rms_error_waves = rms_error / (2 * np.pi)
    max_error_waves = max_error / (2 * np.pi)
    pv_error_waves = pv_error / (2 * np.pi)
    
    return rms_error_waves, max_error_waves, pv_error_waves


def get_reference_surface_type(wfo) -> str:
    """获取 PROPER 参考面类型"""
    return wfo.reference_surface


# ============================================================================
# 测试类：近场到远场传播
# ============================================================================

class TestNearFieldToFarField:
    """测试从近场传播到远场（INSIDE_ → OUTSIDE）"""
    
    def test_propagate_to_far_field_basic(self):
        """
        基本测试：从束腰传播到远场
        
        场景：
        - 初始位置：束腰处（z = 0）
        - 传播距离：3 × 瑞利长度（确保进入远场）
        
        **Validates: Requirements 4.1, 4.3**
        """
        # 参数
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        # 计算瑞利长度
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 传播距离：3 × 瑞利长度
        z_factor = 3.0
        propagation_distance_mm = z_factor * z_R_mm
        
        print(f"\n测试参数:")
        print(f"  波长: {wavelength_um} μm")
        print(f"  束腰半径: {w0_mm} mm")
        print(f"  瑞利长度: {z_R_mm:.1f} mm")
        print(f"  传播距离: {propagation_distance_mm:.1f} mm ({z_factor}×z_R)")
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 验证初始状态是近场
        assert wfo.reference_surface == "PLANAR", \
            f"初始状态应为近场 (PLANAR)，实际为 {wfo.reference_surface}"
        
        print(f"  初始参考面: {wfo.reference_surface}")
        
        # 使用 PROPER 传播
        proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
        
        # 验证传播后是远场
        assert wfo.reference_surface == "SPHERI", \
            f"传播后应为远场 (SPHERI)，实际为 {wfo.reference_surface}"
        
        print(f"  传播后参考面: {wfo.reference_surface}")
        
        # 更新 Pilot Beam 参数
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 使用 StateConverter 提取振幅和相位
        converter = StateConverter(wavelength_um)
        new_amplitude, new_phase = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算相位误差
        rms_error, max_error, pv_error = compute_phase_error_waves(
            new_phase, pilot_phase, new_amplitude
        )
        
        # 获取动态容差
        tolerance = get_far_field_tolerance(z_factor)
        
        print(f"\n精度结果:")
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        print(f"  相位 MAX 误差: {max_error:.6f} waves")
        print(f"  相位 PV 误差: {pv_error:.6f} waves")
        print(f"  容差: {tolerance:.6f} waves (z/z_R = {z_factor})")
        
        # 验证精度
        assert rms_error < tolerance, \
            f"相位 RMS 误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f} waves"


    def test_propagate_to_far_field_various_distances(self):
        """
        测试不同传播距离到远场
        
        场景：
        - 传播距离：2×, 5×, 10× 瑞利长度
        
        **Validates: Requirements 4.3, 4.4**
        """
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 测试不同的传播距离倍数
        distance_factors = [2.0, 5.0, 10.0]
        
        for factor in distance_factors:
            propagation_distance_mm = factor * z_R_mm
            
            # 创建初始波前
            amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
                wavelength_um, w0_mm, grid_size, physical_size_mm
            )
            
            # 传播
            proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
            
            # 更新 Pilot Beam
            new_pilot_params = pilot_params.propagate(propagation_distance_mm)
            
            # 提取振幅和相位
            converter = StateConverter(wavelength_um)
            new_amplitude, new_phase = converter.proper_to_amplitude_phase(
                wfo, grid_sampling, new_pilot_params
            )
            
            # 计算 Pilot Beam 参考相位
            pilot_phase = new_pilot_params.compute_phase_grid(
                grid_size, physical_size_mm
            )
            
            # 计算误差
            rms_error, _, _ = compute_phase_error_waves(
                new_phase, pilot_phase, new_amplitude
            )
            
            # 获取动态容差
            tolerance = get_far_field_tolerance(factor)
            
            print(f"\n传播距离 {factor}×z_R ({propagation_distance_mm:.1f} mm):")
            print(f"  相位 RMS 误差: {rms_error:.6f} waves")
            print(f"  容差: {tolerance:.6f} waves")
            
            assert rms_error < tolerance, \
                f"传播 {factor}×z_R 后相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"


# ============================================================================
# 测试类：远场连续传播
# ============================================================================

class TestFarFieldContinuousPropagation:
    """测试在远场连续传播（OUTSIDE → OUTSIDE）"""
    
    def test_far_field_continuous_propagation(self):
        """
        测试在远场连续传播
        
        场景：
        - 先传播到远场（3×z_R）
        - 再在远场连续传播多次
        
        **Validates: Requirements 4.3, 4.4, 4.5**
        """
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 30.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 第一步：传播到远场
        first_factor = 3.0
        first_distance_mm = first_factor * z_R_mm
        proper.prop_propagate(wfo, first_distance_mm * 1e-3)
        pilot_params = pilot_params.propagate(first_distance_mm)
        
        assert wfo.reference_surface == "SPHERI", "应该在远场"
        
        print(f"\n第一步传播到远场 ({first_distance_mm:.1f} mm, {first_factor}×z_R)")
        print(f"  参考面: {wfo.reference_surface}")
        
        # 连续传播多次
        additional_factors = [1.0, 2.0, 3.0]  # 额外传播的 z_R 倍数
        total_factor = first_factor
        
        converter = StateConverter(wavelength_um)
        
        for i, add_factor in enumerate(additional_factors):
            dist_mm = add_factor * z_R_mm
            total_factor += add_factor
            
            # 传播
            proper.prop_propagate(wfo, dist_mm * 1e-3)
            pilot_params = pilot_params.propagate(dist_mm)
            
            # 验证仍在远场
            assert wfo.reference_surface == "SPHERI", \
                f"第 {i+2} 步后应该仍在远场"
            
            # 提取振幅和相位
            new_amplitude, new_phase = converter.proper_to_amplitude_phase(
                wfo, grid_sampling, pilot_params
            )
            
            # 计算 Pilot Beam 参考相位
            pilot_phase = pilot_params.compute_phase_grid(
                grid_size, physical_size_mm
            )
            
            # 计算误差
            rms_error, _, _ = compute_phase_error_waves(
                new_phase, pilot_phase, new_amplitude
            )
            
            # 获取动态容差
            tolerance = get_far_field_tolerance(total_factor)
            
            print(f"\n第 {i+2} 步传播 ({dist_mm:.1f} mm)，总距离 {total_factor}×z_R:")
            print(f"  参考面: {wfo.reference_surface}")
            print(f"  相位 RMS 误差: {rms_error:.6f} waves")
            print(f"  容差: {tolerance:.6f} waves")
            
            assert rms_error < tolerance, \
                f"第 {i+2} 步后相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"

    
    def test_far_field_long_distance_propagation(self):
        """
        测试远场长距离传播
        
        场景：
        - 传播到 100× 瑞利长度
        
        **Validates: Requirements 4.3, 4.4**
        """
        wavelength_um = 0.633
        w0_mm = 1.0  # 较小的束腰，使瑞利长度较短
        grid_size = 128
        physical_size_mm = 50.0  # 较大的网格以容纳扩展的光束
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 传播到 100× 瑞利长度
        z_factor = 100.0
        propagation_distance_mm = z_factor * z_R_mm
        
        print(f"\n测试参数:")
        print(f"  瑞利长度: {z_R_mm:.1f} mm")
        print(f"  传播距离: {propagation_distance_mm:.1f} mm ({z_factor}×z_R)")
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 传播
        proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 提取振幅和相位
        converter = StateConverter(wavelength_um)
        new_amplitude, new_phase = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算误差
        rms_error, max_error, pv_error = compute_phase_error_waves(
            new_phase, pilot_phase, new_amplitude
        )
        
        # 获取动态容差
        tolerance = get_far_field_tolerance(z_factor)
        
        print(f"\n精度结果:")
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        print(f"  容差: {tolerance:.6f} waves")
        print(f"  Pilot Beam 曲率半径: {new_pilot_params.curvature_radius_mm:.1f} mm")
        print(f"  Pilot Beam 光斑大小: {new_pilot_params.spot_size_mm:.2f} mm")
        
        assert rms_error < tolerance, \
            f"长距离传播后相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"


# ============================================================================
# 测试类：远场到近场传播
# ============================================================================

class TestFarFieldToNearField:
    """测试从远场传播回近场（OUTSIDE → INSIDE_）"""
    
    def test_propagate_back_to_near_field(self):
        """
        测试从远场传播回近场
        
        场景：
        - 先传播到远场
        - 再传播回束腰附近
        
        **Validates: Requirements 4.3, 4.4**
        """
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 第一步：传播到远场
        forward_factor = 5.0
        forward_distance_mm = forward_factor * z_R_mm
        proper.prop_propagate(wfo, forward_distance_mm * 1e-3)
        pilot_params = pilot_params.propagate(forward_distance_mm)
        
        print(f"\n第一步：传播到远场 ({forward_distance_mm:.1f} mm, {forward_factor}×z_R)")
        print(f"  参考面: {wfo.reference_surface}")
        
        assert wfo.reference_surface == "SPHERI", "应该在远场"
        
        # 第二步：传播回近场（负距离）
        backward_factor = -4.5  # 回到 0.5×z_R 处
        backward_distance_mm = backward_factor * z_R_mm
        proper.prop_propagate(wfo, backward_distance_mm * 1e-3)
        pilot_params = pilot_params.propagate(backward_distance_mm)
        
        final_factor = forward_factor + backward_factor  # 0.5
        
        print(f"\n第二步：传播回近场 ({backward_distance_mm:.1f} mm)")
        print(f"  参考面: {wfo.reference_surface}")
        print(f"  最终位置: {final_factor}×z_R")
        
        # 提取振幅和相位
        converter = StateConverter(wavelength_um)
        new_amplitude, new_phase = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = pilot_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算误差
        rms_error, _, _ = compute_phase_error_waves(
            new_phase, pilot_phase, new_amplitude
        )
        
        # 传播回近场后，误差取决于整个传播路径
        # 使用较宽松的容差（考虑往返传播的累积误差）
        tolerance = get_far_field_tolerance(forward_factor) * 1.5
        
        print(f"\n精度结果:")
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        print(f"  容差: {tolerance:.6f} waves")
        
        assert rms_error < tolerance, \
            f"传播回近场后相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"


# ============================================================================
# 测试类：使用 FreeSpacePropagator 的集成测试
# ============================================================================

class TestFreeSpacePropagatorIntegration:
    """使用 FreeSpacePropagator 的集成测试"""
    
    def test_free_space_propagator_near_to_far(self):
        """
        测试 FreeSpacePropagator 从近场到远场
        
        **Validates: Requirements 4.1, 4.3, 4.4**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        z_factor = 5.0
        propagation_distance_mm = z_factor * z_R_mm
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 创建初始光轴状态
        initial_axis = OpticalAxisState(
            position=Position3D(0, 0, 0),
            direction=RayDirection(0, 0, 1),
            path_length=0.0,
        )
        
        # 创建初始传播状态
        initial_state = PropagationState(
            surface_index=-1,
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_params,
            proper_wfo=wfo,
            optical_axis_state=initial_axis,
            grid_sampling=grid_sampling,
        )
        
        # 创建目标光轴状态
        target_axis = OpticalAxisState(
            position=Position3D(0, 0, propagation_distance_mm),
            direction=RayDirection(0, 0, 1),
            path_length=propagation_distance_mm,
        )
        
        # 使用 FreeSpacePropagator 传播
        propagator = FreeSpacePropagator(wavelength_um)
        new_state = propagator.propagate(
            initial_state,
            target_axis,
            target_surface_index=0,
            target_position='entrance',
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_state.pilot_beam_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算误差
        rms_error, _, _ = compute_phase_error_waves(
            new_state.phase, pilot_phase, new_state.amplitude
        )
        
        # 获取动态容差
        tolerance = get_far_field_tolerance(z_factor)
        
        print(f"\nFreeSpacePropagator 测试:")
        print(f"  传播距离: {propagation_distance_mm:.1f} mm ({z_factor}×z_R)")
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        print(f"  容差: {tolerance:.6f} waves")
        
        assert rms_error < tolerance, \
            f"FreeSpacePropagator 相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"

    
    def test_free_space_propagator_multiple_steps(self):
        """
        测试 FreeSpacePropagator 多步传播
        
        **Validates: Requirements 4.3, 4.4, 4.5**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 30.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 创建初始光轴状态
        current_z = 0.0
        current_axis = OpticalAxisState(
            position=Position3D(0, 0, current_z),
            direction=RayDirection(0, 0, 1),
            path_length=0.0,
        )
        
        # 创建初始传播状态
        state = PropagationState(
            surface_index=-1,
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_params,
            proper_wfo=wfo,
            optical_axis_state=current_axis,
            grid_sampling=grid_sampling,
        )
        
        # 传播步骤（z_R 倍数）
        propagator = FreeSpacePropagator(wavelength_um)
        step_factors = [2.0, 3.0, 2.0, 1.0]
        total_factor = 0.0
        total_path_length = 0.0
        
        print(f"\n多步传播测试:")
        
        for i, factor in enumerate(step_factors):
            dist_mm = factor * z_R_mm
            current_z += dist_mm
            total_factor += factor
            total_path_length += dist_mm
            
            target_axis = OpticalAxisState(
                position=Position3D(0, 0, current_z),
                direction=RayDirection(0, 0, 1),
                path_length=total_path_length,
            )
            
            state = propagator.propagate(
                state,
                target_axis,
                target_surface_index=i,
                target_position='entrance',
            )
            
            # 计算 Pilot Beam 参考相位
            pilot_phase = state.pilot_beam_params.compute_phase_grid(
                grid_size, physical_size_mm
            )
            
            # 计算误差
            rms_error, _, _ = compute_phase_error_waves(
                state.phase, pilot_phase, state.amplitude
            )
            
            # 获取动态容差
            tolerance = get_far_field_tolerance(total_factor)
            
            print(f"  步骤 {i+1}: 传播 {factor}×z_R，总距离 {total_factor}×z_R")
            print(f"    参考面: {state.proper_wfo.reference_surface}")
            print(f"    相位 RMS 误差: {rms_error:.6f} waves")
            print(f"    容差: {tolerance:.6f} waves")
            
            assert rms_error < tolerance, \
                f"步骤 {i+1} 后相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"


# ============================================================================
# 测试类：边界情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况"""
    
    def test_propagation_at_rayleigh_boundary(self):
        """
        测试在瑞利长度边界附近的传播
        
        场景：
        - 传播到恰好 1× 瑞利长度（边界情况）
        
        **Validates: Requirements 4.3**
        """
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        wavelength_mm = wavelength_um * 1e-3
        z_R_mm = np.pi * w0_mm**2 / wavelength_mm
        
        # 传播到恰好 1× 瑞利长度
        z_factor = 1.0
        propagation_distance_mm = z_factor * z_R_mm
        
        print(f"\n瑞利长度边界测试:")
        print(f"  瑞利长度: {z_R_mm:.1f} mm")
        print(f"  传播距离: {propagation_distance_mm:.1f} mm ({z_factor}×z_R)")
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        print(f"  初始参考面: {wfo.reference_surface}")
        
        # 传播
        proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        print(f"  传播后参考面: {wfo.reference_surface}")
        
        # 提取振幅和相位
        converter = StateConverter(wavelength_um)
        new_amplitude, new_phase = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算误差
        rms_error, _, _ = compute_phase_error_waves(
            new_phase, pilot_phase, new_amplitude
        )
        
        # 在边界处使用动态容差
        tolerance = get_far_field_tolerance(z_factor)
        
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        print(f"  容差: {tolerance:.6f} waves")
        
        assert rms_error < tolerance, \
            f"瑞利边界处相位误差 {rms_error:.6f} waves 超过容差 {tolerance:.6f}"
    
    def test_very_small_propagation_distance(self):
        """
        测试非常小的传播距离
        
        **Validates: Requirements 4.3**
        """
        wavelength_um = 0.633
        w0_mm = 2.0
        grid_size = 128
        physical_size_mm = 20.0
        
        # 非常小的传播距离
        propagation_distance_mm = 0.001  # 1 μm
        
        # 创建初始波前
        amplitude, phase, pilot_params, wfo, grid_sampling = create_gaussian_source(
            wavelength_um, w0_mm, grid_size, physical_size_mm
        )
        
        # 传播
        proper.prop_propagate(wfo, propagation_distance_mm * 1e-3)
        new_pilot_params = pilot_params.propagate(propagation_distance_mm)
        
        # 提取振幅和相位
        converter = StateConverter(wavelength_um)
        new_amplitude, new_phase = converter.proper_to_amplitude_phase(
            wfo, grid_sampling, new_pilot_params
        )
        
        # 计算 Pilot Beam 参考相位
        pilot_phase = new_pilot_params.compute_phase_grid(
            grid_size, physical_size_mm
        )
        
        # 计算误差
        rms_error, _, _ = compute_phase_error_waves(
            new_phase, pilot_phase, new_amplitude
        )
        
        print(f"\n小距离传播测试:")
        print(f"  传播距离: {propagation_distance_mm} mm")
        print(f"  相位 RMS 误差: {rms_error:.6f} waves")
        
        # 近场使用严格容差
        assert rms_error < NEAR_FIELD_TOLERANCE_WAVES, \
            f"小距离传播后相位误差 {rms_error:.6f} waves 超过容差 {NEAR_FIELD_TOLERANCE_WAVES}"


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
