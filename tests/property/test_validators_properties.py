"""
验证器属性基测试

验证 PhaseContinuityValidator 和 EnergyConservationValidator 的正确性。

**Feature: hybrid-raytracing-validation**
**Validates: Requirements 2.5, 5.1, 5.2, 7.2**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.validators import (
    PhaseContinuityValidator,
    EnergyConservationValidator,
    ValidationResult,
)


# ============================================================================
# 测试策略定义
# ============================================================================

# 网格大小策略
grid_size_strategy = st.sampled_from([32, 64, 128])

# 相位范围策略
phase_range_strategy = st.floats(min_value=0.1, max_value=10.0)

# 振幅策略
amplitude_strategy = st.floats(min_value=0.1, max_value=10.0)


# ============================================================================
# 属性 4：相位连续性验证
# ============================================================================

@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    max_phase=phase_range_strategy,
)
def test_property_4_continuous_phase_passes(
    grid_size: int,
    max_phase: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 4: 相位连续性**
    **Validates: Requirements 2.5, 7.2**
    
    连续相位分布应通过验证。
    """
    # 创建连续相位分布（线性相位）
    x = np.linspace(0, max_phase, grid_size)
    y = np.linspace(0, max_phase, grid_size)
    X, Y = np.meshgrid(x, y)
    phase_grid = X + Y  # 线性相位，梯度恒定
    
    # 确保相邻像素相位差小于 π
    max_gradient = max_phase / (grid_size - 1) * 2  # 对角线方向
    if max_gradient >= np.pi:
        # 跳过会导致不连续的情况
        return
    
    validator = PhaseContinuityValidator()
    result = validator.validate_phase_grid(phase_grid)
    
    assert result.is_valid, (
        f"连续相位应通过验证: {result.message}"
    )


@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
)
def test_property_4_discontinuous_phase_fails(
    grid_size: int,
):
    """
    **Feature: hybrid-raytracing-validation, Property 4: 相位连续性**
    **Validates: Requirements 2.5, 7.2**
    
    包含 2π 跳变的相位分布应失败验证。
    """
    # 创建包含 2π 跳变的相位分布
    phase_grid = np.zeros((grid_size, grid_size))
    # 在中间位置添加 2π 跳变
    mid = grid_size // 2
    phase_grid[:, mid:] = 2 * np.pi
    
    validator = PhaseContinuityValidator()
    result = validator.validate_phase_grid(phase_grid)
    
    assert not result.is_valid, (
        f"包含 2π 跳变的相位应失败验证: {result.message}"
    )


@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
)
def test_property_4_phase_with_mask(
    grid_size: int,
):
    """
    **Feature: hybrid-raytracing-validation, Property 4: 相位连续性**
    **Validates: Requirements 2.5, 7.2**
    
    使用掩模时，只检查有效区域。
    """
    # 创建包含跳变的相位分布
    phase_grid = np.zeros((grid_size, grid_size))
    mid = grid_size // 2
    phase_grid[:, mid:] = 2 * np.pi
    
    # 创建掩模，只包含左半部分（无跳变区域）
    valid_mask = np.zeros((grid_size, grid_size), dtype=bool)
    valid_mask[:, :mid-1] = True  # 不包含跳变边界
    
    validator = PhaseContinuityValidator()
    result = validator.validate_phase_grid(phase_grid, valid_mask=valid_mask)
    
    assert result.is_valid, (
        f"使用掩模排除跳变区域后应通过验证: {result.message}"
    )


@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    threshold=st.floats(min_value=0.1, max_value=np.pi),
)
def test_property_4_custom_threshold(
    grid_size: int,
    threshold: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 4: 相位连续性**
    **Validates: Requirements 2.5, 7.2**
    
    自定义阈值应正确应用。
    """
    # 创建相位分布，相邻像素差为 threshold / 2
    phase_step = threshold / 2
    x = np.arange(grid_size) * phase_step
    phase_grid = np.tile(x, (grid_size, 1))
    
    validator = PhaseContinuityValidator(threshold_rad=threshold)
    result = validator.validate_phase_grid(phase_grid)
    
    assert result.is_valid, (
        f"相位差小于阈值应通过验证: {result.message}"
    )


# ============================================================================
# 属性 5：能量守恒验证
# ============================================================================

@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    amplitude_scale=amplitude_strategy,
)
def test_property_5_energy_conservation_same_amplitude(
    grid_size: int,
    amplitude_scale: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 5: 能量守恒**
    **Validates: Requirements 5.1, 5.2**
    
    相同振幅分布应通过能量守恒验证。
    """
    # 创建振幅分布
    amplitude = np.ones((grid_size, grid_size)) * amplitude_scale
    
    validator = EnergyConservationValidator()
    result = validator.validate_energy_conservation(
        amplitude_before=amplitude,
        amplitude_after=amplitude,
    )
    
    assert result.is_valid, (
        f"相同振幅应通过能量守恒验证: {result.message}"
    )
    assert abs(result.details['relative_change']) < 1e-10, (
        f"相同振幅的能量变化应为零"
    )


@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    amplitude_scale=amplitude_strategy,
    loss_factor=st.floats(min_value=0.5, max_value=0.99),
)
def test_property_5_energy_loss_detection(
    grid_size: int,
    amplitude_scale: float,
    loss_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 5: 能量守恒**
    **Validates: Requirements 5.1, 5.2**
    
    能量损失应被正确检测。
    """
    # 创建振幅分布
    amplitude_before = np.ones((grid_size, grid_size)) * amplitude_scale
    # 振幅减小 -> 能量减小
    amplitude_after = amplitude_before * np.sqrt(loss_factor)
    
    validator = EnergyConservationValidator(tolerance=0.001)  # 严格容差
    result = validator.validate_energy_conservation(
        amplitude_before=amplitude_before,
        amplitude_after=amplitude_after,
    )
    
    # 能量变化应接近 loss_factor - 1
    expected_change = loss_factor - 1
    actual_change = result.details['relative_change']
    
    assert_allclose(
        actual_change,
        expected_change,
        rtol=1e-6,
        err_msg=f"能量变化计算不正确",
    )


@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    amplitude_scale=amplitude_strategy,
)
def test_property_5_pixel_area_scaling(
    grid_size: int,
    amplitude_scale: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 5: 能量守恒**
    **Validates: Requirements 5.1, 5.2**
    
    像素面积变化应正确考虑。
    """
    # 创建振幅分布
    amplitude_before = np.ones((grid_size, grid_size)) * amplitude_scale
    # 振幅减半，但像素面积增加 4 倍 -> 能量不变
    amplitude_after = amplitude_before / 2
    
    validator = EnergyConservationValidator()
    result = validator.validate_energy_conservation(
        amplitude_before=amplitude_before,
        amplitude_after=amplitude_after,
        pixel_area_before=1.0,
        pixel_area_after=4.0,
    )
    
    assert result.is_valid, (
        f"像素面积补偿后应通过能量守恒验证: {result.message}"
    )


# ============================================================================
# 雅可比振幅验证
# ============================================================================

@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
)
def test_jacobian_amplitude_formula(
    grid_size: int,
):
    """
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 5.1**
    
    验证雅可比振幅公式 A = 1/sqrt(|J|)。
    """
    # 创建雅可比行列式
    jacobian_det = np.random.uniform(0.5, 2.0, (grid_size, grid_size))
    
    # 使用正确公式计算振幅
    amplitude = 1.0 / np.sqrt(jacobian_det)
    
    # 创建有效掩模
    valid_mask = np.ones((grid_size, grid_size), dtype=bool)
    
    validator = EnergyConservationValidator()
    result = validator.validate_jacobian_amplitude(
        jacobian_det=jacobian_det,
        amplitude=amplitude,
        valid_mask=valid_mask,
    )
    
    assert result.is_valid, (
        f"正确公式计算的振幅应通过验证: {result.message}"
    )


# ============================================================================
# 波前误差范围验证
# ============================================================================

@settings(max_examples=100)
@given(
    grid_size=grid_size_strategy,
    error_scale=st.floats(min_value=0.01, max_value=0.5),
)
def test_wavefront_error_range(
    grid_size: int,
    error_scale: float,
):
    """
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 7.2**
    
    小波前误差应通过验证。
    """
    # 创建小波前误差
    wavefront_error = np.random.randn(grid_size, grid_size) * error_scale
    
    validator = PhaseContinuityValidator()
    result = validator.validate_wavefront_error_range(
        wavefront_error=wavefront_error,
        expected_max_rad=1.0,
    )
    
    if np.max(np.abs(wavefront_error)) < 1.0:
        assert result.is_valid, (
            f"小波前误差应通过验证: {result.message}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
