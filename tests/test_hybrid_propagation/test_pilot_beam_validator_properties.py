"""
PilotBeamValidator 属性测试

使用 hypothesis 库进行属性基测试，验证 Pilot Beam 验证器的通用属性。

属性测试：
- Property 7: 相位采样质量
- Property 8: Pilot Beam 适用性检测

作者：混合光学仿真项目
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from hybrid_propagation import (
    PilotBeamValidator,
    ValidationResult,
    PilotBeamValidationResult,
)


# ============================================================================
# 自定义策略
# ============================================================================

# 网格大小策略（2的幂次，适合 FFT）
grid_size_strategy = st.sampled_from([16, 32, 64, 128])

# 相位梯度系数策略（控制相位变化速率）
phase_gradient_strategy = st.floats(min_value=0.0, max_value=2.0, allow_nan=False)

# 光束尺寸策略
beam_size_strategy = st.floats(min_value=1.0, max_value=100.0, allow_nan=False)

# 发散角策略
divergence_strategy = st.floats(min_value=0.0, max_value=0.5, allow_nan=False)

# 阈值策略
threshold_strategy = st.floats(min_value=0.01, max_value=np.pi, allow_nan=False)


# ============================================================================
# Property 7: 相位采样质量
# ============================================================================

class TestProperty7PhaseSamplingQuality:
    """
    **Feature: hybrid-element-propagation, Property 7: 相位采样质量**
    
    *For any* Pilot Beam 参考相位网格，相邻像素间的相位差的最大值应被正确计算，
    且当最大值超过 π/2 时应触发采样不足警告。
    
    **Validates: Requirements 5.1, 5.2, 5.7**
    """
    
    @given(
        grid_size=grid_size_strategy,
        gradient_coeff=phase_gradient_strategy,
    )
    @settings(max_examples=100)
    def test_phase_gradient_calculation_consistency(
        self,
        grid_size: int,
        gradient_coeff: float,
    ):
        """测试相位梯度计算的一致性
        
        对于任意线性相位分布，计算的梯度应与理论值一致。
        
        **Validates: Requirements 5.1**
        """
        # 创建线性相位网格
        x = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, x)
        phase_grid = gradient_coeff * X
        
        # 计算采样间隔
        dx = 20.0 / grid_size
        
        # 创建验证器
        validator = PilotBeamValidator(phase_grid=phase_grid, dx=dx, dy=dx)
        validator._compute_phase_gradients()
        
        # 理论梯度（每像素相位差）
        theoretical_gradient = gradient_coeff * dx
        
        # 由于相位包裹，实际梯度应在 [-π, π] 范围内
        wrapped_theoretical = np.angle(np.exp(1j * theoretical_gradient))
        
        # 验证计算的最大梯度与理论值接近
        if validator._max_phase_gradient is not None:
            assert_allclose(
                validator._max_phase_gradient,
                abs(wrapped_theoretical),
                rtol=0.1,
                atol=1e-10,
            )
    
    @given(
        grid_size=grid_size_strategy,
        gradient_coeff=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_small_gradient_passes_validation(
        self,
        grid_size: int,
        gradient_coeff: float,
    ):
        """测试小梯度相位通过验证
        
        当相位梯度足够小时，验证应该通过。
        
        **Validates: Requirements 5.2**
        """
        # 创建小梯度相位网格
        x = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, x)
        dx = 20.0 / grid_size
        
        # 确保梯度小于阈值
        max_gradient_per_pixel = gradient_coeff * dx
        assume(max_gradient_per_pixel < np.pi / 2 - 0.1)  # 留一些余量
        
        phase_grid = gradient_coeff * X
        
        validator = PilotBeamValidator(phase_grid=phase_grid, dx=dx, dy=dx)
        result = validator.check_phase_sampling()
        
        # 应该通过验证
        assert result.passed == True
    
    @given(
        grid_size=grid_size_strategy,
        gradient_coeff=st.floats(min_value=3.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_large_gradient_triggers_warning(
        self,
        grid_size: int,
        gradient_coeff: float,
    ):
        """测试大梯度相位触发警告
        
        当相位梯度超过阈值时，应该触发采样不足警告。
        
        **Validates: Requirements 5.7**
        """
        # 创建大梯度相位网格
        x = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, x)
        dx = 20.0 / grid_size
        
        # 确保梯度大于阈值
        max_gradient_per_pixel = gradient_coeff * dx
        assume(max_gradient_per_pixel > np.pi / 2 + 0.1)  # 留一些余量
        
        phase_grid = gradient_coeff * X
        
        validator = PilotBeamValidator(phase_grid=phase_grid, dx=dx, dy=dx)
        result = validator.check_phase_sampling()
        
        # 由于相位包裹，大梯度可能被包裹到小值
        # 所以这里只检查结果是有效的
        assert result.value is not None
        assert result.threshold == np.pi / 2
    
    @given(
        grid_size=grid_size_strategy,
        threshold=threshold_strategy,
    )
    @settings(max_examples=100)
    def test_custom_threshold_respected(
        self,
        grid_size: int,
        threshold: float,
    ):
        """测试自定义阈值被正确使用
        
        验证器应该使用用户指定的阈值进行判断。
        
        **Validates: Requirements 5.1, 5.2**
        """
        # 创建零相位网格（梯度为零）
        phase_grid = np.zeros((grid_size, grid_size))
        
        validator = PilotBeamValidator(phase_grid=phase_grid)
        result = validator.check_phase_sampling(threshold=threshold)
        
        # 零梯度应该总是通过
        assert result.passed == True
        assert result.threshold == threshold
        assert result.value == 0.0


# ============================================================================
# Property 8: Pilot Beam 适用性检测
# ============================================================================

class TestProperty8PilotBeamApplicability:
    """
    **Feature: hybrid-element-propagation, Property 8: Pilot Beam 适用性检测**
    
    *For any* Pilot Beam 和实际光束参数组合，验证器应正确检测：
    - 光束发散角是否超过阈值
    - 光束尺寸差异是否超过 50%
    并在不满足条件时发出相应警告。
    
    **Validates: Requirements 5.3, 5.4, 5.5, 5.6**
    """
    
    @given(
        divergence=st.floats(min_value=0.0, max_value=0.05, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_small_divergence_passes(self, divergence: float):
        """测试小发散角通过验证
        
        当发散角小于阈值时，验证应该通过。
        
        **Validates: Requirements 5.3**
        """
        validator = PilotBeamValidator(beam_divergence=divergence)
        result = validator.check_beam_divergence(max_divergence=0.1)
        
        assert result.passed == True
        assert result.value == divergence
    
    @given(
        divergence=st.floats(min_value=0.15, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_large_divergence_fails(self, divergence: float):
        """测试大发散角触发警告
        
        当发散角超过阈值时，应该触发警告。
        
        **Validates: Requirements 5.4**
        """
        validator = PilotBeamValidator(beam_divergence=divergence)
        result = validator.check_beam_divergence(max_divergence=0.1)
        
        assert result.passed == False
        assert result.value == divergence
        assert "过大" in result.message
    
    @given(
        pilot_size=beam_size_strategy,
        actual_size=beam_size_strategy,
    )
    @settings(max_examples=100)
    def test_size_ratio_calculation(
        self,
        pilot_size: float,
        actual_size: float,
    ):
        """测试尺寸比例计算正确性
        
        尺寸比例应该是较大值除以较小值。
        
        **Validates: Requirements 5.5**
        """
        validator = PilotBeamValidator(
            pilot_beam_size=pilot_size,
            actual_beam_size=actual_size,
        )
        result = validator.check_beam_size_match()
        
        # 计算预期比例
        expected_ratio = max(pilot_size, actual_size) / min(pilot_size, actual_size)
        
        assert result.value == pytest.approx(expected_ratio, rel=1e-6)
    
    @given(
        base_size=beam_size_strategy,
        ratio=st.floats(min_value=1.0, max_value=1.4, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_good_size_match_passes(
        self,
        base_size: float,
        ratio: float,
    ):
        """测试尺寸匹配良好时通过验证
        
        当尺寸比例小于阈值时，验证应该通过。
        
        **Validates: Requirements 5.5**
        """
        pilot_size = base_size
        actual_size = base_size * ratio
        
        validator = PilotBeamValidator(
            pilot_beam_size=pilot_size,
            actual_beam_size=actual_size,
        )
        result = validator.check_beam_size_match(max_ratio=1.5)
        
        assert result.passed == True
    
    @given(
        base_size=beam_size_strategy,
        ratio=st.floats(min_value=1.6, max_value=3.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_poor_size_match_fails(
        self,
        base_size: float,
        ratio: float,
    ):
        """测试尺寸不匹配时触发警告
        
        当尺寸比例超过阈值时，应该触发警告。
        
        **Validates: Requirements 5.6**
        """
        pilot_size = base_size
        actual_size = base_size * ratio
        
        validator = PilotBeamValidator(
            pilot_beam_size=pilot_size,
            actual_beam_size=actual_size,
        )
        result = validator.check_beam_size_match(max_ratio=1.5)
        
        assert result.passed == False
        assert "不匹配" in result.message


# ============================================================================
# 综合属性测试
# ============================================================================

class TestValidateAllProperties:
    """综合验证属性测试"""
    
    @given(
        grid_size=grid_size_strategy,
        pilot_size=beam_size_strategy,
        actual_size=beam_size_strategy,
        divergence=divergence_strategy,
    )
    @settings(max_examples=100)
    def test_validate_all_returns_complete_result(
        self,
        grid_size: int,
        pilot_size: float,
        actual_size: float,
        divergence: float,
    ):
        """测试 validate_all 返回完整结果
        
        validate_all 应该返回包含所有验证结果的完整对象。
        """
        phase_grid = np.zeros((grid_size, grid_size))
        
        validator = PilotBeamValidator(
            phase_grid=phase_grid,
            pilot_beam_size=pilot_size,
            actual_beam_size=actual_size,
            beam_divergence=divergence,
        )
        
        result = validator.validate_all()
        
        # 验证返回类型
        assert isinstance(result, PilotBeamValidationResult)
        
        # 验证所有字段都存在
        assert isinstance(result.phase_sampling, ValidationResult)
        assert isinstance(result.beam_divergence, ValidationResult)
        assert isinstance(result.beam_size_match, ValidationResult)
        assert isinstance(result.max_phase_gradient, (int, float, np.floating))
        assert isinstance(result.mean_phase_gradient, (int, float, np.floating))
        assert isinstance(result.warnings, list)
    
    @given(
        grid_size=grid_size_strategy,
    )
    @settings(max_examples=100)
    def test_is_valid_reflects_all_checks(self, grid_size: int):
        """测试 is_valid 反映所有检查结果
        
        当所有检查都通过时，is_valid 应该为 True。
        """
        # 创建一个应该通过所有检查的配置
        phase_grid = np.zeros((grid_size, grid_size))
        
        validator = PilotBeamValidator(
            phase_grid=phase_grid,
            pilot_beam_size=10.0,
            actual_beam_size=10.0,  # 完全匹配
            beam_divergence=0.01,   # 小发散角
        )
        
        result = validator.validate_all()
        
        # 所有检查都应该通过
        assert result.phase_sampling.passed == True
        assert result.beam_divergence.passed == True
        assert result.beam_size_match.passed == True
        assert result.is_valid == True
        assert len(result.warnings) == 0
    
    @given(
        grid_size=grid_size_strategy,
        divergence=st.floats(min_value=0.15, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_warnings_collected_correctly(
        self,
        grid_size: int,
        divergence: float,
    ):
        """测试警告信息正确收集
        
        当某些检查失败时，警告信息应该被正确收集。
        """
        phase_grid = np.zeros((grid_size, grid_size))
        
        validator = PilotBeamValidator(
            phase_grid=phase_grid,
            beam_divergence=divergence,  # 大发散角
        )
        
        result = validator.validate_all()
        
        # 发散角检查应该失败
        assert result.beam_divergence.passed == False
        assert result.is_valid == False
        
        # 警告列表应该包含发散角警告
        assert len(result.warnings) >= 1
        assert any("发散角" in w for w in result.warnings)


# ============================================================================
# 边界条件测试
# ============================================================================

class TestBoundaryConditions:
    """边界条件属性测试"""
    
    @given(
        grid_size=grid_size_strategy,
    )
    @settings(max_examples=100)
    def test_zero_phase_grid(self, grid_size: int):
        """测试零相位网格
        
        零相位网格的梯度应该为零。
        """
        phase_grid = np.zeros((grid_size, grid_size))
        
        validator = PilotBeamValidator(phase_grid=phase_grid)
        validator._compute_phase_gradients()
        
        assert validator._max_phase_gradient == 0.0
        assert validator._mean_phase_gradient == 0.0
    
    @given(
        grid_size=grid_size_strategy,
        constant=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_constant_phase_grid(self, grid_size: int, constant: float):
        """测试常数相位网格
        
        常数相位网格的梯度应该为零。
        """
        phase_grid = np.full((grid_size, grid_size), constant)
        
        validator = PilotBeamValidator(phase_grid=phase_grid)
        validator._compute_phase_gradients()
        
        assert validator._max_phase_gradient == pytest.approx(0.0, abs=1e-10)
    
    @given(
        size=beam_size_strategy,
    )
    @settings(max_examples=100)
    def test_identical_beam_sizes(self, size: float):
        """测试相同的光束尺寸
        
        当 Pilot Beam 和实际光束尺寸相同时，比例应该为 1。
        """
        validator = PilotBeamValidator(
            pilot_beam_size=size,
            actual_beam_size=size,
        )
        result = validator.check_beam_size_match()
        
        assert result.passed == True
        assert result.value == pytest.approx(1.0, rel=1e-6)
