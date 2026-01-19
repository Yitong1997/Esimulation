"""
PhaseCorrector 属性测试

使用 hypothesis 库进行属性基测试，验证相位修正器的通用属性。

属性测试：
- Property 9: 残差相位包裹
- Property 10: 相位插值一致性

作者：混合光学仿真项目
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from hybrid_propagation.phase_correction import PhaseCorrector


# ============================================================================
# 自定义策略
# ============================================================================

# 网格大小策略
grid_size_strategy = st.sampled_from([16, 32, 64])

# 相位值策略
phase_strategy = st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False)

# 坐标策略（在网格范围内）
coord_strategy = st.floats(min_value=-9.0, max_value=9.0, allow_nan=False)


# ============================================================================
# Property 9: 残差相位包裹
# ============================================================================

class TestProperty9ResidualPhaseWrapping:
    """
    **Feature: hybrid-element-propagation, Property 9: 残差相位包裹**
    
    *For any* 计算得到的残差相位值，应被正确包裹到 [-π, π] 范围内。
    
    **Validates: Requirements 6.4**
    """
    
    @given(
        ray_phase=st.lists(phase_strategy, min_size=1, max_size=100),
        ref_phase=st.lists(phase_strategy, min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_residual_always_in_range(
        self,
        ray_phase: list,
        ref_phase: list,
    ):
        """测试残差相位总是在 [-π, π] 范围内
        
        **Validates: Requirements 6.4**
        """
        # 确保两个列表长度相同
        min_len = min(len(ray_phase), len(ref_phase))
        ray_phase = np.array(ray_phase[:min_len])
        ref_phase = np.array(ref_phase[:min_len])
        
        # 创建一个简单的修正器
        n = 32
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 计算残差
        residual = corrector.compute_residual_phase(ray_phase, ref_phase)
        
        # 验证范围
        assert np.all(residual >= -np.pi - 1e-10)
        assert np.all(residual <= np.pi + 1e-10)
    
    @given(
        phase=phase_strategy,
    )
    @settings(max_examples=100)
    def test_wrap_phase_idempotent(self, phase: float):
        """测试相位包裹的幂等性
        
        包裹两次应该与包裹一次结果相同。
        """
        phase_array = np.array([phase])
        
        wrapped_once = PhaseCorrector._wrap_phase(phase_array)
        wrapped_twice = PhaseCorrector._wrap_phase(wrapped_once)
        
        assert_allclose(wrapped_once, wrapped_twice, rtol=1e-10)
    
    @given(
        phase=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_wrap_phase_preserves_valid_range(self, phase: float):
        """测试已在有效范围内的相位保持不变
        
        如果相位已经在 [-π, π] 范围内，包裹后应该保持不变。
        """
        phase_array = np.array([phase])
        
        wrapped = PhaseCorrector._wrap_phase(phase_array)
        
        assert_allclose(wrapped, phase_array, rtol=1e-10)


# ============================================================================
# Property 10: 相位插值一致性
# ============================================================================

class TestProperty10PhaseInterpolationConsistency:
    """
    **Feature: hybrid-element-propagation, Property 10: 相位插值一致性**
    
    *For any* 参考相位网格和光线位置，在网格节点位置的插值结果应与网格值完全一致
    （误差 < 1e-10）。
    
    **Validates: Requirements 6.1**
    """
    
    @given(
        grid_size=grid_size_strategy,
        i=st.integers(min_value=1, max_value=30),
        j=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=100)
    def test_interpolation_at_grid_nodes(
        self,
        grid_size: int,
        i: int,
        j: int,
    ):
        """测试在网格节点位置的插值精度
        
        **Validates: Requirements 6.1**
        """
        # 确保索引在有效范围内
        i = i % grid_size
        j = j % grid_size
        
        # 创建参考相位网格
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, y)
        reference_phase = 0.1 * (X**2 + Y**2)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在网格节点位置插值
        ray_x = np.array([x[j]])
        ray_y = np.array([y[i]])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        expected = reference_phase[i, j]
        
        assert_allclose(result[0], expected, rtol=1e-10)
    
    @given(
        grid_size=grid_size_strategy,
        slope_x=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        slope_y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        ray_x=coord_strategy,
        ray_y=coord_strategy,
    )
    @settings(max_examples=100)
    def test_linear_interpolation_exact(
        self,
        grid_size: int,
        slope_x: float,
        slope_y: float,
        ray_x: float,
        ray_y: float,
    ):
        """测试线性相位的插值精度
        
        对于线性相位分布，双线性插值应该是精确的。
        """
        # 创建线性参考相位网格
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, y)
        reference_phase = slope_x * X + slope_y * Y
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在任意位置插值
        ray_x_arr = np.array([ray_x])
        ray_y_arr = np.array([ray_y])
        
        result = corrector.interpolate_reference_phase(ray_x_arr, ray_y_arr)
        expected = slope_x * ray_x + slope_y * ray_y
        
        assert_allclose(result[0], expected, rtol=1e-6, atol=1e-15)
    
    @given(
        grid_size=grid_size_strategy,
        num_rays=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_interpolation_output_shape(
        self,
        grid_size: int,
        num_rays: int,
    ):
        """测试插值输出形状正确
        
        输出数组形状应该与输入光线坐标数组形状相同。
        """
        # 创建参考相位网格
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        reference_phase = np.zeros((grid_size, grid_size))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 随机光线位置
        ray_x = np.random.uniform(-9, 9, num_rays)
        ray_y = np.random.uniform(-9, 9, num_rays)
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        assert result.shape == (num_rays,)


# ============================================================================
# 综合属性测试
# ============================================================================

class TestCorrectionProperties:
    """相位修正综合属性测试"""
    
    @given(
        opd_waves=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_subnormal=False),
        residual_phase=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_subnormal=False),
    )
    @settings(max_examples=100)
    def test_correction_reversibility(
        self,
        opd_waves: float,
        residual_phase: float,
    ):
        """测试修正的可逆性
        
        修正后再加回残差应该得到原始 OPD。
        """
        # 跳过极小值
        assume(abs(opd_waves) > 1e-15 or opd_waves == 0.0)
        
        n = 32
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        opd_array = np.array([opd_waves])
        residual_array = np.array([residual_phase])
        
        # 修正
        corrected = corrector.correct_ray_phase(opd_array, residual_array, 0.633)
        
        # 加回残差
        restored = corrected + residual_phase / (2 * np.pi)
        
        assert_allclose(restored, opd_array, rtol=1e-10, atol=1e-15)
    
    @given(
        grid_size=grid_size_strategy,
        num_rays=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_complete_flow_output_types(
        self,
        grid_size: int,
        num_rays: int,
    ):
        """测试完整流程的输出类型
        
        correct_rays 应该返回正确类型的输出。
        """
        # 创建参考相位网格
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        X, Y = np.meshgrid(x, y)
        reference_phase = 0.01 * (X**2 + Y**2)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 随机光线
        ray_x = np.random.uniform(-9, 9, num_rays)
        ray_y = np.random.uniform(-9, 9, num_rays)
        ray_opd = np.random.uniform(-1, 1, num_rays)
        
        corrected_opd, residual_phase, warnings = corrector.correct_rays(
            ray_x, ray_y, ray_opd, 0.633
        )
        
        # 验证输出类型
        assert isinstance(corrected_opd, np.ndarray)
        assert isinstance(residual_phase, np.ndarray)
        assert isinstance(warnings, list)
        
        # 验证输出形状
        assert corrected_opd.shape == (num_rays,)
        assert residual_phase.shape == (num_rays,)
        
        # 验证残差在有效范围内
        assert np.all(residual_phase >= -np.pi - 1e-10)
        assert np.all(residual_phase <= np.pi + 1e-10)


# ============================================================================
# 边界条件测试
# ============================================================================

class TestBoundaryConditions:
    """边界条件属性测试"""
    
    @given(
        grid_size=grid_size_strategy,
    )
    @settings(max_examples=100)
    def test_zero_reference_phase(self, grid_size: int):
        """测试零参考相位
        
        当参考相位为零时，残差应该等于原始相位（包裹后）。
        """
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        reference_phase = np.zeros((grid_size, grid_size))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在网格内的位置
        ray_x = np.array([0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 1.0, 2.0])
        ray_opd = np.array([0.1, 0.2, 0.3])
        
        corrected_opd, residual_phase, _ = corrector.correct_rays(
            ray_x, ray_y, ray_opd, 0.633
        )
        
        # 参考相位为零，所以残差 = 原始相位
        expected_residual = PhaseCorrector._wrap_phase(2 * np.pi * ray_opd)
        
        assert_allclose(residual_phase, expected_residual, rtol=1e-10)
    
    @given(
        grid_size=grid_size_strategy,
        constant=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_subnormal=False),
    )
    @settings(max_examples=100)
    def test_constant_reference_phase(
        self,
        grid_size: int,
        constant: float,
    ):
        """测试常数参考相位
        
        当参考相位为常数时，所有位置的插值结果应该相同。
        """
        # 跳过极小值
        assume(abs(constant) > 1e-15 or constant == 0.0)
        
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-10, 10, grid_size)
        reference_phase = np.full((grid_size, grid_size), constant)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 多个位置
        ray_x = np.array([0.0, 1.0, 2.0, -1.0, -2.0])
        ray_y = np.array([0.0, 1.0, 2.0, -1.0, -2.0])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        # 所有结果应该相同
        assert_allclose(result, constant, rtol=1e-10, atol=1e-15)
