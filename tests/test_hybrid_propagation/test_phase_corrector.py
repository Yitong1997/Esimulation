"""
PhaseCorrector 单元测试

测试相位修正器的各项功能：
- 插值准确性
- 残差计算正确性
- 相位包裹功能

作者：混合光学仿真项目
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from hybrid_propagation.phase_correction import PhaseCorrector


class TestPhaseCorrectorInit:
    """测试 PhaseCorrector 初始化"""
    
    def test_init_with_valid_params(self):
        """测试使用有效参数初始化"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        reference_phase = 0.1 * (X**2 + Y**2)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        assert corrector.reference_phase is not None
        assert len(corrector.x_coords) == n
        assert len(corrector.y_coords) == n
    
    def test_interpolator_created(self):
        """测试插值器被正确创建"""
        n = 32
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        assert corrector._interpolator is not None


class TestInterpolateReferencePhase:
    """测试参考相位插值功能
    
    **Validates: Requirements 6.1**
    """
    
    def test_interpolate_at_grid_points(self):
        """测试在网格点位置的插值"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        reference_phase = 0.1 * (X**2 + Y**2)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在网格点位置插值
        ray_x = np.array([x[10], x[30], x[50]])
        ray_y = np.array([y[10], y[30], y[50]])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        # 在网格点位置，插值结果应该与网格值完全一致
        expected = np.array([
            reference_phase[10, 10],
            reference_phase[30, 30],
            reference_phase[50, 50],
        ])
        
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_interpolate_between_grid_points(self):
        """测试在网格点之间的插值"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        # 使用线性相位，便于验证插值
        reference_phase = 0.1 * X + 0.2 * Y
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在网格点之间插值
        ray_x = np.array([0.0, 1.5, -2.3])
        ray_y = np.array([0.0, 1.5, -2.3])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        # 对于线性相位，插值应该精确
        expected = 0.1 * ray_x + 0.2 * ray_y
        
        assert_allclose(result, expected, rtol=1e-6, atol=1e-15)
    
    def test_interpolate_outside_grid(self):
        """测试在网格外部的插值"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 在网格外部插值
        ray_x = np.array([15.0, -15.0])
        ray_y = np.array([15.0, -15.0])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        # 网格外部应该返回 NaN
        assert np.all(np.isnan(result))
    
    def test_interpolate_output_shape(self):
        """测试插值输出形状"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 测试不同形状的输入
        ray_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ray_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        result = corrector.interpolate_reference_phase(ray_x, ray_y)
        
        assert result.shape == ray_x.shape


class TestComputeResidualPhase:
    """测试残差相位计算功能
    
    **Validates: Requirements 6.2, 6.4**
    """
    
    def test_zero_residual(self):
        """测试零残差"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        ray_phase = np.array([0.0, 0.5, 1.0])
        ref_phase = np.array([0.0, 0.5, 1.0])
        
        residual = corrector.compute_residual_phase(ray_phase, ref_phase)
        
        assert_allclose(residual, 0.0, atol=1e-10)
    
    def test_small_residual(self):
        """测试小残差"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        ray_phase = np.array([0.1, 0.6, 1.1])
        ref_phase = np.array([0.0, 0.5, 1.0])
        
        residual = corrector.compute_residual_phase(ray_phase, ref_phase)
        
        assert_allclose(residual, 0.1, atol=1e-10)
    
    def test_phase_wrapping(self):
        """测试相位包裹"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 大残差应该被包裹
        ray_phase = np.array([4 * np.pi, -4 * np.pi, 2.5 * np.pi])
        ref_phase = np.array([0.0, 0.0, 0.0])
        
        residual = corrector.compute_residual_phase(ray_phase, ref_phase)
        
        # 包裹后应该在 [-π, π] 范围内
        assert np.all(residual >= -np.pi)
        assert np.all(residual <= np.pi)
    
    def test_wrap_phase_static_method(self):
        """测试静态方法 _wrap_phase"""
        # 测试各种相位值
        phases = np.array([0.0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi, -3*np.pi])
        
        wrapped = PhaseCorrector._wrap_phase(phases)
        
        # 所有值应该在 [-π, π] 范围内
        assert np.all(wrapped >= -np.pi - 1e-10)
        assert np.all(wrapped <= np.pi + 1e-10)
        
        # 0 应该保持为 0
        assert wrapped[0] == pytest.approx(0.0, abs=1e-10)
        
        # π 和 -π 应该相等（或接近）
        assert abs(wrapped[1]) == pytest.approx(np.pi, abs=1e-10)
        assert abs(wrapped[2]) == pytest.approx(np.pi, abs=1e-10)


class TestCorrectRayPhase:
    """测试光线相位修正功能
    
    **Validates: Requirements 6.3**
    """
    
    def test_zero_correction(self):
        """测试零修正"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        ray_opd_waves = np.array([1.0, 2.0, 3.0])
        residual_phase = np.array([0.0, 0.0, 0.0])
        
        corrected = corrector.correct_ray_phase(ray_opd_waves, residual_phase, 0.633)
        
        assert_allclose(corrected, ray_opd_waves, rtol=1e-10)
    
    def test_small_correction(self):
        """测试小修正"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        ray_opd_waves = np.array([1.0, 2.0, 3.0])
        # 残差相位 = 0.1 rad = 0.1/(2π) 波长
        residual_phase = np.array([0.1, 0.1, 0.1])
        
        corrected = corrector.correct_ray_phase(ray_opd_waves, residual_phase, 0.633)
        
        expected = ray_opd_waves - 0.1 / (2 * np.pi)
        assert_allclose(corrected, expected, rtol=1e-10)
    
    def test_one_wave_correction(self):
        """测试一个波长的修正"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        ray_opd_waves = np.array([1.0, 2.0, 3.0])
        # 残差相位 = 2π rad = 1 波长
        residual_phase = np.array([2*np.pi, 2*np.pi, 2*np.pi])
        
        corrected = corrector.correct_ray_phase(ray_opd_waves, residual_phase, 0.633)
        
        expected = ray_opd_waves - 1.0
        assert_allclose(corrected, expected, rtol=1e-10)


class TestCheckResidualRange:
    """测试残差范围检查功能
    
    **Validates: Requirements 6.5**
    """
    
    def test_valid_residual(self):
        """测试有效残差"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        residual = np.array([0.1, 0.2, 0.3])
        
        is_valid, warnings = corrector.check_residual_range(residual)
        
        assert is_valid == True
        assert len(warnings) == 0
    
    def test_nan_in_residual(self):
        """测试残差中有 NaN"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        residual = np.array([0.1, np.nan, 0.3])
        
        is_valid, warnings = corrector.check_residual_range(residual)
        
        assert is_valid == False
        assert len(warnings) >= 1
        assert any("NaN" in w for w in warnings)
    
    def test_large_residual(self):
        """测试大残差"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 接近 π 的残差
        residual = np.array([0.95 * np.pi, 0.1, 0.2])
        
        is_valid, warnings = corrector.check_residual_range(residual)
        
        assert is_valid == False
        assert len(warnings) >= 1
        assert any("接近" in w for w in warnings)


class TestCorrectRays:
    """测试完整的光线修正流程"""
    
    def test_complete_correction_flow(self):
        """测试完整的修正流程"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        # 简单的二次参考相位
        reference_phase = 0.01 * (X**2 + Y**2)
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 光线位置
        ray_x = np.array([0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 1.0, 2.0])
        
        # 光线 OPD（波长数）
        ray_opd_waves = np.array([0.0, 0.1, 0.2])
        
        corrected_opd, residual_phase, warnings = corrector.correct_rays(
            ray_x, ray_y, ray_opd_waves, 0.633
        )
        
        assert corrected_opd.shape == ray_opd_waves.shape
        assert residual_phase.shape == ray_opd_waves.shape
        assert isinstance(warnings, list)
    
    def test_correction_preserves_shape(self):
        """测试修正保持形状"""
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        reference_phase = np.zeros((n, n))
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 不同大小的输入
        for size in [10, 50, 100]:
            ray_x = np.random.uniform(-5, 5, size)
            ray_y = np.random.uniform(-5, 5, size)
            ray_opd_waves = np.random.uniform(0, 1, size)
            
            corrected_opd, residual_phase, _ = corrector.correct_rays(
                ray_x, ray_y, ray_opd_waves, 0.633
            )
            
            assert corrected_opd.shape == (size,)
            assert residual_phase.shape == (size,)
