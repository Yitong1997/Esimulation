"""
AmplitudeReconstructor 单元测试

测试复振幅重建器的各项功能：
- 网格插值功能
- 参考相位加回功能
- 无效区域处理

作者：混合光学仿真项目
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from hybrid_propagation.amplitude_reconstruction import AmplitudeReconstructor


class TestAmplitudeReconstructorInit:
    """测试 AmplitudeReconstructor 初始化"""
    
    def test_init_with_valid_params(self):
        """测试使用有效参数初始化"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        assert reconstructor.grid_size == 64
        assert reconstructor.physical_size == 20.0
        assert reconstructor.wavelength == 0.633
    
    def test_grid_coordinates_created(self):
        """测试网格坐标被正确创建"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        assert len(reconstructor.x_grid) == 64
        assert len(reconstructor.y_grid) == 64
        assert reconstructor.X_grid.shape == (64, 64)
        assert reconstructor.Y_grid.shape == (64, 64)
        
        # 检查范围
        assert reconstructor.x_grid[0] == pytest.approx(-10.0, rel=1e-6)
        assert reconstructor.x_grid[-1] == pytest.approx(10.0, rel=1e-6)


class TestInterpolateToGrid:
    """测试网格插值功能
    
    **Validates: Requirements 7.2**
    """
    
    def test_interpolate_uniform_values(self):
        """测试均匀值的插值"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        # 均匀分布的光线
        n_rays = 100
        ray_x = np.random.uniform(-8, 8, n_rays)
        ray_y = np.random.uniform(-8, 8, n_rays)
        ray_values = np.ones(n_rays)  # 均匀值
        
        result = reconstructor._interpolate_to_grid(ray_x, ray_y, ray_values)
        
        # 在光线覆盖区域内，值应该接近 1
        # 边缘可能有 NaN
        valid_mask = ~np.isnan(result)
        if np.any(valid_mask):
            assert_allclose(result[valid_mask], 1.0, rtol=0.1)
    
    def test_interpolate_linear_values(self):
        """测试线性值的插值"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        # 规则网格上的光线
        ray_x = np.linspace(-8, 8, 20)
        ray_y = np.linspace(-8, 8, 20)
        ray_X, ray_Y = np.meshgrid(ray_x, ray_y)
        ray_x_flat = ray_X.ravel()
        ray_y_flat = ray_Y.ravel()
        
        # 线性值
        ray_values = 0.1 * ray_x_flat + 0.2 * ray_y_flat
        
        result = reconstructor._interpolate_to_grid(ray_x_flat, ray_y_flat, ray_values)
        
        # 在光线覆盖区域内，插值应该接近线性
        valid_mask = ~np.isnan(result)
        if np.any(valid_mask):
            expected = 0.1 * reconstructor.X_grid + 0.2 * reconstructor.Y_grid
            # 只比较有效区域
            assert_allclose(result[valid_mask], expected[valid_mask], rtol=0.1)
    
    def test_interpolate_output_shape(self):
        """测试插值输出形状"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        # 使用更多的非共线点
        ray_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        ray_values = np.ones(9)
        
        result = reconstructor._interpolate_to_grid(ray_x, ray_y, ray_values)
        
        assert result.shape == (64, 64)


class TestApplyReferencePhase:
    """测试参考相位加回功能
    
    **Validates: Requirements 7.3**
    """
    
    def test_apply_zero_reference_phase(self):
        """测试加回零参考相位"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        residual_phase = np.ones((32, 32)) * 0.5
        reference_phase = np.zeros((32, 32))
        
        result = reconstructor._apply_reference_phase(residual_phase, reference_phase)
        
        assert_allclose(result, residual_phase, rtol=1e-10)
    
    def test_apply_nonzero_reference_phase(self):
        """测试加回非零参考相位"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        residual_phase = np.ones((32, 32)) * 0.5
        reference_phase = np.ones((32, 32)) * 0.3
        
        result = reconstructor._apply_reference_phase(residual_phase, reference_phase)
        
        expected = residual_phase + reference_phase
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_apply_reference_phase_different_size(self):
        """测试不同大小的参考相位"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        residual_phase = np.zeros((64, 64))
        # 不同大小的参考相位
        reference_phase = np.ones((32, 32)) * 0.5
        
        result = reconstructor._apply_reference_phase(residual_phase, reference_phase)
        
        # 应该被重采样到正确大小
        assert result.shape == (64, 64)


class TestReconstruct:
    """测试完整重建功能
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    
    def test_reconstruct_output_shape(self):
        """测试重建输出形状"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        n_rays = 100
        ray_x = np.random.uniform(-8, 8, n_rays)
        ray_y = np.random.uniform(-8, 8, n_rays)
        ray_intensity = np.ones(n_rays)
        ray_opd_waves = np.zeros(n_rays)
        reference_phase = np.zeros((64, 64))
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x, ray_y, ray_intensity, ray_opd_waves,
            reference_phase, valid_mask
        )
        
        assert result.shape == (64, 64)
        assert result.dtype == np.complex128
    
    def test_reconstruct_with_no_valid_rays(self):
        """测试没有有效光线时的重建"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        ray_x = np.array([0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 1.0, 2.0])
        ray_intensity = np.array([1.0, 1.0, 1.0])
        ray_opd_waves = np.array([0.0, 0.0, 0.0])
        reference_phase = np.zeros((32, 32))
        valid_mask = np.array([False, False, False])  # 没有有效光线
        
        with pytest.warns(UserWarning, match="没有有效光线"):
            result = reconstructor.reconstruct(
                ray_x, ray_y, ray_intensity, ray_opd_waves,
                reference_phase, valid_mask
            )
        
        # 应该返回零数组
        assert_allclose(result, 0.0)
    
    def test_reconstruct_uniform_amplitude(self):
        """测试均匀振幅的重建"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        # 规则网格上的光线
        ray_x = np.linspace(-8, 8, 20)
        ray_y = np.linspace(-8, 8, 20)
        ray_X, ray_Y = np.meshgrid(ray_x, ray_y)
        ray_x_flat = ray_X.ravel()
        ray_y_flat = ray_Y.ravel()
        
        n_rays = len(ray_x_flat)
        ray_intensity = np.ones(n_rays)
        ray_opd_waves = np.zeros(n_rays)
        reference_phase = np.zeros((32, 32))
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_flat, ray_y_flat, ray_intensity, ray_opd_waves,
            reference_phase, valid_mask
        )
        
        # 在光线覆盖区域内，振幅应该接近 1
        amplitude = np.abs(result)
        valid_region = amplitude > 0.1
        if np.any(valid_region):
            assert_allclose(amplitude[valid_region], 1.0, rtol=0.2)
    
    def test_reconstruct_with_phase(self):
        """测试带相位的重建"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        # 规则网格上的光线
        ray_x = np.linspace(-8, 8, 20)
        ray_y = np.linspace(-8, 8, 20)
        ray_X, ray_Y = np.meshgrid(ray_x, ray_y)
        ray_x_flat = ray_X.ravel()
        ray_y_flat = ray_Y.ravel()
        
        n_rays = len(ray_x_flat)
        ray_intensity = np.ones(n_rays)
        # 非零 OPD
        ray_opd_waves = 0.1 * np.ones(n_rays)
        reference_phase = np.zeros((32, 32))
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_flat, ray_y_flat, ray_intensity, ray_opd_waves,
            reference_phase, valid_mask
        )
        
        # 相位应该不为零
        phase = np.angle(result)
        valid_region = np.abs(result) > 0.1
        if np.any(valid_region):
            # 相位应该接近 2π × 0.1 = 0.628 rad
            expected_phase = 2 * np.pi * 0.1
            assert np.mean(np.abs(phase[valid_region])) > 0.1


class TestGetCoverageMask:
    """测试覆盖区域掩模功能
    
    **Validates: Requirements 7.5**
    """
    
    def test_coverage_mask_shape(self):
        """测试覆盖掩模形状"""
        reconstructor = AmplitudeReconstructor(
            grid_size=64,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        ray_x = np.array([0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 1.0, 2.0])
        valid_mask = np.array([True, True, True])
        
        result = reconstructor.get_coverage_mask(ray_x, ray_y, valid_mask)
        
        assert result.shape == (64, 64)
        assert result.dtype == bool
    
    def test_coverage_mask_no_valid_rays(self):
        """测试没有有效光线时的覆盖掩模"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        ray_x = np.array([0.0, 1.0, 2.0])
        ray_y = np.array([0.0, 1.0, 2.0])
        valid_mask = np.array([False, False, False])
        
        result = reconstructor.get_coverage_mask(ray_x, ray_y, valid_mask)
        
        # 应该全为 False
        assert not np.any(result)
    
    def test_coverage_mask_with_margin(self):
        """测试带边缘余量的覆盖掩模"""
        reconstructor = AmplitudeReconstructor(
            grid_size=32,
            physical_size=20.0,
            wavelength=0.633,
        )
        
        ray_x = np.array([0.0])
        ray_y = np.array([0.0])
        valid_mask = np.array([True])
        
        # 无余量
        mask_no_margin = reconstructor.get_coverage_mask(ray_x, ray_y, valid_mask, margin=0.0)
        
        # 有余量
        mask_with_margin = reconstructor.get_coverage_mask(ray_x, ray_y, valid_mask, margin=5.0)
        
        # 有余量的掩模应该覆盖更大区域
        assert np.sum(mask_with_margin) > np.sum(mask_no_margin)
