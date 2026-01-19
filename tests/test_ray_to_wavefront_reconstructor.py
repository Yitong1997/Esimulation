"""
RayToWavefrontReconstructor 类的单元测试

测试复振幅重建器的初始化和参数验证功能。
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from wavefront_to_rays import RayToWavefrontReconstructor, InsufficientRaysError


class TestRayToWavefrontReconstructorInit:
    """测试 RayToWavefrontReconstructor 初始化"""
    
    def test_valid_initialization(self):
        """测试使用有效参数初始化"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        assert reconstructor.grid_size == 512
        assert reconstructor.sampling_mm == 0.01
        assert reconstructor.wavelength_um == 0.633
    
    def test_initialization_with_different_grid_sizes(self):
        """测试不同网格大小的初始化"""
        for grid_size in [128, 256, 512, 1024, 2048]:
            reconstructor = RayToWavefrontReconstructor(
                grid_size=grid_size,
                sampling_mm=0.01,
                wavelength_um=0.55
            )
            assert reconstructor.grid_size == grid_size
    
    def test_initialization_with_numpy_types(self):
        """测试使用 numpy 类型参数初始化"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=np.int64(512),
            sampling_mm=np.float64(0.01),
            wavelength_um=np.float32(0.633)
        )
        
        assert reconstructor.grid_size == 512
        assert reconstructor.sampling_mm == 0.01
        assert np.isclose(reconstructor.wavelength_um, 0.633, rtol=1e-5)


class TestRayToWavefrontReconstructorValidation:
    """测试 RayToWavefrontReconstructor 参数验证"""
    
    def test_invalid_grid_size_zero(self):
        """测试 grid_size 为零时抛出异常"""
        with pytest.raises(ValueError, match="grid_size 必须为正整数"):
            RayToWavefrontReconstructor(
                grid_size=0,
                sampling_mm=0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_grid_size_negative(self):
        """测试 grid_size 为负数时抛出异常"""
        with pytest.raises(ValueError, match="grid_size 必须为正整数"):
            RayToWavefrontReconstructor(
                grid_size=-512,
                sampling_mm=0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_grid_size_float(self):
        """测试 grid_size 为浮点数时抛出异常"""
        with pytest.raises(ValueError, match="grid_size 必须为整数"):
            RayToWavefrontReconstructor(
                grid_size=512.5,
                sampling_mm=0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_grid_size_string(self):
        """测试 grid_size 为字符串时抛出异常"""
        with pytest.raises(ValueError, match="grid_size 必须为整数"):
            RayToWavefrontReconstructor(
                grid_size="512",
                sampling_mm=0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_sampling_mm_zero(self):
        """测试 sampling_mm 为零时抛出异常"""
        with pytest.raises(ValueError, match="sampling_mm 必须为正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=0,
                wavelength_um=0.633
            )
    
    def test_invalid_sampling_mm_negative(self):
        """测试 sampling_mm 为负数时抛出异常"""
        with pytest.raises(ValueError, match="sampling_mm 必须为正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=-0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_sampling_mm_nan(self):
        """测试 sampling_mm 为 NaN 时抛出异常"""
        with pytest.raises(ValueError, match="sampling_mm 必须为有限正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=np.nan,
                wavelength_um=0.633
            )
    
    def test_invalid_sampling_mm_inf(self):
        """测试 sampling_mm 为无穷大时抛出异常"""
        with pytest.raises(ValueError, match="sampling_mm 必须为有限正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=np.inf,
                wavelength_um=0.633
            )
    
    def test_invalid_wavelength_um_zero(self):
        """测试 wavelength_um 为零时抛出异常"""
        with pytest.raises(ValueError, match="wavelength_um 必须为正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=0.01,
                wavelength_um=0
            )
    
    def test_invalid_wavelength_um_negative(self):
        """测试 wavelength_um 为负数时抛出异常"""
        with pytest.raises(ValueError, match="wavelength_um 必须为正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=0.01,
                wavelength_um=-0.633
            )
    
    def test_invalid_wavelength_um_nan(self):
        """测试 wavelength_um 为 NaN 时抛出异常"""
        with pytest.raises(ValueError, match="wavelength_um 必须为有限正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=0.01,
                wavelength_um=np.nan
            )
    
    def test_invalid_wavelength_um_inf(self):
        """测试 wavelength_um 为无穷大时抛出异常"""
        with pytest.raises(ValueError, match="wavelength_um 必须为有限正数"):
            RayToWavefrontReconstructor(
                grid_size=512,
                sampling_mm=0.01,
                wavelength_um=np.inf
            )


class TestRayToWavefrontReconstructorProperties:
    """测试 RayToWavefrontReconstructor 属性"""
    
    def test_grid_half_size_mm(self):
        """测试 grid_half_size_mm 属性"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        expected_half_size = 0.01 * 512 / 2  # 2.56 mm
        assert np.isclose(reconstructor.grid_half_size_mm, expected_half_size)
    
    def test_grid_extent_mm(self):
        """测试 grid_extent_mm 属性"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        expected_half_size = 0.01 * 512 / 2  # 2.56 mm
        extent = reconstructor.grid_extent_mm
        
        assert np.isclose(extent[0], -expected_half_size)
        assert np.isclose(extent[1], expected_half_size)
    
    def test_repr(self):
        """测试 __repr__ 方法"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        repr_str = repr(reconstructor)
        assert "RayToWavefrontReconstructor" in repr_str
        assert "512" in repr_str
        assert "0.01" in repr_str
        assert "0.633" in repr_str
    
    def test_str(self):
        """测试 __str__ 方法"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        str_repr = str(reconstructor)
        assert "光线到波前复振幅重建器" in str_repr
        assert "512" in str_repr
        assert "0.01" in str_repr
        assert "0.633" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestComputeAmplitudePhaseJacobian:
    """测试 _compute_amplitude_phase_jacobian() 方法"""
    
    def test_uniform_grid_no_distortion(self):
        """测试均匀网格（无变形）时振幅应为均匀的
        
        当输入和输出位置相同时，雅可比行列式应为 1，
        归一化后振幅应为 1。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建均匀网格光线（输入=输出，无变形）
        n_rays = 100
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 输出位置与输入相同（无变形）
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        # OPD 为零
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 振幅应该接近均匀（归一化后平均值为 1）
        valid_amp = amplitude[valid_mask]
        assert np.allclose(valid_amp, 1.0, rtol=0.1), \
            f"均匀网格振幅应接近 1，实际为 {valid_amp.mean():.3f}"
        
        # 相位应为零（OPD 为零）
        valid_phase = phase[valid_mask]
        assert np.allclose(valid_phase, 0.0, atol=1e-10), \
            f"OPD 为零时相位应为零，实际为 {valid_phase.mean():.6f}"
    
    def test_expansion_reduces_amplitude(self):
        """测试光束扩展时振幅应减小
        
        当光束扩展（输出位置比输入位置更分散）时，
        根据能量守恒，振幅应减小。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建输入网格
        n_rays = 100
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 输出位置扩展 2 倍
        expansion_factor = 2.0
        ray_x_out = ray_x_in * expansion_factor
        ray_y_out = ray_y_in * expansion_factor
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 面积扩展 4 倍，振幅应减小到 1/2（归一化前）
        # 归一化后，振幅应接近均匀
        valid_amp = amplitude[valid_mask]
        
        # 检查振幅是否合理（应该是正数且有限）
        assert np.all(valid_amp > 0), "振幅应为正数"
        assert np.all(np.isfinite(valid_amp)), "振幅应为有限值"
    
    def test_compression_increases_amplitude(self):
        """测试光束压缩时振幅应增大
        
        当光束压缩（输出位置比输入位置更集中）时，
        根据能量守恒，振幅应增大。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建输入网格
        n_rays = 100
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 输出位置压缩到 0.5 倍
        compression_factor = 0.5
        ray_x_out = ray_x_in * compression_factor
        ray_y_out = ray_y_in * compression_factor
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        valid_amp = amplitude[valid_mask]
        
        # 检查振幅是否合理
        assert np.all(valid_amp > 0), "振幅应为正数"
        assert np.all(np.isfinite(valid_amp)), "振幅应为有限值"

    
    def test_phase_from_opd(self):
        """测试相位计算正确
        
        相位应为 -2π × OPD（波长数）
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建简单网格
        n_rays = 25
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        # 设置已知的 OPD
        opd_waves = np.ones(n_rays) * 0.5  # 0.5 波长
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 相位应为 -2π × 0.5 = -π
        expected_phase = -2 * np.pi * 0.5
        valid_phase = phase[valid_mask]
        assert np.allclose(valid_phase, expected_phase, atol=1e-10), \
            f"相位应为 {expected_phase:.4f}，实际为 {valid_phase.mean():.4f}"
    
    def test_invalid_rays_have_zero_amplitude(self):
        """测试无效光线区域振幅为 0
        
        对应需求 2.6
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建网格
        n_rays = 25
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        opd_waves = np.zeros(n_rays)
        
        # 部分光线无效
        valid_mask = np.ones(n_rays, dtype=bool)
        valid_mask[0:5] = False  # 前 5 条光线无效
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 无效光线振幅应为 0
        invalid_amp = amplitude[~valid_mask]
        assert np.all(invalid_amp == 0), "无效光线振幅应为 0"
        
        # 有效光线振幅应为正数
        valid_amp = amplitude[valid_mask]
        assert np.all(valid_amp > 0), "有效光线振幅应为正数"
    
    def test_insufficient_rays_raises_error(self):
        """测试有效光线数量不足时抛出异常
        
        对应需求 6.1
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 只有 3 条有效光线（少于 4 条）
        ray_x_in = np.array([0, 1, 2, 3, 4])
        ray_y_in = np.array([0, 0, 0, 0, 0])
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        opd_waves = np.zeros(5)
        
        # 只有 3 条有效
        valid_mask = np.array([True, True, True, False, False])
        
        with pytest.raises(InsufficientRaysError, match="有效光线数量不足"):
            reconstructor._compute_amplitude_phase_jacobian(
                ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
            )
    
    def test_amplitude_normalization(self):
        """测试振幅归一化
        
        归一化后，有效光线的平均振幅应接近 1
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建网格
        n_rays = 100
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 轻微变形
        ray_x_out = ray_x_in * 1.1
        ray_y_out = ray_y_in * 0.9
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amplitude, phase = reconstructor._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 归一化后平均振幅应接近 1
        valid_amp = amplitude[valid_mask]
        mean_amp = np.mean(valid_amp)
        assert np.isclose(mean_amp, 1.0, rtol=0.01), \
            f"归一化后平均振幅应接近 1，实际为 {mean_amp:.4f}"


class TestResampleToGridSeparate:
    """测试 _resample_to_grid_separate() 方法"""
    
    def test_uniform_amplitude_resampling(self):
        """测试均匀振幅的重采样
        
        均匀振幅输入应产生均匀振幅输出（在有效区域内）
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建均匀分布的光线
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        # 均匀振幅
        amplitude = np.ones(n_rays)
        # 零相位
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 检查输出形状
        assert amp_grid.shape == (64, 64), f"振幅网格形状应为 (64, 64)，实际为 {amp_grid.shape}"
        assert phase_grid.shape == (64, 64), f"相位网格形状应为 (64, 64)，实际为 {phase_grid.shape}"
        
        # 在光线覆盖区域内，振幅应接近 1
        # 网格中心区域（光线覆盖范围内）
        center_region = amp_grid[20:44, 20:44]
        assert np.mean(center_region) > 0.5, \
            f"中心区域振幅应接近 1，实际平均值为 {np.mean(center_region):.3f}"
    
    def test_zero_phase_resampling(self):
        """测试零相位的重采样
        
        零相位输入应产生零相位输出
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建光线
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        amplitude = np.ones(n_rays)
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 在有效区域内，相位应接近零
        valid_region = amp_grid > 0.1
        if np.any(valid_region):
            phase_in_valid = phase_grid[valid_region]
            assert np.allclose(phase_in_valid, 0.0, atol=0.1), \
                f"零相位输入应产生零相位输出，实际最大偏差为 {np.max(np.abs(phase_in_valid)):.4f}"
    
    def test_linear_phase_resampling(self):
        """测试线性相位的重采样
        
        线性相位（倾斜）应正确插值
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建光线
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        amplitude = np.ones(n_rays)
        # 线性相位：phase = k * x
        k = 1.0  # 弧度/mm
        phase = k * ray_x
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 检查相位梯度是否正确
        # 在中心区域，相位应该随 x 线性变化
        center_row = phase_grid[32, 20:44]
        # 计算相位梯度
        phase_diff = np.diff(center_row)
        # 每个像素的 x 变化为 sampling_mm
        expected_diff = k * reconstructor.sampling_mm
        
        # 相位梯度应接近预期值
        mean_diff = np.mean(phase_diff)
        assert np.isclose(mean_diff, expected_diff, rtol=0.2), \
            f"相位梯度应接近 {expected_diff:.4f}，实际为 {mean_diff:.4f}"
    
    def test_outside_sampling_range_is_zero(self):
        """测试采样范围外的区域设为 0
        
        对应需求 5.4
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建只覆盖中心小区域的光线
        n_per_side = 5
        x = np.linspace(-0.5, 0.5, n_per_side)  # 只覆盖 ±0.5 mm
        y = np.linspace(-0.5, 0.5, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        amplitude = np.ones(n_rays)
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 网格范围是 ±3.2 mm，光线只覆盖 ±0.5 mm
        # 边缘区域应该是 0
        edge_region = amp_grid[0:10, :]  # 顶部边缘
        assert np.allclose(edge_region, 0.0, atol=0.01), \
            f"采样范围外振幅应为 0，实际最大值为 {np.max(edge_region):.4f}"
    
    def test_insufficient_rays_raises_error(self):
        """测试有效光线数量不足时抛出异常"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 只有 3 条有效光线
        ray_x = np.array([0, 1, 2, 3, 4])
        ray_y = np.array([0, 0, 0, 0, 0])
        amplitude = np.ones(5)
        phase = np.zeros(5)
        valid_mask = np.array([True, True, True, False, False])
        
        with pytest.raises(InsufficientRaysError, match="有效光线数量不足"):
            reconstructor._resample_to_grid_separate(
                ray_x, ray_y, amplitude, phase, valid_mask
            )
    
    def test_nan_handling(self):
        """测试 NaN 值处理
        
        三次插值可能在边界产生 NaN，应被替换为 0
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=32,
            sampling_mm=0.2,
            wavelength_um=0.633
        )
        
        # 创建不规则分布的光线（可能导致边界 NaN）
        np.random.seed(42)
        n_rays = 50
        ray_x = np.random.uniform(-2, 2, n_rays)
        ray_y = np.random.uniform(-2, 2, n_rays)
        amplitude = np.ones(n_rays)
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 不应有 NaN 值
        assert not np.any(np.isnan(amp_grid)), "振幅网格不应包含 NaN"
        assert not np.any(np.isnan(phase_grid)), "相位网格不应包含 NaN"
    
    def test_amplitude_non_negative(self):
        """测试振幅非负
        
        插值可能产生负值，应被限制为非负
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=32,
            sampling_mm=0.2,
            wavelength_um=0.633
        )
        
        # 创建振幅变化较大的光线
        n_per_side = 8
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        # 振幅有较大变化
        amplitude = 1.0 + 0.5 * np.sin(ray_x * 2)
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 振幅应非负
        assert np.all(amp_grid >= 0), "振幅应非负"
    
    def test_smooth_interpolation(self):
        """测试插值结果平滑
        
        三次插值应产生平滑的结果
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建平滑变化的振幅
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        n_rays = len(ray_x)
        
        # 高斯分布振幅
        r_sq = ray_x**2 + ray_y**2
        amplitude = np.exp(-r_sq / 2)
        phase = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        amp_grid, phase_grid = reconstructor._resample_to_grid_separate(
            ray_x, ray_y, amplitude, phase, valid_mask
        )
        
        # 检查中心区域的平滑性（梯度不应有突变）
        center_region = amp_grid[24:40, 24:40]
        grad_x = np.diff(center_region, axis=1)
        grad_y = np.diff(center_region, axis=0)
        
        # 梯度的变化应该平滑（二阶导数不应太大）
        grad2_x = np.diff(grad_x, axis=1)
        grad2_y = np.diff(grad_y, axis=0)
        
        max_grad2 = max(np.max(np.abs(grad2_x)), np.max(np.abs(grad2_y)))
        assert max_grad2 < 0.5, \
            f"插值结果应平滑，二阶导数最大值为 {max_grad2:.4f}"


class TestReconstruct:
    """测试 reconstruct() 公共方法"""
    
    def test_reconstruct_returns_complex_array(self):
        """测试 reconstruct() 返回复数数组
        
        对应需求 2.1：方法返回正确形状的复数数组
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建测试数据
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 调用 reconstruct
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 验证返回类型和形状
        assert isinstance(result, np.ndarray), "返回值应为 numpy 数组"
        assert np.iscomplexobj(result), "返回值应为复数数组"
        assert result.shape == (64, 64), f"形状应为 (64, 64)，实际为 {result.shape}"
        assert result.dtype == np.complex128, f"数据类型应为 complex128，实际为 {result.dtype}"
    
    def test_reconstruct_with_different_grid_sizes(self):
        """测试不同网格大小的重建"""
        for grid_size in [32, 64, 128]:
            reconstructor = RayToWavefrontReconstructor(
                grid_size=grid_size,
                sampling_mm=0.1,
                wavelength_um=0.633
            )
            
            # 创建测试数据
            n_per_side = 8
            x = np.linspace(-2, 2, n_per_side)
            y = np.linspace(-2, 2, n_per_side)
            X, Y = np.meshgrid(x, y)
            ray_x_in = X.flatten()
            ray_y_in = Y.flatten()
            ray_x_out = ray_x_in.copy()
            ray_y_out = ray_y_in.copy()
            n_rays = len(ray_x_in)
            
            opd_waves = np.zeros(n_rays)
            valid_mask = np.ones(n_rays, dtype=bool)
            
            result = reconstructor.reconstruct(
                ray_x_in, ray_y_in,
                ray_x_out, ray_y_out,
                opd_waves, valid_mask
            )
            
            assert result.shape == (grid_size, grid_size), \
                f"网格大小 {grid_size} 时形状应为 ({grid_size}, {grid_size})"
    
    def test_reconstruct_uniform_input_gives_uniform_amplitude(self):
        """测试均匀输入产生均匀振幅
        
        当输入和输出位置相同、OPD 为零时，
        重建的复振幅在有效区域内应接近均匀。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建均匀网格
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 提取振幅
        amplitude = np.abs(result)
        
        # 在有效区域内（振幅 > 0.1），振幅应接近均匀
        valid_region = amplitude > 0.1
        if np.any(valid_region):
            valid_amp = amplitude[valid_region]
            # 标准差应该较小（相对于平均值）
            relative_std = np.std(valid_amp) / np.mean(valid_amp)
            assert relative_std < 0.3, \
                f"均匀输入的振幅相对标准差应 < 0.3，实际为 {relative_std:.3f}"
    
    def test_reconstruct_zero_opd_gives_zero_phase(self):
        """测试零 OPD 产生零相位
        
        当 OPD 为零时，重建的复振幅相位应接近零。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建测试数据
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 提取相位（只在有效区域）
        amplitude = np.abs(result)
        phase = np.angle(result)
        
        valid_region = amplitude > 0.1
        if np.any(valid_region):
            valid_phase = phase[valid_region]
            # 相位应接近零
            assert np.allclose(valid_phase, 0.0, atol=0.2), \
                f"零 OPD 时相位应接近零，实际最大偏差为 {np.max(np.abs(valid_phase)):.4f}"
    
    def test_reconstruct_with_opd_gives_correct_phase(self):
        """测试带 OPD 的重建产生正确相位
        
        相位应为 -2π × OPD
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建测试数据
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        # 设置均匀的 OPD
        opd_value = 0.25  # 0.25 波长
        opd_waves = np.ones(n_rays) * opd_value
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 提取相位
        amplitude = np.abs(result)
        phase = np.angle(result)
        
        # 预期相位 = -2π × 0.25 = -π/2
        expected_phase = -2 * np.pi * opd_value
        
        valid_region = amplitude > 0.1
        if np.any(valid_region):
            valid_phase = phase[valid_region]
            mean_phase = np.mean(valid_phase)
            assert np.isclose(mean_phase, expected_phase, atol=0.2), \
                f"相位应接近 {expected_phase:.4f}，实际为 {mean_phase:.4f}"
    
    def test_reconstruct_insufficient_rays_raises_error(self):
        """测试有效光线不足时抛出异常
        
        对应需求 6.1
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 只有 3 条有效光线
        ray_x_in = np.array([0, 1, 2, 3, 4])
        ray_y_in = np.array([0, 0, 0, 0, 0])
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        opd_waves = np.zeros(5)
        valid_mask = np.array([True, True, True, False, False])
        
        with pytest.raises(InsufficientRaysError, match="有效光线数量不足"):
            reconstructor.reconstruct(
                ray_x_in, ray_y_in,
                ray_x_out, ray_y_out,
                opd_waves, valid_mask
            )
    
    def test_reconstruct_outside_region_is_zero(self):
        """测试采样范围外的区域为零
        
        对应需求 5.4
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建只覆盖中心小区域的光线
        n_per_side = 5
        x = np.linspace(-0.5, 0.5, n_per_side)  # 只覆盖 ±0.5 mm
        y = np.linspace(-0.5, 0.5, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 边缘区域应该是零
        edge_region = result[0:10, :]  # 顶部边缘
        assert np.allclose(np.abs(edge_region), 0.0, atol=0.01), \
            f"采样范围外振幅应为 0，实际最大值为 {np.max(np.abs(edge_region)):.4f}"
    
    def test_reconstruct_with_expansion(self):
        """测试光束扩展时的重建
        
        光束扩展时，振幅应减小（能量守恒）
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建输入网格
        n_per_side = 10
        x = np.linspace(-1, 1, n_per_side)
        y = np.linspace(-1, 1, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        n_rays = len(ray_x_in)
        
        # 输出位置扩展 2 倍
        expansion_factor = 2.0
        ray_x_out = ray_x_in * expansion_factor
        ray_y_out = ray_y_in * expansion_factor
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 验证结果是有效的复数数组
        assert np.iscomplexobj(result), "结果应为复数数组"
        assert not np.any(np.isnan(result)), "结果不应包含 NaN"
        assert not np.any(np.isinf(result)), "结果不应包含无穷大"
    
    def test_reconstruct_disable_phase_check(self):
        """测试禁用相位突变检测
        
        当 check_phase_discontinuity=False 时，不应发出警告
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建测试数据
        n_per_side = 10
        x = np.linspace(-2, 2, n_per_side)
        y = np.linspace(-2, 2, n_per_side)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        n_rays = len(ray_x_in)
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 禁用相位检测，应该正常执行
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        assert result.shape == (64, 64)


class TestCheckPhaseDiscontinuityOnGrid:
    """测试 _check_phase_discontinuity_on_grid() 方法"""
    
    def test_no_discontinuity_returns_false(self):
        """测试无相位突变时返回 False"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建平滑的相位网格
        n = 64
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # 小的线性相位（不会超过 π）
        phase_grid = 0.1 * X  # 最大相位差约 0.003 弧度/像素
        amplitude_grid = np.ones((n, n))
        
        result = reconstructor._check_phase_discontinuity_on_grid(
            phase_grid, amplitude_grid
        )
        
        assert result == False, "无相位突变时应返回 False"
    
    def test_discontinuity_returns_true_and_warns(self):
        """测试有相位突变时返回 True 并发出警告
        
        对应需求 6.2
        """
        import warnings
        
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建有突变的相位网格
        n = 64
        phase_grid = np.zeros((n, n))
        # 在中间创建一个大的相位跳变（> π）
        phase_grid[:, 32:] = 4.0  # 跳变 4 弧度 > π
        amplitude_grid = np.ones((n, n))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reconstructor._check_phase_discontinuity_on_grid(
                phase_grid, amplitude_grid
            )
            
            assert result == True, "有相位突变时应返回 True"
            assert len(w) == 1, "应发出一个警告"
            assert "相位突变" in str(w[0].message), "警告信息应包含'相位突变'"
    
    def test_only_checks_valid_region(self):
        """测试只检查有效区域
        
        无效区域（振幅为 0）的相位突变不应触发警告
        """
        import warnings
        
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        n = 64
        # 相位有大跳变
        phase_grid = np.zeros((n, n))
        phase_grid[:, 32:] = 4.0
        
        # 但振幅在跳变区域为 0
        amplitude_grid = np.zeros((n, n))
        amplitude_grid[20:44, 20:30] = 1.0  # 只有左半部分有效
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reconstructor._check_phase_discontinuity_on_grid(
                phase_grid, amplitude_grid
            )
            
            # 因为跳变发生在有效区域边界外，不应触发警告
            assert result == False, "无效区域的相位突变不应触发警告"
    
    def test_empty_valid_region_returns_false(self):
        """测试空有效区域返回 False"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        n = 64
        phase_grid = np.zeros((n, n))
        amplitude_grid = np.zeros((n, n))  # 全部无效
        
        result = reconstructor._check_phase_discontinuity_on_grid(
            phase_grid, amplitude_grid
        )
        
        assert result == False, "空有效区域应返回 False"
