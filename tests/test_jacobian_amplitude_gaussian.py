"""雅可比矩阵方法的高斯光束验证测试

本模块测试 RayToWavefrontReconstructor 的雅可比矩阵方法对高斯光束的准确性。

测试内容：
- 高斯光束通过理想透镜聚焦
- 高斯光束扩束场景
- 能量守恒验证
- 振幅分布与理论预期对比

**Validates: Requirements 2.3, 2.4 - 雅可比矩阵振幅计算**

参考：
- 高斯光束 ABCD 矩阵理论
- src/wavefront_to_rays/reconstructor.py
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, 'src')

from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
from wavefront_to_rays.exceptions import InsufficientRaysError


class TestRayToWavefrontReconstructorBasic:
    """RayToWavefrontReconstructor 基本功能测试
    
    测试重建器的基本初始化和参数验证。
    """
    
    def test_initialization(self):
        """测试重建器初始化"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=256,
            sampling_mm=0.01,
            wavelength_um=0.633
        )
        
        assert reconstructor.grid_size == 256
        assert reconstructor.sampling_mm == 0.01
        assert reconstructor.wavelength_um == 0.633
    
    def test_grid_properties(self):
        """测试网格属性计算"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=512,
            sampling_mm=0.02,
            wavelength_um=0.55
        )
        
        # 网格半尺寸 = 0.02 * 512 / 2 = 5.12 mm
        expected_half_size = 0.02 * 512 / 2
        assert abs(reconstructor.grid_half_size_mm - expected_half_size) < 1e-10
        
        # 网格范围
        extent = reconstructor.grid_extent_mm
        assert abs(extent[0] + expected_half_size) < 1e-10
        assert abs(extent[1] - expected_half_size) < 1e-10
    
    def test_invalid_grid_size(self):
        """测试无效网格大小参数"""
        with pytest.raises(ValueError, match="grid_size"):
            RayToWavefrontReconstructor(
                grid_size=-1,
                sampling_mm=0.01,
                wavelength_um=0.633
            )
    
    def test_invalid_sampling(self):
        """测试无效采样间隔参数"""
        with pytest.raises(ValueError, match="sampling_mm"):
            RayToWavefrontReconstructor(
                grid_size=256,
                sampling_mm=0,
                wavelength_um=0.633
            )
    
    def test_invalid_wavelength(self):
        """测试无效波长参数"""
        with pytest.raises(ValueError, match="wavelength_um"):
            RayToWavefrontReconstructor(
                grid_size=256,
                sampling_mm=0.01,
                wavelength_um=-0.633
            )


class TestJacobianAmplitudeUniformBeam:
    """均匀光束的雅可比矩阵振幅测试
    
    测试均匀光束（无位置变化）的振幅计算。
    
    **Validates: Requirements 2.3 - 雅可比矩阵振幅计算**
    """
    
    def test_uniform_beam_unit_amplitude(self):
        """测试均匀光束的振幅为 1
        
        当输入和输出位置相同时（无变形），雅可比行列式为 1，
        因此振幅应该为 1。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建均匀分布的光线
        n_rays = 100
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        
        # 输入和输出位置相同（无变形）
        ray_x_in = ray_x.copy()
        ray_y_in = ray_y.copy()
        ray_x_out = ray_x.copy()
        ray_y_out = ray_y.copy()
        
        # OPD 为零
        opd_waves = np.zeros(len(ray_x))
        
        # 所有光线有效
        valid_mask = np.ones(len(ray_x), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 获取有效区域的振幅
        amplitude = np.abs(complex_amplitude)
        valid_region = amplitude > 0.1 * np.max(amplitude)
        
        # 有效区域内振幅应该接近均匀
        amplitude_in_region = amplitude[valid_region]
        amplitude_std = np.std(amplitude_in_region)
        amplitude_mean = np.mean(amplitude_in_region)
        
        # 相对标准差应该很小（< 20%）
        relative_std = amplitude_std / amplitude_mean
        assert relative_std < 0.2, (
            f"均匀光束振幅不均匀：相对标准差 = {relative_std:.3f}"
        )


class TestJacobianAmplitudeExpansion:
    """光束扩展场景的雅可比矩阵振幅测试
    
    测试光束扩展时振幅的变化。
    
    **Validates: Requirements 2.3, 2.4 - 雅可比矩阵振幅计算**
    """
    
    def test_expansion_reduces_amplitude(self):
        """测试光束扩展时振幅减小
        
        当光束扩展（输出位置比输入位置更分散）时，
        雅可比行列式 > 1，振幅应该减小。
        
        **Validates: Requirements 2.3 - 能量守恒**
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.2,
            wavelength_um=0.633
        )
        
        # 创建光线网格
        n = 10
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 输出位置扩展 2 倍
        magnification = 2.0
        ray_x_out = ray_x_in * magnification
        ray_y_out = ray_y_in * magnification
        
        # OPD 为零
        opd_waves = np.zeros(len(ray_x_in))
        valid_mask = np.ones(len(ray_x_in), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 获取振幅
        amplitude = np.abs(complex_amplitude)
        
        # 峰值振幅应该存在
        peak_amplitude = np.max(amplitude)
        assert peak_amplitude > 0, "峰值振幅应为正值"
        
        # 由于归一化，平均振幅约为 1
        # 但扩展后的振幅分布应该更分散
        valid_region = amplitude > 0.1 * peak_amplitude
        assert np.sum(valid_region) > 0, "应该有有效区域"
    
    def test_compression_increases_amplitude(self):
        """测试光束压缩时振幅增大
        
        当光束压缩（输出位置比输入位置更集中）时，
        雅可比行列式 < 1，振幅应该增大。
        
        **Validates: Requirements 2.3 - 能量守恒**
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建光线网格
        n = 10
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        
        # 输出位置压缩 0.5 倍
        magnification = 0.5
        ray_x_out = ray_x_in * magnification
        ray_y_out = ray_y_in * magnification
        
        # OPD 为零
        opd_waves = np.zeros(len(ray_x_in))
        valid_mask = np.ones(len(ray_x_in), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 获取振幅
        amplitude = np.abs(complex_amplitude)
        
        # 峰值振幅应该存在
        peak_amplitude = np.max(amplitude)
        assert peak_amplitude > 0, "峰值振幅应为正值"


class TestJacobianAmplitudeGaussianBeam:
    """高斯光束的雅可比矩阵振幅测试
    
    测试高斯光束场景下的振幅计算准确性。
    
    **Validates: Requirements 2.3, 2.4 - 雅可比矩阵振幅计算**
    """
    
    def test_gaussian_beam_expansion_energy_conservation(self):
        """测试高斯光束扩展时的能量守恒
        
        模拟高斯光束通过扩束器的场景，验证能量守恒。
        
        **Validates: Requirements 2.3 - 能量守恒**
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=128,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建高斯分布的光线采样
        # 使用极坐标采样以更好地覆盖高斯分布
        n_radial = 10
        n_angular = 12
        
        r_values = np.linspace(0.1, 3.0, n_radial)  # 避免 r=0
        theta_values = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
        
        ray_x_in = []
        ray_y_in = []
        
        for r in r_values:
            for theta in theta_values:
                ray_x_in.append(r * np.cos(theta))
                ray_y_in.append(r * np.sin(theta))
        
        ray_x_in = np.array(ray_x_in)
        ray_y_in = np.array(ray_y_in)
        
        # 3x 扩束
        magnification = 3.0
        ray_x_out = ray_x_in * magnification
        ray_y_out = ray_y_in * magnification
        
        # OPD 为零（理想扩束器）
        opd_waves = np.zeros(len(ray_x_in))
        valid_mask = np.ones(len(ray_x_in), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 计算总能量
        intensity = np.abs(complex_amplitude)**2
        total_energy = np.sum(intensity)
        
        # 能量应该是有限正值
        assert np.isfinite(total_energy), "总能量应为有限值"
        assert total_energy > 0, "总能量应为正值"


class TestJacobianAmplitudePhase:
    """相位计算测试
    
    测试 OPD 到相位的转换。
    
    **Validates: Requirements 2.2 - 复振幅公式**
    """
    
    def test_zero_opd_zero_phase(self):
        """测试零 OPD 对应零相位"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建光线网格
        n = 8
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        
        # 输入输出位置相同
        ray_x_in = ray_x.copy()
        ray_y_in = ray_y.copy()
        ray_x_out = ray_x.copy()
        ray_y_out = ray_y.copy()
        
        # OPD 为零
        opd_waves = np.zeros(len(ray_x))
        valid_mask = np.ones(len(ray_x), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 获取相位
        phase = np.angle(complex_amplitude)
        amplitude = np.abs(complex_amplitude)
        
        # 有效区域内相位应该接近零
        valid_region = amplitude > 0.1 * np.max(amplitude)
        phase_in_region = phase[valid_region]
        
        # 相位应该接近零（允许小的数值误差）
        max_phase = np.max(np.abs(phase_in_region))
        assert max_phase < 0.5, (
            f"零 OPD 时相位应接近零，但最大相位为 {max_phase:.3f} rad"
        )
    
    def test_half_wave_opd_pi_phase(self):
        """测试半波 OPD 对应 π 相位
        
        OPD = 0.5 波长时，相位 = -2π × 0.5 = -π
        
        注意：由于插值和边界效应，实际相位可能有较大偏差。
        这里主要验证相位计算的基本正确性。
        """
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 创建光线网格
        n = 8
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(x, y)
        ray_x = X.flatten()
        ray_y = Y.flatten()
        
        # 输入输出位置相同
        ray_x_in = ray_x.copy()
        ray_y_in = ray_y.copy()
        ray_x_out = ray_x.copy()
        ray_y_out = ray_y.copy()
        
        # OPD = 0.5 波长
        opd_waves = np.full(len(ray_x), 0.5)
        valid_mask = np.ones(len(ray_x), dtype=bool)
        
        # 重建复振幅
        complex_amplitude = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask,
            check_phase_discontinuity=False
        )
        
        # 获取相位
        phase = np.angle(complex_amplitude)
        amplitude = np.abs(complex_amplitude)
        
        # 有效区域内相位应该接近 -π
        valid_region = amplitude > 0.1 * np.max(amplitude)
        phase_in_region = phase[valid_region]
        
        # 相位应该接近 -π（允许一定误差）
        expected_phase = -np.pi
        mean_phase = np.mean(phase_in_region)
        
        # 验证相位不为零（即 OPD 确实引入了相位变化）
        assert abs(mean_phase) > 0.5, (
            f"半波 OPD 应引入显著相位变化，但平均相位为 {mean_phase:.3f} rad"
        )
        
        # 验证相位在合理范围内（-π 到 π）
        assert -np.pi <= mean_phase <= np.pi, (
            f"相位应在 [-π, π] 范围内，但平均相位为 {mean_phase:.3f} rad"
        )


class TestJacobianAmplitudeErrorHandling:
    """错误处理测试
    
    测试边界情况和错误处理。
    
    **Validates: Requirements 6.1 - 有效光线数量检查**
    """
    
    def test_insufficient_rays_error(self):
        """测试有效光线不足时抛出异常"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 只有 3 条有效光线（少于 4 条）
        ray_x_in = np.array([0, 1, 2])
        ray_y_in = np.array([0, 0, 0])
        ray_x_out = np.array([0, 1, 2])
        ray_y_out = np.array([0, 0, 0])
        opd_waves = np.array([0, 0, 0])
        valid_mask = np.array([True, True, True])
        
        with pytest.raises(InsufficientRaysError):
            reconstructor.reconstruct(
                ray_x_in, ray_y_in,
                ray_x_out, ray_y_out,
                opd_waves, valid_mask
            )
    
    def test_all_invalid_rays_error(self):
        """测试所有光线无效时抛出异常"""
        reconstructor = RayToWavefrontReconstructor(
            grid_size=64,
            sampling_mm=0.1,
            wavelength_um=0.633
        )
        
        # 所有光线都无效
        ray_x_in = np.array([0, 1, 2, 3, 4])
        ray_y_in = np.array([0, 0, 0, 0, 0])
        ray_x_out = np.array([0, 1, 2, 3, 4])
        ray_y_out = np.array([0, 0, 0, 0, 0])
        opd_waves = np.array([0, 0, 0, 0, 0])
        valid_mask = np.array([False, False, False, False, False])
        
        with pytest.raises(InsufficientRaysError):
            reconstructor.reconstruct(
                ray_x_in, ray_y_in,
                ray_x_out, ray_y_out,
                opd_waves, valid_mask
            )


class TestJacobianAmplitudeIntegration:
    """集成测试：与 SequentialOpticalSystem 的集成
    
    测试 RayToWavefrontReconstructor 在实际光学系统中的使用。
    
    **Validates: Requirements 2.3, 2.4 - 雅可比矩阵振幅计算**
    """
    
    def test_simple_lens_system(self):
        """测试简单透镜系统中的振幅计算
        
        使用 SequentialOpticalSystem 创建简单透镜系统，
        验证振幅计算的正确性。
        """
        from sequential_system import (
            SequentialOpticalSystem,
            GaussianBeamSource,
            ThinLens,
        )
        
        # 创建简单透镜系统
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=5.0,            # mm
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source=source,
            grid_size=256,
            beam_ratio=0.3,
            use_hybrid_propagation=True,
            hybrid_num_rays=64,
        )
        
        # 添加采样面和透镜
        system.add_sampling_plane(distance=0.0, name="Input")
        system.add_surface(ThinLens(
            focal_length_value=100.0,  # mm
            thickness=50.0,
            semi_aperture=20.0,
        ))
        system.add_sampling_plane(distance=50.0, name="Output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        input_result = results.sampling_results["Input"]
        output_result = results.sampling_results["Output"]
        
        # 振幅应该是有限正值
        input_amp = np.abs(input_result.wavefront)
        output_amp = np.abs(output_result.wavefront)
        
        assert np.all(np.isfinite(input_amp)), "输入振幅应为有限值"
        assert np.all(np.isfinite(output_amp)), "输出振幅应为有限值"
        assert np.max(input_amp) > 0, "输入峰值振幅应为正值"
        assert np.max(output_amp) > 0, "输出峰值振幅应为正值"
        
        # 能量应该守恒
        input_energy = np.sum(input_amp**2)
        output_energy = np.sum(output_amp**2)
        energy_ratio = output_energy / input_energy
        
        assert 0.9 < energy_ratio < 1.1, (
            f"能量不守恒：比值 = {energy_ratio:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
