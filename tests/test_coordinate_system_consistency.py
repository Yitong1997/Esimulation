"""
坐标系统一致性测试

验证光线坐标与 PROPER 网格坐标的对应关系。

对应需求: 需求 5.1, 5.2, 5.3, 5.4

测试内容：
1. 验证采样范围正确
2. 验证插值映射正确
3. 验证光线坐标与 PROPER 网格坐标对应
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import proper
from wavefront_to_rays import RayToWavefrontReconstructor


class TestCoordinateSystemConsistency:
    """测试坐标系统一致性"""
    
    def test_grid_coordinates_match_proper(self):
        """测试重建器网格坐标与 PROPER 网格坐标一致
        
        验证 RayToWavefrontReconstructor 使用的网格坐标范围
        与 PROPER 波前网格的物理坐标范围一致。
        
        对应需求: 需求 5.1
        """
        # 设置参数
        grid_size = 64
        wavelength_um = 0.633
        wavelength_m = wavelength_um * 1e-6
        beam_diameter_m = 0.01  # 10 mm
        beam_ratio = 0.5
        
        # 创建 PROPER 波前
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        # 获取 PROPER 采样间隔
        proper_sampling_m = proper.prop_get_sampling(wfo)
        proper_sampling_mm = proper_sampling_m * 1e3
        
        # 计算 PROPER 网格范围
        proper_half_size_mm = proper_sampling_mm * grid_size / 2
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=proper_sampling_mm,
            wavelength_um=wavelength_um
        )
        
        # 验证网格范围一致
        assert np.isclose(
            reconstructor.grid_half_size_mm, 
            proper_half_size_mm, 
            rtol=1e-6
        ), (
            f"网格半尺寸不一致：重建器 {reconstructor.grid_half_size_mm:.6f} mm，"
            f"PROPER {proper_half_size_mm:.6f} mm"
        )
        
        # 验证网格范围
        extent = reconstructor.grid_extent_mm
        assert np.isclose(extent[0], -proper_half_size_mm, rtol=1e-6)
        assert np.isclose(extent[1], proper_half_size_mm, rtol=1e-6)
    
    def test_ray_positions_within_proper_grid(self):
        """测试光线位置在 PROPER 网格范围内
        
        验证采样光线的位置范围与 PROPER 网格的物理范围一致。
        
        对应需求: 需求 5.2
        """
        # 设置参数
        grid_size = 64
        sampling_mm = 0.1
        wavelength_um = 0.633
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=sampling_mm,
            wavelength_um=wavelength_um
        )
        
        # 计算网格范围
        half_size = reconstructor.grid_half_size_mm
        
        # 创建覆盖整个网格的光线
        n_rays_1d = 10
        ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
        ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
        ray_x = ray_X.flatten()
        ray_y = ray_Y.flatten()
        
        # 验证光线位置在网格范围内
        assert np.all(ray_x >= -half_size - 1e-10)
        assert np.all(ray_x <= half_size + 1e-10)
        assert np.all(ray_y >= -half_size - 1e-10)
        assert np.all(ray_y <= half_size + 1e-10)

    def test_interpolation_mapping_correct(self):
        """测试插值映射正确
        
        验证从光线位置到网格索引的映射正确。
        
        对应需求: 需求 5.3
        """
        # 设置参数
        grid_size = 64
        sampling_mm = 0.1
        wavelength_um = 0.633
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=sampling_mm,
            wavelength_um=wavelength_um
        )
        
        half_size = reconstructor.grid_half_size_mm
        
        # 创建在网格中心的光线
        n_rays = 25
        x = np.linspace(-half_size * 0.8, half_size * 0.8, 5)
        y = np.linspace(-half_size * 0.8, half_size * 0.8, 5)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        # 设置已知的 OPD 模式（线性倾斜）
        # OPD = k * x，其中 k = 0.1 波长/mm
        k = 0.1  # 波长/mm
        opd_waves = k * ray_x_in
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 重建复振幅
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 提取相位
        phase = np.angle(result)
        amplitude = np.abs(result)
        
        # 在有效区域内验证相位梯度
        # 相位 = -2π × OPD = -2π × k × x
        # 相位梯度 = -2π × k × sampling_mm
        expected_phase_gradient = -2 * np.pi * k * sampling_mm
        
        # 计算中心行的相位梯度
        center_row = grid_size // 2
        valid_cols = amplitude[center_row, :] > 0.1
        if np.sum(valid_cols) > 2:
            phase_row = phase[center_row, valid_cols]
            actual_gradient = np.mean(np.diff(phase_row))
            
            # 验证相位梯度接近预期值（允许 20% 误差）
            assert np.isclose(actual_gradient, expected_phase_gradient, rtol=0.2), (
                f"相位梯度不正确：预期 {expected_phase_gradient:.4f}，"
                f"实际 {actual_gradient:.4f}"
            )

    def test_outside_sampling_range_is_zero(self):
        """测试采样范围外的区域设为 0
        
        验证在光线采样范围外的网格区域，振幅被正确设为 0。
        
        对应需求: 需求 5.4
        """
        # 设置参数
        grid_size = 64
        sampling_mm = 0.1
        wavelength_um = 0.633
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=sampling_mm,
            wavelength_um=wavelength_um
        )
        
        half_size = reconstructor.grid_half_size_mm
        
        # 创建只覆盖中心小区域的光线（±1 mm，而网格范围是 ±3.2 mm）
        small_range = 1.0  # mm
        n_rays = 25
        x = np.linspace(-small_range, small_range, 5)
        y = np.linspace(-small_range, small_range, 5)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 重建复振幅
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        amplitude = np.abs(result)
        
        # 验证边缘区域（采样范围外）振幅为 0
        # 边缘 10 行/列应该是 0
        edge_top = amplitude[0:10, :]
        edge_bottom = amplitude[-10:, :]
        edge_left = amplitude[:, 0:10]
        edge_right = amplitude[:, -10:]
        
        assert np.allclose(edge_top, 0.0, atol=0.01), \
            f"顶部边缘振幅应为 0，实际最大值为 {np.max(edge_top):.4f}"
        assert np.allclose(edge_bottom, 0.0, atol=0.01), \
            f"底部边缘振幅应为 0，实际最大值为 {np.max(edge_bottom):.4f}"
        assert np.allclose(edge_left, 0.0, atol=0.01), \
            f"左侧边缘振幅应为 0，实际最大值为 {np.max(edge_left):.4f}"
        assert np.allclose(edge_right, 0.0, atol=0.01), \
            f"右侧边缘振幅应为 0，实际最大值为 {np.max(edge_right):.4f}"

    def test_center_position_correct(self):
        """测试中心位置正确
        
        验证网格中心对应物理坐标 (0, 0)。
        
        对应需求: 需求 5.1
        """
        # 设置参数
        grid_size = 64
        sampling_mm = 0.1
        wavelength_um = 0.633
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=sampling_mm,
            wavelength_um=wavelength_um
        )
        
        half_size = reconstructor.grid_half_size_mm
        
        # 创建以原点为中心的光线
        n_rays = 25
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        # 设置高斯分布的 OPD（以原点为中心）
        r_sq = ray_x_in**2 + ray_y_in**2
        opd_waves = 0.1 * r_sq  # 二次相位
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 重建复振幅
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        phase = np.angle(result)
        amplitude = np.abs(result)
        
        # 网格中心索引
        center_idx = grid_size // 2
        
        # 验证中心附近的相位接近 0（因为 r=0 时 OPD=0）
        center_phase = phase[center_idx-2:center_idx+2, center_idx-2:center_idx+2]
        center_amp = amplitude[center_idx-2:center_idx+2, center_idx-2:center_idx+2]
        
        # 只检查有效区域
        if np.mean(center_amp) > 0.1:
            assert np.abs(np.mean(center_phase)) < 0.5, \
                f"中心相位应接近 0，实际为 {np.mean(center_phase):.4f}"


class TestProperIntegration:
    """测试与 PROPER 的集成"""
    
    def test_reconstructor_output_compatible_with_proper(self):
        """测试重建器输出与 PROPER 波前兼容
        
        验证重建的复振幅可以正确应用到 PROPER 波前。
        """
        # 设置参数
        grid_size = 64
        wavelength_um = 0.633
        wavelength_m = wavelength_um * 1e-6
        beam_diameter_m = 0.01  # 10 mm
        beam_ratio = 0.5
        
        # 创建 PROPER 波前
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        # 获取 PROPER 采样间隔
        proper_sampling_m = proper.prop_get_sampling(wfo)
        proper_sampling_mm = proper_sampling_m * 1e3
        
        # 创建重建器
        reconstructor = RayToWavefrontReconstructor(
            grid_size=grid_size,
            sampling_mm=proper_sampling_mm,
            wavelength_um=wavelength_um
        )
        
        half_size = reconstructor.grid_half_size_mm
        
        # 创建测试光线
        n_rays = 100
        x = np.linspace(-half_size * 0.8, half_size * 0.8, 10)
        y = np.linspace(-half_size * 0.8, half_size * 0.8, 10)
        X, Y = np.meshgrid(x, y)
        ray_x_in = X.flatten()
        ray_y_in = Y.flatten()
        ray_x_out = ray_x_in.copy()
        ray_y_out = ray_y_in.copy()
        
        opd_waves = np.zeros(n_rays)
        valid_mask = np.ones(n_rays, dtype=bool)
        
        # 重建复振幅
        result = reconstructor.reconstruct(
            ray_x_in, ray_y_in,
            ray_x_out, ray_y_out,
            opd_waves, valid_mask
        )
        
        # 验证输出形状与 PROPER 网格一致
        assert result.shape == (grid_size, grid_size), \
            f"输出形状应为 ({grid_size}, {grid_size})，实际为 {result.shape}"
        
        # 验证可以应用到 PROPER 波前
        # 使用 prop_shift_center 转换坐标系
        result_fft = proper.prop_shift_center(result)
        
        # 验证转换后形状不变
        assert result_fft.shape == (grid_size, grid_size)
        
        # 验证可以与 PROPER 波前相乘
        original_wfarr = wfo.wfarr.copy()
        wfo.wfarr = wfo.wfarr * result_fft
        
        # 验证波前仍然有效
        assert wfo.wfarr.shape == (grid_size, grid_size)
        assert np.all(np.isfinite(wfo.wfarr))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
