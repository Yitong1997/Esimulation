"""
Fixtures 验证测试

验证 conftest.py 中定义的 fixtures 是否正常工作。
此文件仅用于验证测试基础设施，不是正式的功能测试。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest

from wavefront_to_rays.element_raytracer import SurfaceDefinition


class TestAmplitudeFixtures:
    """测试复振幅 fixtures"""
    
    def test_gaussian_amplitude_32_shape(self, gaussian_amplitude_32):
        """验证 32x32 高斯振幅形状"""
        assert gaussian_amplitude_32.shape == (32, 32)
        assert gaussian_amplitude_32.dtype == np.complex128
    
    def test_gaussian_amplitude_64_shape(self, gaussian_amplitude_64):
        """验证 64x64 高斯振幅形状"""
        assert gaussian_amplitude_64.shape == (64, 64)
        assert gaussian_amplitude_64.dtype == np.complex128
    
    def test_gaussian_amplitude_128_shape(self, gaussian_amplitude_128):
        """验证 128x128 高斯振幅形状"""
        assert gaussian_amplitude_128.shape == (128, 128)
    
    def test_gaussian_amplitude_256_shape(self, gaussian_amplitude_256):
        """验证 256x256 高斯振幅形状"""
        assert gaussian_amplitude_256.shape == (256, 256)
    
    def test_gaussian_amplitude_has_nonzero_values(self, gaussian_amplitude_64):
        """验证高斯振幅有非零值"""
        assert np.max(np.abs(gaussian_amplitude_64)) > 0
    
    def test_gaussian_amplitude_center_is_maximum(self, gaussian_amplitude_64):
        """验证高斯振幅中心是最大值"""
        amplitude = np.abs(gaussian_amplitude_64)
        center = amplitude.shape[0] // 2
        center_value = amplitude[center, center]
        assert center_value == np.max(amplitude)
    
    def test_spherical_wave_has_phase(self, spherical_wave_amplitude_64):
        """验证球面波有相位变化"""
        phase = np.angle(spherical_wave_amplitude_64)
        # 相位应该有变化（不是常数）
        assert np.std(phase) > 0
    
    def test_uniform_amplitude_is_constant(self, uniform_amplitude_64):
        """验证均匀振幅是常数"""
        assert np.all(uniform_amplitude_64 == 1.0)


class TestMirrorFixtures:
    """测试反射镜 fixtures"""
    
    def test_flat_mirror_is_plane(self, flat_mirror):
        """验证平面镜是平面"""
        assert flat_mirror.is_plane
        assert np.isinf(flat_mirror.radius)
    
    def test_flat_mirror_is_mirror(self, flat_mirror):
        """验证平面镜是反射镜"""
        assert flat_mirror.is_mirror
        assert flat_mirror.surface_type == 'mirror'
    
    def test_concave_mirror_f100_focal_length(self, concave_mirror_f100):
        """验证凹面镜焦距"""
        assert concave_mirror_f100.focal_length == 100.0
        assert concave_mirror_f100.radius == 200.0
    
    def test_concave_mirror_f50_focal_length(self, concave_mirror_f50):
        """验证凹面镜焦距"""
        assert concave_mirror_f50.focal_length == 50.0
        assert concave_mirror_f50.radius == 100.0
    
    def test_convex_mirror_f100_focal_length(self, convex_mirror_f100):
        """验证凸面镜焦距"""
        assert convex_mirror_f100.focal_length == -100.0
        assert convex_mirror_f100.radius == -200.0
    
    def test_fold_mirror_45deg_tilt(self, fold_mirror_45deg):
        """验证 45° 折叠镜倾斜角度"""
        assert fold_mirror_45deg.tilt_x == pytest.approx(np.pi / 4)
        assert fold_mirror_45deg.tilt_y == 0.0
        assert fold_mirror_45deg.is_plane
    
    def test_fold_mirror_30deg_tilt(self, fold_mirror_30deg):
        """验证 30° 折叠镜倾斜角度"""
        assert fold_mirror_30deg.tilt_x == pytest.approx(np.pi / 6)
    
    def test_tilted_concave_mirror_has_tilt(self, tilted_concave_mirror):
        """验证倾斜凹面镜有倾斜"""
        assert tilted_concave_mirror.tilt_x == pytest.approx(np.pi / 12)
        assert not tilted_concave_mirror.is_plane
    
    def test_parabolic_mirror_conic(self, parabolic_mirror_f100):
        """验证抛物面镜圆锥常数"""
        assert parabolic_mirror_f100.conic == -1.0


class TestParameterFixtures:
    """测试参数 fixtures"""
    
    def test_wavelength_visible(self, wavelength_visible):
        """验证可见光波长"""
        assert wavelength_visible == 0.633
    
    def test_wavelength_infrared(self, wavelength_infrared):
        """验证红外波长"""
        assert wavelength_infrared == 1.064
    
    def test_physical_sizes(self, physical_size_small, physical_size_medium, physical_size_large):
        """验证物理尺寸"""
        assert physical_size_small == 10.0
        assert physical_size_medium == 25.0
        assert physical_size_large == 50.0
    
    def test_num_rays(self, num_rays_small, num_rays_medium, num_rays_large):
        """验证光线数量"""
        assert num_rays_small == 25
        assert num_rays_medium == 100
        assert num_rays_large == 400


class TestDirectionFixtures:
    """测试方向 fixtures"""
    
    def test_normal_incidence_direction(self, normal_incidence_direction):
        """验证正入射方向"""
        L, M, N = normal_incidence_direction
        assert L == 0.0
        assert M == 0.0
        assert N == 1.0
    
    def test_tilted_incidence_45deg(self, tilted_incidence_45deg):
        """验证 45° 倾斜入射方向"""
        L, M, N = tilted_incidence_45deg
        assert L == 0.0
        assert M == pytest.approx(np.sin(np.pi / 4))
        assert N == pytest.approx(np.cos(np.pi / 4))
        # 验证归一化
        assert L**2 + M**2 + N**2 == pytest.approx(1.0)
    
    def test_tilted_incidence_30deg(self, tilted_incidence_30deg):
        """验证 30° 倾斜入射方向"""
        L, M, N = tilted_incidence_30deg
        assert L == 0.0
        assert M == pytest.approx(np.sin(np.pi / 6))
        assert N == pytest.approx(np.cos(np.pi / 6))
        # 验证归一化
        assert L**2 + M**2 + N**2 == pytest.approx(1.0)


class TestPhaseFixtures:
    """测试相位 fixtures"""
    
    def test_flat_phase_is_zero(self, flat_phase_64):
        """验证平坦相位为零"""
        assert flat_phase_64.shape == (64, 64)
        assert np.all(flat_phase_64 == 0)
    
    def test_linear_phase_varies(self, linear_phase_64):
        """验证线性相位有变化"""
        assert linear_phase_64.shape == (64, 64)
        assert np.std(linear_phase_64) > 0
    
    def test_quadratic_phase_varies(self, quadratic_phase_64):
        """验证二次相位有变化"""
        assert quadratic_phase_64.shape == (64, 64)
        assert np.std(quadratic_phase_64) > 0
    
    def test_random_phase_in_range(self, random_phase_64):
        """验证随机相位在范围内"""
        assert random_phase_64.shape == (64, 64)
        assert np.min(random_phase_64) >= -np.pi
        assert np.max(random_phase_64) <= np.pi


class TestHelperFixtures:
    """测试辅助函数 fixtures"""
    
    def test_compute_total_energy(self, compute_total_energy, gaussian_amplitude_64):
        """验证能量计算函数"""
        energy = compute_total_energy(gaussian_amplitude_64)
        assert energy > 0
        assert isinstance(energy, float)
    
    def test_compute_energy_ratio(self, compute_energy_ratio, gaussian_amplitude_64):
        """验证能量比计算函数"""
        # 相同输入应该得到比值 1.0
        ratio = compute_energy_ratio(gaussian_amplitude_64, gaussian_amplitude_64)
        assert ratio == pytest.approx(1.0)
    
    def test_assert_energy_conservation_passes(
        self, assert_energy_conservation, gaussian_amplitude_64
    ):
        """验证能量守恒断言函数（通过情况）"""
        # 相同输入应该通过
        assert_energy_conservation(gaussian_amplitude_64, gaussian_amplitude_64)
    
    def test_assert_energy_conservation_fails(
        self, assert_energy_conservation, gaussian_amplitude_64
    ):
        """验证能量守恒断言函数（失败情况）"""
        # 能量减半应该失败
        half_amplitude = gaussian_amplitude_64 * 0.5
        with pytest.raises(AssertionError):
            assert_energy_conservation(half_amplitude, gaussian_amplitude_64, rtol=0.01)


class TestParametrizedFixtures:
    """测试参数化 fixtures"""
    
    def test_gaussian_amplitude_parametrized(self, gaussian_amplitude_parametrized):
        """验证参数化高斯振幅"""
        amplitude, grid_size = gaussian_amplitude_parametrized
        assert amplitude.shape == (grid_size, grid_size)
        assert amplitude.dtype == np.complex128
    
    def test_mirror_parametrized(self, mirror_parametrized):
        """验证参数化反射镜"""
        mirror, mirror_type = mirror_parametrized
        assert isinstance(mirror, SurfaceDefinition)
        assert mirror.is_mirror
        assert mirror_type in ['flat', 'concave', 'convex', 'fold_45']
