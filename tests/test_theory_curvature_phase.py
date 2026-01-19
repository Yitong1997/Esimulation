"""
测试理论曲率相位计算

验证 SequentialOpticalSystem._compute_theory_curvature_phase() 方法的正确性。

验收标准：
- 平面镜返回零相位
- 聚焦元件返回正确的二次相位
- 相位单位为弧度

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')

from sequential_system import SequentialOpticalSystem, GaussianBeamSource
from gaussian_beam_simulation.optical_elements import SphericalMirror, ThinLens


class TestTheoryCurvaturePhase:
    """测试理论曲率相位计算"""
    
    @pytest.fixture
    def source(self):
        """创建测试光源"""
        return GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
    
    @pytest.fixture
    def system(self, source):
        """创建测试系统"""
        return SequentialOpticalSystem(
            source,
            grid_size=256,
            beam_ratio=0.5,
        )
    
    def test_plane_mirror_returns_zero_phase(self, system, source):
        """测试平面镜返回零相位（需求 3.2）
        
        平面镜的焦距为无穷大，不改变波前曲率，
        因此理论曲率相位应该为零。
        """
        import proper
        
        # 创建平面镜（焦距无穷大）
        plane_mirror = SphericalMirror(
            radius_of_curvature=np.inf,  # 平面镜
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, plane_mirror)
        
        # 验证：平面镜应返回全零相位
        assert theory_phase.shape == (system.grid_size, system.grid_size)
        np.testing.assert_array_equal(theory_phase, np.zeros_like(theory_phase))
    
    def test_concave_mirror_returns_correct_phase(self, system, source):
        """测试凹面镜返回正确的二次相位（需求 3.1）
        
        凹面镜（f > 0）应该返回负的二次相位（边缘波前滞后）。
        公式：φ = -k × r² / (2f)
        """
        import proper
        
        # 创建凹面镜（焦距 100mm）
        focal_length_mm = 100.0
        concave_mirror = SphericalMirror(
            radius_of_curvature=2 * focal_length_mm,  # R = 2f
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, concave_mirror)
        
        # 验证形状
        assert theory_phase.shape == (system.grid_size, system.grid_size)
        
        # 验证边缘相位为负（凹面镜，边缘波前滞后）
        center = system.grid_size // 2
        assert theory_phase[0, center] < 0  # 上边缘
        assert theory_phase[-1, center] < 0  # 下边缘
        assert theory_phase[center, 0] < 0  # 左边缘
        assert theory_phase[center, -1] < 0  # 右边缘
        
        # 验证相位公式正确性
        # 在特定位置验证：φ = -k × r² / (2f)
        n = system.grid_size
        sampling_m = proper.prop_get_sampling(wfo)
        half_size_m = sampling_m * n / 2
        
        # 创建与实现相同的坐标网格
        coords_m = np.linspace(-half_size_m, half_size_m, n)
        X_m, Y_m = np.meshgrid(coords_m, coords_m)
        r_sq_m = X_m**2 + Y_m**2
        
        # 计算期望的理论相位
        k = 2 * np.pi / wavelength_m
        focal_length_m = focal_length_mm * 1e-3
        expected_phase = -k * r_sq_m / (2 * focal_length_m)
        
        # 验证整个网格的相位
        np.testing.assert_array_almost_equal(theory_phase, expected_phase, decimal=10)
    
    def test_convex_mirror_returns_correct_phase(self, system, source):
        """测试凸面镜返回正确的二次相位
        
        凸面镜（f < 0）应该返回正的二次相位（边缘波前超前）。
        """
        import proper
        
        # 创建凸面镜（焦距 -100mm）
        focal_length_mm = -100.0
        convex_mirror = SphericalMirror(
            radius_of_curvature=2 * focal_length_mm,  # R = 2f
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, convex_mirror)
        
        # 验证边缘相位为正（凸面镜，边缘波前超前）
        center = system.grid_size // 2
        assert theory_phase[0, center] > 0  # 上边缘
        assert theory_phase[-1, center] > 0  # 下边缘
    
    def test_thin_lens_returns_correct_phase(self, system, source):
        """测试薄透镜返回正确的二次相位
        
        薄透镜的理论曲率相位与反射镜相同。
        """
        import proper
        
        # 创建会聚透镜（焦距 50mm）
        focal_length_mm = 50.0
        thin_lens = ThinLens(
            focal_length_value=focal_length_mm,
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, thin_lens)
        
        # 验证边缘相位为负（会聚透镜）
        center = system.grid_size // 2
        assert theory_phase[0, center] < 0
        
        # 验证相位公式正确性
        n = system.grid_size
        sampling_m = proper.prop_get_sampling(wfo)
        half_size_m = sampling_m * n / 2
        
        coords_m = np.linspace(-half_size_m, half_size_m, n)
        X_m, Y_m = np.meshgrid(coords_m, coords_m)
        r_sq_m = X_m**2 + Y_m**2
        
        k = 2 * np.pi / wavelength_m
        focal_length_m = focal_length_mm * 1e-3
        expected_phase = -k * r_sq_m / (2 * focal_length_m)
        
        np.testing.assert_array_almost_equal(theory_phase, expected_phase, decimal=10)
    
    def test_phase_is_radially_symmetric(self, system, source):
        """测试相位是径向对称的
        
        理论曲率相位只依赖于 r²，应该是径向对称的。
        """
        import proper
        
        # 创建凹面镜
        concave_mirror = SphericalMirror(
            radius_of_curvature=200.0,
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, concave_mirror)
        
        # 验证径向对称性：沿 x 和 y 方向相同距离处的相位应该相等
        center = system.grid_size // 2
        offset = 20
        
        # 比较 (+offset, 0) 和 (0, +offset) 处的相位
        phase_x = theory_phase[center, center + offset]
        phase_y = theory_phase[center + offset, center]
        np.testing.assert_almost_equal(phase_x, phase_y, decimal=10)
        
        # 比较对角线位置（相同 r² 的点）
        # 注意：对于偶数网格，需要选择相同 r² 的点
        # 使用 (center+offset, center) 和 (center, center+offset)
        # 它们的 r² 相同
        phase_diag1 = theory_phase[center + offset, center]
        phase_diag2 = theory_phase[center, center + offset]
        np.testing.assert_almost_equal(phase_diag1, phase_diag2, decimal=10)
    
    def test_phase_does_not_include_tilt(self, system, source):
        """测试相位不包含倾斜分量（需求 3.3）
        
        即使元件有倾斜，理论曲率相位也不应该包含倾斜分量。
        倾斜效果在其他地方处理。
        """
        import proper
        
        # 创建带倾斜的凹面镜
        tilted_mirror = SphericalMirror(
            radius_of_curvature=200.0,
            thickness=100.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,  # 45度倾斜
        )
        
        # 创建不带倾斜的凹面镜
        untilted_mirror = SphericalMirror(
            radius_of_curvature=200.0,
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算两种情况的理论曲率相位
        phase_tilted = system._compute_theory_curvature_phase(wfo, tilted_mirror)
        phase_untilted = system._compute_theory_curvature_phase(wfo, untilted_mirror)
        
        # 验证：两者应该完全相同（倾斜不影响理论曲率相位）
        np.testing.assert_array_almost_equal(phase_tilted, phase_untilted, decimal=10)



class TestTheoryCurvaturePhaseIntegration:
    """理论曲率相位的集成测试"""
    
    @pytest.fixture
    def source(self):
        """创建测试光源"""
        return GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
    
    def test_phase_magnitude_is_reasonable(self, source):
        """测试相位幅度在合理范围内
        
        对于典型的光学系统参数，边缘相位应该在合理范围内。
        """
        import proper
        
        system = SequentialOpticalSystem(source, grid_size=256)
        
        # 创建典型的凹面镜
        mirror = SphericalMirror(
            radius_of_curvature=200.0,  # f = 100mm
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, mirror)
        
        # 验证相位范围合理
        max_phase = np.max(np.abs(theory_phase))
        assert max_phase > 0  # 应该有非零相位
        assert max_phase < 10000  # 但不应该太大（数值稳定性）
    
    def test_phase_units_are_radians(self, source):
        """测试相位单位为弧度
        
        验证返回的相位值在弧度单位下是合理的。
        """
        import proper
        
        system = SequentialOpticalSystem(source, grid_size=256)
        
        # 创建一个已知参数的凹面镜
        focal_length_mm = 100.0
        mirror = SphericalMirror(
            radius_of_curvature=2 * focal_length_mm,
            thickness=100.0,
            semi_aperture=10.0,
        )
        
        # 初始化 PROPER 波前
        wavelength_m = source.wavelength * 1e-6
        w_init = source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system.grid_size, system.beam_ratio
        )
        
        # 计算理论曲率相位
        theory_phase = system._compute_theory_curvature_phase(wfo, mirror)
        
        # 在已知位置验证相位值
        # 选择网格边缘的一个点
        n = system.grid_size
        sampling_m = proper.prop_get_sampling(wfo)
        half_size_m = sampling_m * n / 2
        
        # 边缘位置的 r²
        r_edge_m = half_size_m
        r_sq_edge_m = r_edge_m**2
        
        # 期望的相位（弧度）
        k = 2 * np.pi / wavelength_m
        focal_length_m = focal_length_mm * 1e-3
        expected_phase_edge = -k * r_sq_edge_m / (2 * focal_length_m)
        
        # 获取实际的边缘相位（取中间行的最右边点）
        center = n // 2
        actual_phase_edge = theory_phase[center, -1]
        
        # 验证相位值在同一数量级
        # 由于边缘点不完全在 r = half_size 位置，允许一定误差
        assert abs(actual_phase_edge) > abs(expected_phase_edge) * 0.5
        assert abs(actual_phase_edge) < abs(expected_phase_edge) * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
