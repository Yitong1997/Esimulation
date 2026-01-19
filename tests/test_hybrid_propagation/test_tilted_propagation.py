# -*- coding: utf-8 -*-
"""
TiltedPropagation 模块单元测试

本模块测试 TiltedPropagation 类的功能，包括：
- 旋转矩阵计算正确性
- 正入射情况的等价性
- 能量守恒
- propagate_to_tangent_plane 方法

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hybrid_propagation.tilted_propagation import TiltedPropagation

# 检查 finufft 是否可用
try:
    import finufft
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False

# 跳过需要 finufft 的测试的装饰器
requires_finufft = pytest.mark.skipif(
    not HAS_FINUFFT,
    reason="finufft 库未安装，跳过倾斜传播测试"
)


class TestTiltedPropagationInit:
    """TiltedPropagation 初始化测试"""
    
    def test_valid_initialization(self):
        """测试有效参数初始化"""
        propagator = TiltedPropagation(
            wavelength=0.633e-3,  # 633 nm in mm
            dx=0.1,
            dy=0.1,
        )
        
        assert propagator.wavelength == 0.633e-3
        assert propagator.dx == 0.1
        assert propagator.dy == 0.1
    
    def test_invalid_wavelength_raises(self):
        """测试无效波长抛出异常"""
        with pytest.raises(ValueError, match="波长必须为正数"):
            TiltedPropagation(wavelength=-0.633e-3, dx=0.1, dy=0.1)
        
        with pytest.raises(ValueError, match="波长必须为正数"):
            TiltedPropagation(wavelength=0, dx=0.1, dy=0.1)
    
    def test_invalid_dx_raises(self):
        """测试无效 dx 抛出异常"""
        with pytest.raises(ValueError, match="x 方向采样间隔必须为正数"):
            TiltedPropagation(wavelength=0.633e-3, dx=-0.1, dy=0.1)
    
    def test_invalid_dy_raises(self):
        """测试无效 dy 抛出异常"""
        with pytest.raises(ValueError, match="y 方向采样间隔必须为正数"):
            TiltedPropagation(wavelength=0.633e-3, dx=0.1, dy=-0.1)


class TestRotationMatrix:
    """旋转矩阵计算测试"""
    
    def test_identity_rotation(self):
        """测试零倾斜返回单位矩阵
        
        Validates: Requirements 1.3
        """
        R = TiltedPropagation._compute_rotation_matrix(0.0, 0.0)
        
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_rotation_matrix_orthogonality(self):
        """测试旋转矩阵正交性：R @ R.T = I
        
        Validates: Requirements 1.3, 8.2
        """
        test_angles = [
            (0.0, 0.0),
            (np.pi/6, 0.0),
            (0.0, np.pi/6),
            (np.pi/4, np.pi/4),
            (-np.pi/4, np.pi/6),
        ]
        
        for tilt_x, tilt_y in test_angles:
            R = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
            
            # 检查正交性
            assert_allclose(R @ R.T, np.eye(3), atol=1e-10,
                           err_msg=f"旋转矩阵不正交: tilt_x={tilt_x}, tilt_y={tilt_y}")
    
    def test_rotation_matrix_determinant(self):
        """测试旋转矩阵行列式为 1
        
        Validates: Requirements 1.3, 8.2
        """
        test_angles = [
            (0.0, 0.0),
            (np.pi/6, 0.0),
            (0.0, np.pi/6),
            (np.pi/4, np.pi/4),
            (-np.pi/4, np.pi/6),
        ]
        
        for tilt_x, tilt_y in test_angles:
            R = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
            
            # 检查行列式
            det = np.linalg.det(R)
            assert_allclose(det, 1.0, atol=1e-10,
                           err_msg=f"旋转矩阵行列式不为 1: tilt_x={tilt_x}, tilt_y={tilt_y}")
    
    def test_rotation_x_axis_only(self):
        """测试仅绕 X 轴旋转
        
        Validates: Requirements 1.3
        """
        tilt_x = np.pi / 4  # 45 度
        R = TiltedPropagation._compute_rotation_matrix(tilt_x, 0.0)
        
        # 绕 X 轴旋转的矩阵形式
        expected = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_x), -np.sin(tilt_x)],
            [0, np.sin(tilt_x), np.cos(tilt_x)],
        ])
        
        assert_allclose(R, expected, atol=1e-10)
    
    def test_rotation_y_axis_only(self):
        """测试仅绕 Y 轴旋转
        
        Validates: Requirements 1.3
        """
        tilt_y = np.pi / 4  # 45 度
        R = TiltedPropagation._compute_rotation_matrix(0.0, tilt_y)
        
        # 绕 Y 轴旋转的矩阵形式
        expected = np.array([
            [np.cos(tilt_y), 0, np.sin(tilt_y)],
            [0, 1, 0],
            [-np.sin(tilt_y), 0, np.cos(tilt_y)],
        ])
        
        assert_allclose(R, expected, atol=1e-10)


class TestPropagateToTangentPlane:
    """propagate_to_tangent_plane 方法测试"""
    
    @pytest.fixture
    def propagator(self):
        """创建测试用传播器"""
        return TiltedPropagation(
            wavelength=0.633e-3,  # 633 nm in mm
            dx=0.1,
            dy=0.1,
        )
    
    @pytest.fixture
    def gaussian_amplitude(self):
        """创建高斯光束复振幅"""
        n = 64
        physical_size = 6.4  # mm (n * dx)
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, n)
        y = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = 1.5  # mm
        return np.exp(-R_sq / beam_radius**2).astype(np.complex128)
    
    def test_normal_incidence_returns_copy(self, propagator, gaussian_amplitude):
        """测试正入射情况返回输入复振幅的副本
        
        Validates: Requirements 1.2
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
        )
        
        # 应该返回相同的值
        assert_allclose(result, gaussian_amplitude, rtol=1e-10)
        
        # 但应该是副本，不是同一个对象
        assert result is not gaussian_amplitude
    
    def test_normal_incidence_with_tiny_tilt(self, propagator, gaussian_amplitude):
        """测试极小倾斜角被视为正入射
        
        Validates: Requirements 1.2
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=1e-12,
            tilt_y=1e-12,
        )
        
        # 极小倾斜应该被视为正入射
        assert_allclose(result, gaussian_amplitude, rtol=1e-10)
    
    @requires_finufft
    def test_output_shape_matches_input(self, propagator, gaussian_amplitude):
        """测试输出形状与输入相同
        
        Validates: Requirements 1.1
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
        )
        
        assert result.shape == gaussian_amplitude.shape
    
    @requires_finufft
    def test_output_is_complex(self, propagator, gaussian_amplitude):
        """测试输出是复数类型"""
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
        )
        
        assert np.iscomplexobj(result)
    
    @requires_finufft
    def test_energy_conservation(self, propagator, gaussian_amplitude):
        """测试能量守恒（误差 < 1%）
        
        Validates: Requirements 1.4
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,  # 30 度
            tilt_y=0.0,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"能量不守恒：比值 = {ratio:.4f}"
    
    @requires_finufft
    def test_energy_conservation_45deg(self, propagator, gaussian_amplitude):
        """测试 45 度倾斜时的能量守恒
        
        Validates: Requirements 1.4
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 4,  # 45 度
            tilt_y=0.0,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"能量不守恒：比值 = {ratio:.4f}"
    
    @requires_finufft
    def test_energy_conservation_combined_tilt(self, propagator, gaussian_amplitude):
        """测试组合倾斜时的能量守恒
        
        Validates: Requirements 1.4
        """
        result = propagator.propagate_to_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,  # 30 度
            tilt_y=np.pi / 8,  # 22.5 度
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"能量不守恒：比值 = {ratio:.4f}"
    
    def test_invalid_input_dimension_raises(self, propagator):
        """测试无效输入维度抛出异常"""
        # 1D 数组
        with pytest.raises(ValueError, match="必须是 2D 数组"):
            propagator.propagate_to_tangent_plane(
                np.ones(64, dtype=complex),
                tilt_x=0.0,
                tilt_y=0.0,
            )
        
        # 3D 数组
        with pytest.raises(ValueError, match="必须是 2D 数组"):
            propagator.propagate_to_tangent_plane(
                np.ones((64, 64, 3), dtype=complex),
                tilt_x=0.0,
                tilt_y=0.0,
            )
    
    @requires_finufft
    def test_different_grid_sizes(self, propagator):
        """测试不同网格大小
        
        Validates: Requirements 1.1
        """
        for grid_size in [32, 64, 128]:
            # 创建测试复振幅
            n = grid_size
            physical_size = n * propagator.dx
            half_size = physical_size / 2.0
            x = np.linspace(-half_size, half_size, n)
            y = np.linspace(-half_size, half_size, n)
            X, Y = np.meshgrid(x, y)
            R_sq = X**2 + Y**2
            beam_radius = physical_size / 4.0
            amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
            
            # 执行传播
            result = propagator.propagate_to_tangent_plane(
                amplitude,
                tilt_x=np.pi / 6,
                tilt_y=0.0,
            )
            
            # 检查输出形状
            assert result.shape == (grid_size, grid_size)
            
            # 检查能量守恒
            is_conserved, ratio = propagator.check_energy_conservation(
                amplitude, result, tolerance=0.01
            )
            assert is_conserved, f"网格大小 {grid_size}: 能量不守恒，比值 = {ratio:.4f}"


class TestPropagateFromTangentPlane:
    """propagate_from_tangent_plane 方法测试
    
    测试从切平面传播到出射面的功能，包括：
    - 正入射情况（tilt_x=0, tilt_y=0）直接返回输入复振幅
    - 反射元件的光轴方向变化处理
    - 能量守恒（误差 < 1%）
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    @pytest.fixture
    def propagator(self):
        """创建测试用传播器"""
        return TiltedPropagation(
            wavelength=0.633e-3,
            dx=0.1,
            dy=0.1,
        )
    
    @pytest.fixture
    def gaussian_amplitude(self):
        """创建高斯光束复振幅"""
        n = 64
        physical_size = 6.4
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, n)
        y = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = 1.5
        return np.exp(-R_sq / beam_radius**2).astype(np.complex128)
    
    # =========================================================================
    # 正入射情况测试 (Requirements 8.3)
    # =========================================================================
    
    def test_normal_incidence_returns_copy(self, propagator, gaussian_amplitude):
        """测试正入射情况返回输入复振幅的副本
        
        Validates: Requirements 8.3
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        assert_allclose(result, gaussian_amplitude, rtol=1e-10)
        assert result is not gaussian_amplitude
    
    def test_normal_incidence_transmissive(self, propagator, gaussian_amplitude):
        """测试透射元件正入射情况
        
        Validates: Requirements 8.3
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
            is_reflective=False,
        )
        
        assert_allclose(result, gaussian_amplitude, rtol=1e-10)
        assert result is not gaussian_amplitude
    
    def test_normal_incidence_with_tiny_tilt(self, propagator, gaussian_amplitude):
        """测试极小倾斜角被视为正入射
        
        Validates: Requirements 8.3
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=1e-12,
            tilt_y=1e-12,
            is_reflective=True,
        )
        
        # 极小倾斜应该被视为正入射
        assert_allclose(result, gaussian_amplitude, rtol=1e-10)
    
    def test_normal_incidence_preserves_phase(self, propagator):
        """测试正入射情况保持相位不变
        
        Validates: Requirements 8.3
        """
        n = 64
        physical_size = 6.4
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, n)
        y = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = 1.5
        
        # 创建带有相位的复振幅
        amplitude = np.exp(-R_sq / beam_radius**2)
        phase = 0.5 * (X + Y)  # 线性相位
        complex_amplitude = amplitude * np.exp(1j * phase)
        
        result = propagator.propagate_from_tangent_plane(
            complex_amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 检查相位保持不变
        assert_allclose(np.angle(result), np.angle(complex_amplitude), atol=1e-10)
    
    # =========================================================================
    # 输出形状和类型测试 (Requirements 8.1)
    # =========================================================================
    
    @requires_finufft
    def test_output_shape_matches_input(self, propagator, gaussian_amplitude):
        """测试输出形状与输入相同
        
        Validates: Requirements 8.1
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        assert result.shape == gaussian_amplitude.shape
    
    @requires_finufft
    def test_output_is_complex(self, propagator, gaussian_amplitude):
        """测试输出是复数类型
        
        Validates: Requirements 8.1
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        assert np.iscomplexobj(result)
    
    @requires_finufft
    def test_different_grid_sizes(self, propagator):
        """测试不同网格大小
        
        Validates: Requirements 8.1
        """
        for grid_size in [32, 64, 128]:
            # 创建测试复振幅
            n = grid_size
            physical_size = n * propagator.dx
            half_size = physical_size / 2.0
            x = np.linspace(-half_size, half_size, n)
            y = np.linspace(-half_size, half_size, n)
            X, Y = np.meshgrid(x, y)
            R_sq = X**2 + Y**2
            beam_radius = physical_size / 4.0
            amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
            
            # 执行传播
            result = propagator.propagate_from_tangent_plane(
                amplitude,
                tilt_x=np.pi / 6,
                tilt_y=0.0,
                is_reflective=True,
            )
            
            # 检查输出形状
            assert result.shape == (grid_size, grid_size)
    
    # =========================================================================
    # 能量守恒测试 (Requirements 8.4)
    # =========================================================================
    
    @requires_finufft
    def test_energy_conservation_reflective(self, propagator, gaussian_amplitude):
        """测试反射元件的能量守恒
        
        Validates: Requirements 8.4
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"能量不守恒：比值 = {ratio:.4f}"
    
    @requires_finufft
    def test_energy_conservation_45deg_fold_mirror(self, propagator, gaussian_amplitude):
        """测试 45 度折叠镜的能量守恒
        
        Validates: Requirements 8.4
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 4,  # 45 度
            tilt_y=0.0,
            is_reflective=True,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"45° 折叠镜能量不守恒：比值 = {ratio:.4f}"
    
    @requires_finufft
    def test_energy_conservation_combined_tilt(self, propagator, gaussian_amplitude):
        """测试组合倾斜时的能量守恒
        
        Validates: Requirements 8.4
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,  # 30 度
            tilt_y=np.pi / 8,  # 22.5 度
            is_reflective=True,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"组合倾斜能量不守恒：比值 = {ratio:.4f}"
    
    @requires_finufft
    def test_energy_conservation_transmissive(self, propagator, gaussian_amplitude):
        """测试透射元件的能量守恒
        
        Validates: Requirements 8.4
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=False,
        )
        
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        
        assert is_conserved, f"透射元件能量不守恒：比值 = {ratio:.4f}"
    
    # =========================================================================
    # 输入验证测试
    # =========================================================================
    
    def test_invalid_input_dimension_raises(self, propagator):
        """测试无效输入维度抛出异常"""
        # 1D 数组
        with pytest.raises(ValueError, match="必须是 2D 数组"):
            propagator.propagate_from_tangent_plane(
                np.ones(64, dtype=complex),
                tilt_x=0.0,
                tilt_y=0.0,
                is_reflective=True,
            )
        
        # 3D 数组
        with pytest.raises(ValueError, match="必须是 2D 数组"):
            propagator.propagate_from_tangent_plane(
                np.ones((64, 64, 3), dtype=complex),
                tilt_x=0.0,
                tilt_y=0.0,
                is_reflective=True,
            )
    
    # =========================================================================
    # 反射元件光轴方向变化测试 (Requirements 8.2)
    # =========================================================================
    
    @requires_finufft
    def test_reflective_uses_exit_rotation_matrix(self, propagator, gaussian_amplitude):
        """测试反射元件使用出射旋转矩阵
        
        验证反射元件调用 _compute_exit_rotation_matrix 计算正确的旋转矩阵
        
        Validates: Requirements 8.2
        """
        # 对于反射元件，出射旋转矩阵应该考虑反射后的光轴方向
        # 这里通过比较反射和透射的结果来间接验证
        
        result_reflective = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        result_transmissive = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,
            tilt_y=0.0,
            is_reflective=False,
        )
        
        # 反射和透射的结果应该不同（因为使用不同的旋转矩阵）
        # 透射元件使用单位矩阵，反射元件使用考虑反射的旋转矩阵
        assert not np.allclose(result_reflective, result_transmissive)
    
    @requires_finufft
    def test_30deg_tilt_reflective(self, propagator, gaussian_amplitude):
        """测试 30 度倾斜反射元件
        
        Validates: Requirements 8.1, 8.2
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=np.pi / 6,  # 30 度
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 检查输出有效
        assert result.shape == gaussian_amplitude.shape
        assert np.iscomplexobj(result)
        assert np.sum(np.abs(result)**2) > 0  # 有能量
    
    @requires_finufft
    def test_negative_tilt_reflective(self, propagator, gaussian_amplitude):
        """测试负倾斜角反射元件
        
        Validates: Requirements 8.1, 8.2
        """
        result = propagator.propagate_from_tangent_plane(
            gaussian_amplitude,
            tilt_x=-np.pi / 6,  # -30 度
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 检查输出有效
        assert result.shape == gaussian_amplitude.shape
        assert np.iscomplexobj(result)
        
        # 检查能量守恒
        is_conserved, ratio = propagator.check_energy_conservation(
            gaussian_amplitude, result, tolerance=0.01
        )
        assert is_conserved, f"负倾斜角能量不守恒：比值 = {ratio:.4f}"


class TestExitRotationMatrix:
    """出射旋转矩阵计算测试
    
    测试 _compute_exit_rotation_matrix 方法，验证：
    - 透射元件返回单位矩阵
    - 反射元件旋转矩阵正交性
    - 反射方向计算正确性
    
    Validates: Requirements 8.2
    """
    
    def test_transmissive_element_identity(self):
        """测试透射元件返回单位矩阵
        
        Validates: Requirements 8.2
        """
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=np.pi / 4,
            tilt_y=0.0,
            is_reflective=False,
        )
        
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_transmissive_element_identity_combined_tilt(self):
        """测试透射元件组合倾斜返回单位矩阵
        
        Validates: Requirements 8.2
        """
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=np.pi / 6,
            tilt_y=np.pi / 8,
            is_reflective=False,
        )
        
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_reflective_element_orthogonality(self):
        """测试反射元件旋转矩阵正交性
        
        Validates: Requirements 8.2
        """
        test_angles = [
            (np.pi/6, 0.0),
            (np.pi/4, 0.0),
            (0.0, np.pi/6),
            (np.pi/6, np.pi/6),
            (-np.pi/4, 0.0),
            (np.pi/8, -np.pi/8),
        ]
        
        for tilt_x, tilt_y in test_angles:
            R = TiltedPropagation._compute_exit_rotation_matrix(
                tilt_x, tilt_y, is_reflective=True
            )
            
            # 检查正交性
            assert_allclose(R @ R.T, np.eye(3), atol=1e-10,
                           err_msg=f"出射旋转矩阵不正交: tilt_x={tilt_x}, tilt_y={tilt_y}")
    
    def test_reflective_element_determinant(self):
        """测试反射元件旋转矩阵行列式为 1
        
        Validates: Requirements 8.2
        """
        test_angles = [
            (np.pi/6, 0.0),
            (np.pi/4, 0.0),
            (0.0, np.pi/6),
            (np.pi/6, np.pi/6),
        ]
        
        for tilt_x, tilt_y in test_angles:
            R = TiltedPropagation._compute_exit_rotation_matrix(
                tilt_x, tilt_y, is_reflective=True
            )
            
            # 检查行列式
            det = np.linalg.det(R)
            assert_allclose(det, 1.0, atol=1e-10,
                           err_msg=f"出射旋转矩阵行列式不为 1: tilt_x={tilt_x}, tilt_y={tilt_y}")
    
    def test_45deg_fold_mirror_reflection_direction(self):
        """测试 45 度折叠镜的反射方向
        
        入射光沿 +Z 方向，45 度折叠镜应将光反射到 -Y 方向
        
        根据 coordinate_conventions.md 的约定：
        - 入射光沿 +Z 方向
        - 45° 折叠镜：tilt_x = π/4
        - 表面法向量初始为 (0, 0, -1)
        - 绕 X 轴旋转 45° 后：(0, -sin(45°), -cos(45°)) = (0, -0.707, -0.707)
        - 反射后光线方向：(0, -1, 0)，沿 -Y 方向
        
        Validates: Requirements 8.2
        """
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=np.pi / 4,  # 45 度
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 出射面的 Z 轴应该沿反射方向
        # 对于 45 度折叠镜，反射方向应该是 (0, -1, 0)
        z_axis = R[:, 2]  # 第三列是 Z 轴
        expected_z = np.array([0.0, -1.0, 0.0])
        
        assert_allclose(z_axis, expected_z, atol=1e-10)
    
    def test_30deg_tilt_reflection_direction(self):
        """测试 30 度倾斜镜的反射方向
        
        入射光沿 +Z 方向，30 度倾斜镜的反射方向计算
        
        Validates: Requirements 8.2
        """
        tilt_x = np.pi / 6  # 30 度
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=tilt_x,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 计算预期的反射方向
        # 入射方向
        incident = np.array([0.0, 0.0, 1.0])
        
        # 表面法向量（初始为 (0, 0, -1)，绕 X 轴旋转 tilt_x）
        # 使用被动旋转
        R_tilt = TiltedPropagation._compute_rotation_matrix(tilt_x, 0.0)
        initial_normal = np.array([0.0, 0.0, -1.0])
        surface_normal = R_tilt.T @ initial_normal
        
        # 反射方向：r = d - 2(d·n)n
        d_dot_n = np.dot(incident, surface_normal)
        expected_reflected = incident - 2 * d_dot_n * surface_normal
        expected_reflected = expected_reflected / np.linalg.norm(expected_reflected)
        
        # 出射面的 Z 轴应该沿反射方向
        z_axis = R[:, 2]
        
        assert_allclose(z_axis, expected_reflected, atol=1e-10)
    
    def test_negative_tilt_reflection_direction(self):
        """测试负倾斜角的反射方向
        
        Validates: Requirements 8.2
        """
        tilt_x = -np.pi / 4  # -45 度
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=tilt_x,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 出射面的 Z 轴应该沿反射方向
        # 对于 -45 度倾斜，反射方向应该是 (0, 1, 0)，沿 +Y 方向
        z_axis = R[:, 2]
        expected_z = np.array([0.0, 1.0, 0.0])
        
        assert_allclose(z_axis, expected_z, atol=1e-10)
    
    def test_y_axis_tilt_reflection_direction(self):
        """测试绕 Y 轴倾斜的反射方向
        
        Validates: Requirements 8.2
        """
        tilt_y = np.pi / 4  # 45 度
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=0.0,
            tilt_y=tilt_y,
            is_reflective=True,
        )
        
        # 计算预期的反射方向
        incident = np.array([0.0, 0.0, 1.0])
        
        # 表面法向量（初始为 (0, 0, -1)，绕 Y 轴旋转 tilt_y）
        R_tilt = TiltedPropagation._compute_rotation_matrix(0.0, tilt_y)
        initial_normal = np.array([0.0, 0.0, -1.0])
        surface_normal = R_tilt.T @ initial_normal
        
        # 反射方向
        d_dot_n = np.dot(incident, surface_normal)
        expected_reflected = incident - 2 * d_dot_n * surface_normal
        expected_reflected = expected_reflected / np.linalg.norm(expected_reflected)
        
        # 出射面的 Z 轴应该沿反射方向
        z_axis = R[:, 2]
        
        assert_allclose(z_axis, expected_reflected, atol=1e-10)
    
    def test_combined_tilt_reflection_direction(self):
        """测试组合倾斜的反射方向
        
        Validates: Requirements 8.2
        """
        tilt_x = np.pi / 6  # 30 度
        tilt_y = np.pi / 8  # 22.5 度
        R = TiltedPropagation._compute_exit_rotation_matrix(
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            is_reflective=True,
        )
        
        # 计算预期的反射方向
        incident = np.array([0.0, 0.0, 1.0])
        
        # 表面法向量
        R_tilt = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
        initial_normal = np.array([0.0, 0.0, -1.0])
        surface_normal = R_tilt.T @ initial_normal
        
        # 反射方向
        d_dot_n = np.dot(incident, surface_normal)
        expected_reflected = incident - 2 * d_dot_n * surface_normal
        expected_reflected = expected_reflected / np.linalg.norm(expected_reflected)
        
        # 出射面的 Z 轴应该沿反射方向
        z_axis = R[:, 2]
        
        assert_allclose(z_axis, expected_reflected, atol=1e-10)
    
    def test_exit_coordinate_system_right_handed(self):
        """测试出射坐标系是右手系
        
        Validates: Requirements 8.2
        """
        test_angles = [
            (np.pi/6, 0.0),
            (np.pi/4, 0.0),
            (0.0, np.pi/6),
            (np.pi/6, np.pi/6),
        ]
        
        for tilt_x, tilt_y in test_angles:
            R = TiltedPropagation._compute_exit_rotation_matrix(
                tilt_x, tilt_y, is_reflective=True
            )
            
            # 提取坐标轴
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]
            
            # 检查右手系：x × y = z
            cross_product = np.cross(x_axis, y_axis)
            assert_allclose(cross_product, z_axis, atol=1e-10,
                           err_msg=f"出射坐标系不是右手系: tilt_x={tilt_x}, tilt_y={tilt_y}")


class TestEnergyComputation:
    """能量计算测试"""
    
    @pytest.fixture
    def propagator(self):
        """创建测试用传播器"""
        return TiltedPropagation(
            wavelength=0.633e-3,
            dx=0.1,
            dy=0.1,
        )
    
    def test_compute_energy_uniform(self, propagator):
        """测试均匀振幅的能量计算"""
        amplitude = np.ones((64, 64), dtype=complex)
        energy = propagator.compute_energy(amplitude)
        
        # 期望能量 = 64 * 64 * 1^2 * (0.1 * 0.1) = 40.96
        expected = 64 * 64 * 0.1 * 0.1
        assert_allclose(energy, expected, rtol=1e-10)
    
    def test_compute_energy_gaussian(self, propagator):
        """测试高斯振幅的能量计算"""
        n = 64
        physical_size = 6.4
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, n)
        y = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = 1.5
        amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
        
        energy = propagator.compute_energy(amplitude)
        
        # 能量应该是正数
        assert energy > 0
    
    def test_check_energy_conservation_pass(self, propagator):
        """测试能量守恒检查通过"""
        amplitude = np.ones((64, 64), dtype=complex)
        
        is_conserved, ratio = propagator.check_energy_conservation(
            amplitude, amplitude, tolerance=0.01
        )
        
        assert is_conserved
        assert_allclose(ratio, 1.0, rtol=1e-10)
    
    def test_check_energy_conservation_fail(self, propagator):
        """测试能量守恒检查失败"""
        amplitude_in = np.ones((64, 64), dtype=complex)
        amplitude_out = 0.5 * amplitude_in  # 能量减少到 25%
        
        is_conserved, ratio = propagator.check_energy_conservation(
            amplitude_in, amplitude_out, tolerance=0.01
        )
        
        assert not is_conserved
        assert_allclose(ratio, 0.25, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
