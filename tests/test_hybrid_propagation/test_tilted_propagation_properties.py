# -*- coding: utf-8 -*-
"""
TiltedPropagation 模块属性测试

本模块使用 hypothesis 库对 TiltedPropagation 类进行属性基测试（Property-Based Testing）。

测试的属性：
- Property 2: 旋转矩阵正交性
- Property 15: 正入射等价性

Validates: Requirements 1.2, 1.3, 8.2, 8.3
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, settings, assume

from hybrid_propagation.tilted_propagation import TiltedPropagation


# =============================================================================
# Property 2: 旋转矩阵正交性
# =============================================================================

class TestRotationMatrixOrthogonalityProperty:
    """
    Property 2: 旋转矩阵正交性
    
    *For any* 倾斜角度 (tilt_x, tilt_y)，计算得到的旋转矩阵 R 应满足：
    - R @ R.T = I（正交性）
    - det(R) = 1（行列式为 1）
    
    **Validates: Requirements 1.3, 8.2**
    """
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_rotation_matrix_orthogonality(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 1.3, 8.2**
        
        测试旋转矩阵满足正交性条件：R @ R.T = I
        
        对于任意倾斜角度 (tilt_x, tilt_y)，计算得到的旋转矩阵 R 应满足：
        R @ R.T = I（单位矩阵）
        
        这是旋转矩阵的基本性质，确保旋转变换保持向量长度不变。
        """
        # 计算旋转矩阵
        R = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
        
        # 验证正交性：R @ R.T = I
        identity = np.eye(3)
        product = R @ R.T
        
        assert_allclose(
            product, 
            identity, 
            atol=1e-10,
            err_msg=f"旋转矩阵不满足正交性 R @ R.T = I，tilt_x={tilt_x:.6f}, tilt_y={tilt_y:.6f}"
        )
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_rotation_matrix_determinant_is_one(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 1.3, 8.2**
        
        测试旋转矩阵行列式为 1：det(R) = 1
        
        对于任意倾斜角度 (tilt_x, tilt_y)，计算得到的旋转矩阵 R 应满足：
        det(R) = 1
        
        行列式为 1 确保这是一个纯旋转（不包含反射或缩放）。
        """
        # 计算旋转矩阵
        R = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
        
        # 验证行列式为 1
        det = np.linalg.det(R)
        
        assert_allclose(
            det, 
            1.0, 
            atol=1e-10,
            err_msg=f"旋转矩阵行列式不为 1，det(R)={det:.10f}, tilt_x={tilt_x:.6f}, tilt_y={tilt_y:.6f}"
        )
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_rotation_matrix_transpose_equals_inverse(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 1.3, 8.2**
        
        测试旋转矩阵的转置等于其逆：R.T = R^(-1)
        
        这是正交矩阵的另一个等价性质，验证 R.T @ R = I。
        """
        # 计算旋转矩阵
        R = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
        
        # 验证 R.T @ R = I
        identity = np.eye(3)
        product = R.T @ R
        
        assert_allclose(
            product, 
            identity, 
            atol=1e-10,
            err_msg=f"旋转矩阵不满足 R.T @ R = I，tilt_x={tilt_x:.6f}, tilt_y={tilt_y:.6f}"
        )


class TestExitRotationMatrixOrthogonalityProperty:
    """
    Property 2 扩展: 出射旋转矩阵正交性
    
    对于反射元件，出射旋转矩阵也应满足正交性条件。
    
    **Validates: Requirements 8.2**
    """
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_exit_rotation_matrix_orthogonality_reflective(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 8.2**
        
        测试反射元件的出射旋转矩阵满足正交性条件。
        """
        # 计算出射旋转矩阵（反射元件）
        R = TiltedPropagation._compute_exit_rotation_matrix(tilt_x, tilt_y, is_reflective=True)
        
        # 验证正交性：R @ R.T = I
        identity = np.eye(3)
        product = R @ R.T
        
        assert_allclose(
            product, 
            identity, 
            atol=1e-10,
            err_msg=f"出射旋转矩阵（反射）不满足正交性，tilt_x={tilt_x:.6f}, tilt_y={tilt_y:.6f}"
        )
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_exit_rotation_matrix_determinant_reflective(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 8.2**
        
        测试反射元件的出射旋转矩阵行列式为 1。
        """
        # 计算出射旋转矩阵（反射元件）
        R = TiltedPropagation._compute_exit_rotation_matrix(tilt_x, tilt_y, is_reflective=True)
        
        # 验证行列式为 1
        det = np.linalg.det(R)
        
        assert_allclose(
            det, 
            1.0, 
            atol=1e-10,
            err_msg=f"出射旋转矩阵（反射）行列式不为 1，det(R)={det:.10f}"
        )
    
    @settings(max_examples=100)
    @given(
        tilt_x=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
        tilt_y=st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False),
    )
    def test_exit_rotation_matrix_transmissive_is_identity(self, tilt_x: float, tilt_y: float):
        """
        **Feature: hybrid-element-propagation, Property 2: 旋转矩阵正交性**
        **Validates: Requirements 8.2**
        
        测试透射元件的出射旋转矩阵为单位矩阵。
        
        对于透射元件（薄元件近似），出射面与入射面平行，
        因此出射旋转矩阵应为单位矩阵。
        """
        # 计算出射旋转矩阵（透射元件）
        R = TiltedPropagation._compute_exit_rotation_matrix(tilt_x, tilt_y, is_reflective=False)
        
        # 验证为单位矩阵
        identity = np.eye(3)
        
        assert_allclose(
            R, 
            identity, 
            atol=1e-10,
            err_msg=f"透射元件出射旋转矩阵不是单位矩阵，tilt_x={tilt_x:.6f}, tilt_y={tilt_y:.6f}"
        )


# =============================================================================
# Property 15: 正入射等价性
# =============================================================================

class TestNormalIncidenceEquivalenceProperty:
    """
    Property 15: 正入射等价性
    
    *For any* 无倾斜的元件（tilt_x=0, tilt_y=0），入射面到切平面的传播
    应返回与输入相同的复振幅（在数值精度范围内）。
    
    **Validates: Requirements 1.2, 8.3**
    """
    
    @settings(max_examples=100)
    @given(
        wavelength=st.floats(min_value=0.3e-3, max_value=2.0e-3, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        grid_size=st.sampled_from([16, 32, 64]),
        beam_radius_ratio=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    )
    def test_normal_incidence_to_tangent_plane_equivalence(
        self, 
        wavelength: float, 
        dx: float, 
        dy: float, 
        grid_size: int,
        beam_radius_ratio: float,
    ):
        """
        **Feature: hybrid-element-propagation, Property 15: 正入射等价性**
        **Validates: Requirements 1.2**
        
        测试正入射情况下，propagate_to_tangent_plane 返回与输入相同的复振幅。
        
        当 tilt_x=0 且 tilt_y=0 时（正入射），入射面与切平面重合，
        因此传播应该返回输入复振幅的副本。
        """
        # 创建传播器
        propagator = TiltedPropagation(wavelength=wavelength, dx=dx, dy=dy)
        
        # 创建高斯光束复振幅
        physical_size = grid_size * dx
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, grid_size)
        y = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = physical_size * beam_radius_ratio
        amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
        
        # 正入射传播
        result = propagator.propagate_to_tangent_plane(
            amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
        )
        
        # 验证结果与输入相同
        assert_allclose(
            result, 
            amplitude, 
            rtol=1e-10,
            err_msg="正入射情况下，propagate_to_tangent_plane 结果与输入不同"
        )
        
        # 验证返回的是副本，不是同一个对象
        assert result is not amplitude, "正入射情况下应返回副本，而不是同一个对象"
    
    @settings(max_examples=100)
    @given(
        wavelength=st.floats(min_value=0.3e-3, max_value=2.0e-3, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        grid_size=st.sampled_from([16, 32, 64]),
        beam_radius_ratio=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    )
    def test_normal_incidence_from_tangent_plane_equivalence_reflective(
        self, 
        wavelength: float, 
        dx: float, 
        dy: float, 
        grid_size: int,
        beam_radius_ratio: float,
    ):
        """
        **Feature: hybrid-element-propagation, Property 15: 正入射等价性**
        **Validates: Requirements 8.3**
        
        测试正入射情况下，propagate_from_tangent_plane（反射元件）返回与输入相同的复振幅。
        
        当 tilt_x=0 且 tilt_y=0 时（正入射），切平面与出射面重合，
        因此传播应该返回输入复振幅的副本。
        """
        # 创建传播器
        propagator = TiltedPropagation(wavelength=wavelength, dx=dx, dy=dy)
        
        # 创建高斯光束复振幅
        physical_size = grid_size * dx
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, grid_size)
        y = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = physical_size * beam_radius_ratio
        amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
        
        # 正入射传播（反射元件）
        result = propagator.propagate_from_tangent_plane(
            amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
            is_reflective=True,
        )
        
        # 验证结果与输入相同
        assert_allclose(
            result, 
            amplitude, 
            rtol=1e-10,
            err_msg="正入射情况下，propagate_from_tangent_plane（反射）结果与输入不同"
        )
        
        # 验证返回的是副本
        assert result is not amplitude, "正入射情况下应返回副本，而不是同一个对象"
    
    @settings(max_examples=100)
    @given(
        wavelength=st.floats(min_value=0.3e-3, max_value=2.0e-3, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        grid_size=st.sampled_from([16, 32, 64]),
        beam_radius_ratio=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    )
    def test_normal_incidence_from_tangent_plane_equivalence_transmissive(
        self, 
        wavelength: float, 
        dx: float, 
        dy: float, 
        grid_size: int,
        beam_radius_ratio: float,
    ):
        """
        **Feature: hybrid-element-propagation, Property 15: 正入射等价性**
        **Validates: Requirements 8.3**
        
        测试正入射情况下，propagate_from_tangent_plane（透射元件）返回与输入相同的复振幅。
        """
        # 创建传播器
        propagator = TiltedPropagation(wavelength=wavelength, dx=dx, dy=dy)
        
        # 创建高斯光束复振幅
        physical_size = grid_size * dx
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, grid_size)
        y = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, y)
        R_sq = X**2 + Y**2
        beam_radius = physical_size * beam_radius_ratio
        amplitude = np.exp(-R_sq / beam_radius**2).astype(np.complex128)
        
        # 正入射传播（透射元件）
        result = propagator.propagate_from_tangent_plane(
            amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
            is_reflective=False,
        )
        
        # 验证结果与输入相同
        assert_allclose(
            result, 
            amplitude, 
            rtol=1e-10,
            err_msg="正入射情况下，propagate_from_tangent_plane（透射）结果与输入不同"
        )
        
        # 验证返回的是副本
        assert result is not amplitude, "正入射情况下应返回副本，而不是同一个对象"
    
    @settings(max_examples=100)
    @given(
        wavelength=st.floats(min_value=0.3e-3, max_value=2.0e-3, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        grid_size=st.sampled_from([16, 32, 64]),
        tiny_tilt=st.floats(min_value=1e-15, max_value=1e-11, allow_nan=False, allow_infinity=False),
    )
    def test_tiny_tilt_treated_as_normal_incidence(
        self, 
        wavelength: float, 
        dx: float, 
        dy: float, 
        grid_size: int,
        tiny_tilt: float,
    ):
        """
        **Feature: hybrid-element-propagation, Property 15: 正入射等价性**
        **Validates: Requirements 1.2, 8.3**
        
        测试极小倾斜角（< 1e-10）被视为正入射。
        
        当倾斜角非常小（接近数值精度极限）时，应该被视为正入射处理，
        直接返回输入复振幅的副本。
        """
        # 创建传播器
        propagator = TiltedPropagation(wavelength=wavelength, dx=dx, dy=dy)
        
        # 创建简单的复振幅
        amplitude = np.ones((grid_size, grid_size), dtype=np.complex128)
        
        # 极小倾斜传播
        result = propagator.propagate_to_tangent_plane(
            amplitude,
            tilt_x=tiny_tilt,
            tilt_y=tiny_tilt,
        )
        
        # 验证结果与输入相同（极小倾斜被视为正入射）
        assert_allclose(
            result, 
            amplitude, 
            rtol=1e-10,
            err_msg=f"极小倾斜角 {tiny_tilt} 应被视为正入射"
        )
    
    @settings(max_examples=100)
    @given(
        wavelength=st.floats(min_value=0.3e-3, max_value=2.0e-3, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        grid_size=st.sampled_from([16, 32, 64]),
        phase_tilt_x=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        phase_tilt_y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_normal_incidence_preserves_phase(
        self, 
        wavelength: float, 
        dx: float, 
        dy: float, 
        grid_size: int,
        phase_tilt_x: float,
        phase_tilt_y: float,
    ):
        """
        **Feature: hybrid-element-propagation, Property 15: 正入射等价性**
        **Validates: Requirements 1.2, 8.3**
        
        测试正入射情况下相位保持不变。
        
        对于带有任意相位分布的复振幅，正入射传播应该保持相位不变。
        """
        # 创建传播器
        propagator = TiltedPropagation(wavelength=wavelength, dx=dx, dy=dy)
        
        # 创建带有线性相位的复振幅
        physical_size = grid_size * dx
        half_size = physical_size / 2.0
        x = np.linspace(-half_size, half_size, grid_size)
        y = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 高斯振幅 + 线性相位
        R_sq = X**2 + Y**2
        beam_radius = physical_size * 0.25
        amplitude_mag = np.exp(-R_sq / beam_radius**2)
        phase = phase_tilt_x * X + phase_tilt_y * Y
        amplitude = amplitude_mag * np.exp(1j * phase)
        
        # 正入射传播
        result = propagator.propagate_to_tangent_plane(
            amplitude,
            tilt_x=0.0,
            tilt_y=0.0,
        )
        
        # 验证相位保持不变
        assert_allclose(
            np.angle(result), 
            np.angle(amplitude), 
            atol=1e-10,
            err_msg="正入射情况下相位应保持不变"
        )
        
        # 验证振幅保持不变
        assert_allclose(
            np.abs(result), 
            np.abs(amplitude), 
            rtol=1e-10,
            err_msg="正入射情况下振幅应保持不变"
        )


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--hypothesis-show-statistics'])
