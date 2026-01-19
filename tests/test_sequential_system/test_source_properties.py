"""
GaussianBeamSource 属性测试

使用 Hypothesis 进行属性基测试，验证：
- Property 1: 高斯光束参数计算正确性
- Property 2: 无效输入参数拒绝

**Validates: Requirements 1.2, 1.4, 1.7, 1.8, 1.9, 1.10**
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from sequential_system.source import GaussianBeamSource
from sequential_system.exceptions import SourceConfigurationError


# ============================================================================
# Property 1: 高斯光束参数计算正确性
# ============================================================================

@given(
    wavelength=st.floats(min_value=0.3, max_value=2.0, allow_nan=False, allow_infinity=False),
    w0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_rayleigh_range_formula(wavelength, w0, m2):
    """
    **Feature: sequential-optical-system, Property 1: 高斯光束参数计算正确性**
    **Validates: Requirements 1.8**
    
    验证瑞利距离计算公式：zR = π * w0² / (M² * λ)
    """
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, m2=m2)
    
    # 计算期望的瑞利距离
    wavelength_mm = wavelength * 1e-3
    expected_zR = np.pi * w0**2 / (m2 * wavelength_mm)
    
    np.testing.assert_allclose(source.zR, expected_zR, rtol=1e-10)


@given(
    wavelength=st.floats(min_value=0.3, max_value=2.0, allow_nan=False, allow_infinity=False),
    w0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    z0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    z=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_beam_radius_formula(wavelength, w0, z0, m2, z):
    """
    **Feature: sequential-optical-system, Property 1: 高斯光束参数计算正确性**
    **Validates: Requirements 1.9**
    
    验证光束半径计算公式：w(z) = w0 * sqrt(1 + ((z-z0)/zR)²)
    """
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=z0, m2=m2)
    
    # 计算期望的光束半径
    dz = z - z0
    expected_w = w0 * np.sqrt(1 + (dz / source.zR)**2)
    
    np.testing.assert_allclose(source.w(z), expected_w, rtol=1e-10)


@given(
    wavelength=st.floats(min_value=0.3, max_value=2.0, allow_nan=False, allow_infinity=False),
    w0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    z0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    z=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_wavefront_curvature_formula(wavelength, w0, z0, m2, z):
    """
    **Feature: sequential-optical-system, Property 1: 高斯光束参数计算正确性**
    **Validates: Requirements 1.10**
    
    验证波前曲率半径计算公式：R(z) = (z-z0) * (1 + (zR/(z-z0))²)
    """
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=z0, m2=m2)
    
    dz = z - z0
    
    # 在束腰附近，R 应该趋近无穷大
    if abs(dz) < 1e-10:
        assert np.isinf(source.R(z))
    else:
        expected_R = dz * (1 + (source.zR / dz)**2)
        np.testing.assert_allclose(source.R(z), expected_R, rtol=1e-10)


@given(
    wavelength=st.floats(min_value=0.3, max_value=2.0, allow_nan=False, allow_infinity=False),
    w0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    z0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_beam_radius_minimum_at_waist(wavelength, w0, z0, m2):
    """
    **Feature: sequential-optical-system, Property 1: 高斯光束参数计算正确性**
    **Validates: Requirements 1.9**
    
    验证光束半径在束腰处最小，等于 w0
    """
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=z0, m2=m2)
    
    # 在束腰处，光束半径应该等于 w0
    np.testing.assert_allclose(source.w(z0), w0, rtol=1e-10)
    
    # 在其他位置，光束半径应该大于 w0
    for offset in [-10.0, -1.0, 1.0, 10.0]:
        assert source.w(z0 + offset) >= w0


# ============================================================================
# Property 2: 无效输入参数拒绝
# ============================================================================

@given(
    wavelength=st.floats(max_value=0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_reject_non_positive_wavelength(wavelength):
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.2**
    
    验证拒绝非正波长
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=wavelength, w0=1.0)
    
    assert 'wavelength' in str(exc_info.value)


@given(
    w0=st.floats(max_value=0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_reject_non_positive_w0(w0):
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.4**
    
    验证拒绝非正束腰半径
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=w0)
    
    assert 'w0' in str(exc_info.value)


@given(
    m2=st.floats(max_value=0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_reject_m2_less_than_one(m2):
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.7**
    
    验证拒绝 M² < 1.0
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=1.0, m2=m2)
    
    assert 'm2' in str(exc_info.value)


def test_reject_nan_wavelength():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.2**
    
    验证拒绝 NaN 波长
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=float('nan'), w0=1.0)
    
    assert 'wavelength' in str(exc_info.value)


def test_reject_inf_wavelength():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.2**
    
    验证拒绝无穷大波长
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=float('inf'), w0=1.0)
    
    assert 'wavelength' in str(exc_info.value)


def test_reject_nan_w0():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.4**
    
    验证拒绝 NaN 束腰半径
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=float('nan'))
    
    assert 'w0' in str(exc_info.value)


def test_reject_inf_w0():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.4**
    
    验证拒绝无穷大束腰半径
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=float('inf'))
    
    assert 'w0' in str(exc_info.value)


def test_reject_nan_m2():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.7**
    
    验证拒绝 NaN M² 因子
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=1.0, m2=float('nan'))
    
    assert 'm2' in str(exc_info.value)


def test_reject_inf_m2():
    """
    **Feature: sequential-optical-system, Property 2: 无效输入参数拒绝**
    **Validates: Requirements 1.7**
    
    验证拒绝无穷大 M² 因子
    """
    with pytest.raises(SourceConfigurationError) as exc_info:
        GaussianBeamSource(wavelength=0.633, w0=1.0, m2=float('inf'))
    
    assert 'm2' in str(exc_info.value)


# ============================================================================
# 额外验证：to_gaussian_beam() 方法
# ============================================================================

@given(
    wavelength=st.floats(min_value=0.3, max_value=2.0, allow_nan=False, allow_infinity=False),
    w0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    z0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_to_gaussian_beam_preserves_parameters(wavelength, w0, z0, m2):
    """
    **Feature: sequential-optical-system, Property 1: 高斯光束参数计算正确性**
    **Validates: Requirements 1.1, 1.3, 1.5, 1.6**
    
    验证 to_gaussian_beam() 方法正确传递参数
    """
    source = GaussianBeamSource(wavelength=wavelength, w0=w0, z0=z0, m2=m2)
    beam = source.to_gaussian_beam()
    
    assert beam.wavelength == wavelength
    assert beam.w0 == w0
    assert beam.z0 == z0
    assert beam.m2 == m2
    assert beam.z_init == 0.0
    
    # 验证瑞利距离一致
    np.testing.assert_allclose(beam.zR, source.zR, rtol=1e-10)
