"""
单位转换属性基测试

使用 hypothesis 库验证单位转换函数的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.unit_conversion import (
    mm_to_m,
    m_to_mm,
    um_to_mm,
    mm_to_um,
    um_to_m,
    m_to_um,
    opd_waves_to_phase_rad,
    phase_rad_to_opd_waves,
    opd_mm_to_phase_rad,
    phase_rad_to_opd_mm,
    wavenumber_mm,
    wavenumber_m,
    wavelength_um_to_mm,
    wavelength_um_to_m,
    UnitConverter,
)


# ============================================================================
# 测试策略定义
# ============================================================================

# 长度值策略（避免极端值）
length_strategy = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.1, max_value=10.0)

# 相位策略
phase_strategy = st.floats(min_value=-100.0, max_value=100.0)

# OPD 策略（波长数）
opd_waves_strategy = st.floats(min_value=-100.0, max_value=100.0)


# ============================================================================
# Property 6: 单位转换正确性
# ============================================================================

@settings(max_examples=100)
@given(value_mm=length_strategy)
def test_property_6_mm_to_m_conversion(value_mm: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.3**
    
    mm_value × 1e-3 = m_value
    """
    value_m = mm_to_m(value_mm)
    expected = value_mm * 1e-3
    
    assert_allclose(
        value_m, expected, rtol=1e-15,
        err_msg="mm 到 m 转换不正确"
    )


@settings(max_examples=100)
@given(value_m=length_strategy)
def test_property_6_m_to_mm_conversion(value_m: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.3**
    
    m_value × 1e3 = mm_value
    """
    value_mm = m_to_mm(value_m)
    expected = value_m * 1e3
    
    assert_allclose(
        value_mm, expected, rtol=1e-15,
        err_msg="m 到 mm 转换不正确"
    )


@settings(max_examples=100)
@given(value_mm=length_strategy)
def test_property_6_mm_m_roundtrip(value_mm: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.3**
    
    mm → m → mm 应该得到原始值
    """
    value_m = mm_to_m(value_mm)
    value_mm_back = m_to_mm(value_m)
    
    assert_allclose(
        value_mm_back, value_mm, rtol=1e-14,
        err_msg="mm ↔ m 往返转换不一致"
    )


@settings(max_examples=100)
@given(value_um=length_strategy)
def test_property_6_um_to_mm_conversion(value_um: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.4**
    
    μm × 1e-3 = mm
    """
    value_mm = um_to_mm(value_um)
    expected = value_um * 1e-3
    
    assert_allclose(
        value_mm, expected, rtol=1e-15,
        err_msg="μm 到 mm 转换不正确"
    )


@settings(max_examples=100)
@given(value_um=length_strategy)
def test_property_6_um_to_m_conversion(value_um: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.5**
    
    μm × 1e-6 = m
    """
    value_m = um_to_m(value_um)
    expected = value_um * 1e-6
    
    assert_allclose(
        value_m, expected, rtol=1e-15,
        err_msg="μm 到 m 转换不正确"
    )


@settings(max_examples=100)
@given(opd_waves=opd_waves_strategy)
def test_property_6_opd_to_phase_conversion(opd_waves: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.6**
    
    相位（弧度）= 2π × OPD（波长数）
    """
    phase_rad = opd_waves_to_phase_rad(opd_waves)
    expected = 2 * np.pi * opd_waves
    
    assert_allclose(
        phase_rad, expected, rtol=1e-15,
        err_msg="OPD 到相位转换不正确"
    )


@settings(max_examples=100)
@given(phase_rad=phase_strategy)
def test_property_6_phase_to_opd_conversion(phase_rad: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.6**
    
    OPD（波长数）= 相位 / (2π)
    """
    opd_waves = phase_rad_to_opd_waves(phase_rad)
    expected = phase_rad / (2 * np.pi)
    
    assert_allclose(
        opd_waves, expected, rtol=1e-15,
        err_msg="相位到 OPD 转换不正确"
    )


@settings(max_examples=100)
@given(opd_waves=opd_waves_strategy)
def test_property_6_opd_phase_roundtrip(opd_waves: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.6**
    
    OPD → 相位 → OPD 应该得到原始值
    """
    phase_rad = opd_waves_to_phase_rad(opd_waves)
    opd_waves_back = phase_rad_to_opd_waves(phase_rad)
    
    assert_allclose(
        opd_waves_back, opd_waves, rtol=1e-14,
        err_msg="OPD ↔ 相位往返转换不一致"
    )


@settings(max_examples=100)
@given(
    opd_mm=st.floats(min_value=-10.0, max_value=10.0),
    wavelength_um=wavelength_strategy,
)
def test_property_6_opd_mm_to_phase_conversion(opd_mm: float, wavelength_um: float):
    """
    **Feature: hybrid-optical-propagation, Property 6: 单位转换正确性**
    **Validates: Requirements 12.7**
    
    相位 = 2π × opd_mm / wavelength_mm
    """
    wavelength_mm = wavelength_um * 1e-3
    phase_rad = opd_mm_to_phase_rad(opd_mm, wavelength_mm)
    expected = 2 * np.pi * opd_mm / wavelength_mm
    
    assert_allclose(
        phase_rad, expected, rtol=1e-14,
        err_msg="OPD(mm) 到相位转换不正确"
    )


# ============================================================================
# UnitConverter 类测试
# ============================================================================

@settings(max_examples=100)
@given(wavelength_um=wavelength_strategy)
def test_unit_converter_wavelength_properties(wavelength_um: float):
    """
    测试 UnitConverter 的波长属性。
    """
    converter = UnitConverter(wavelength_um)
    
    # 验证波长转换
    assert_allclose(
        converter.wavelength_mm,
        wavelength_um * 1e-3,
        rtol=1e-15,
    )
    assert_allclose(
        converter.wavelength_m,
        wavelength_um * 1e-6,
        rtol=1e-15,
    )
    
    # 验证波数
    assert_allclose(
        converter.k_mm,
        2 * np.pi / (wavelength_um * 1e-3),
        rtol=1e-14,
    )
    assert_allclose(
        converter.k_m,
        2 * np.pi / (wavelength_um * 1e-6),
        rtol=1e-14,
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    opd_waves=opd_waves_strategy,
)
def test_unit_converter_opd_phase_methods(wavelength_um: float, opd_waves: float):
    """
    测试 UnitConverter 的 OPD/相位转换方法。
    """
    converter = UnitConverter(wavelength_um)
    
    # OPD(波长数) → 相位 → OPD(波长数)
    phase = converter.opd_waves_to_phase(opd_waves)
    opd_back = converter.phase_to_opd_waves(phase)
    
    assert_allclose(
        opd_back, opd_waves, rtol=1e-14,
        err_msg="UnitConverter OPD ↔ 相位往返不一致"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    opd_mm=st.floats(min_value=-10.0, max_value=10.0),
)
def test_unit_converter_opd_mm_phase_methods(wavelength_um: float, opd_mm: float):
    """
    测试 UnitConverter 的 OPD(mm)/相位转换方法。
    """
    converter = UnitConverter(wavelength_um)
    
    # OPD(mm) → 相位 → OPD(mm)
    phase = converter.opd_mm_to_phase(opd_mm)
    opd_back = converter.phase_to_opd_mm(phase)
    
    assert_allclose(
        opd_back, opd_mm, rtol=1e-14,
        err_msg="UnitConverter OPD(mm) ↔ 相位往返不一致"
    )


# ============================================================================
# 数组转换测试
# ============================================================================

@settings(max_examples=50)
@given(
    values=st.lists(
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    )
)
def test_array_conversion(values):
    """
    测试数组转换的正确性。
    """
    arr = np.array(values)
    
    # mm → m → mm
    arr_m = mm_to_m(arr)
    arr_mm_back = m_to_mm(arr_m)
    
    assert_allclose(
        arr_mm_back, arr, rtol=1e-14,
        err_msg="数组 mm ↔ m 往返转换不一致"
    )
    
    # OPD → 相位 → OPD
    phase = opd_waves_to_phase_rad(arr)
    opd_back = phase_rad_to_opd_waves(phase)
    
    assert_allclose(
        opd_back, arr, rtol=1e-14,
        err_msg="数组 OPD ↔ 相位往返转换不一致"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

