"""
GaussianSource 属性基测试

使用 hypothesis 库验证 GaussianSource 类的正确性属性。

**Feature: matlab-style-api**
**Validates: Requirements 3.1, 3.2, 3.6**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from bts.source import GaussianSource


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
# 使用正浮点数，避免极端值
wavelength_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# 束腰半径策略（单位 mm）
# 使用正浮点数，避免极端值
waist_radius_strategy = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)

# 物理尺寸策略（单位 mm）
physical_size_strategy = st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)

# 网格大小策略
grid_size_strategy = st.integers(min_value=16, max_value=1024)


# ============================================================================
# Property 3: GaussianSource 参数存储正确性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_3_parameter_storage_basic(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 3: GaussianSource 参数存储正确性**
    **Validates: Requirements 3.1, 3.2**
    
    *For any* 有效的波长值 `wavelength_um`（正浮点数）和束腰半径 `w0_mm`（正浮点数），
    创建 `GaussianSource(wavelength_um, w0_mm)` 后，对象的 `wavelength_um` 和 `w0_mm` 
    属性应该等于传入的值。
    """
    # 创建 GaussianSource
    source = GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm)
    
    # 验证波长存储正确
    assert_allclose(
        source.wavelength_um,
        wavelength_um,
        rtol=1e-10,
        err_msg=f"波长存储不正确：期望 {wavelength_um}，实际 {source.wavelength_um}",
    )
    
    # 验证束腰半径存储正确
    assert_allclose(
        source.w0_mm,
        w0_mm,
        rtol=1e-10,
        err_msg=f"束腰半径存储不正确：期望 {w0_mm}，实际 {source.w0_mm}",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_3_parameter_storage_with_grid_size(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 3: GaussianSource 参数存储正确性**
    **Validates: Requirements 3.1, 3.2**
    
    验证带有 grid_size 参数时，wavelength_um 和 w0_mm 仍然正确存储。
    """
    # 创建 GaussianSource
    source = GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 验证波长存储正确
    assert_allclose(
        source.wavelength_um,
        wavelength_um,
        rtol=1e-10,
        err_msg=f"波长存储不正确：期望 {wavelength_um}，实际 {source.wavelength_um}",
    )
    
    # 验证束腰半径存储正确
    assert_allclose(
        source.w0_mm,
        w0_mm,
        rtol=1e-10,
        err_msg=f"束腰半径存储不正确：期望 {w0_mm}，实际 {source.w0_mm}",
    )
    
    # 验证网格大小存储正确
    assert source.grid_size == grid_size, (
        f"网格大小存储不正确：期望 {grid_size}，实际 {source.grid_size}"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    physical_size_mm=physical_size_strategy,
)
def test_property_3_parameter_storage_with_physical_size(
    wavelength_um: float,
    w0_mm: float,
    physical_size_mm: float,
):
    """
    **Feature: matlab-style-api, Property 3: GaussianSource 参数存储正确性**
    **Validates: Requirements 3.1, 3.2**
    
    验证带有 physical_size_mm 参数时，wavelength_um 和 w0_mm 仍然正确存储。
    """
    # 创建 GaussianSource
    source = GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        physical_size_mm=physical_size_mm,
    )
    
    # 验证波长存储正确
    assert_allclose(
        source.wavelength_um,
        wavelength_um,
        rtol=1e-10,
        err_msg=f"波长存储不正确：期望 {wavelength_um}，实际 {source.wavelength_um}",
    )
    
    # 验证束腰半径存储正确
    assert_allclose(
        source.w0_mm,
        w0_mm,
        rtol=1e-10,
        err_msg=f"束腰半径存储不正确：期望 {w0_mm}，实际 {source.w0_mm}",
    )
    
    # 验证物理尺寸存储正确（显式指定时）
    assert_allclose(
        source.physical_size_mm,
        physical_size_mm,
        rtol=1e-10,
        err_msg=f"物理尺寸存储不正确：期望 {physical_size_mm}，实际 {source.physical_size_mm}",
    )


# ============================================================================
# Property 4: physical_size_mm 默认值计算
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_4_physical_size_default_calculation(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 4: physical_size_mm 默认值计算**
    **Validates: Requirements 3.6**
    
    *For any* 有效的束腰半径 `w0_mm`（正浮点数），当创建 
    `GaussianSource(wavelength_um, w0_mm)` 时未指定 `physical_size_mm`，
    则 `physical_size_mm` 属性应该等于 `8 * w0_mm`。
    """
    # 创建 GaussianSource，不指定 physical_size_mm
    source = GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm)
    
    # 验证默认物理尺寸为 8 倍束腰半径
    expected_physical_size = 8.0 * w0_mm
    
    assert_allclose(
        source.physical_size_mm,
        expected_physical_size,
        rtol=1e-10,
        err_msg=(
            f"physical_size_mm 默认值计算不正确：\n"
            f"  w0_mm = {w0_mm}\n"
            f"  期望 physical_size_mm = 8 * {w0_mm} = {expected_physical_size}\n"
            f"  实际 physical_size_mm = {source.physical_size_mm}"
        ),
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_4_physical_size_default_with_other_params(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 4: physical_size_mm 默认值计算**
    **Validates: Requirements 3.6**
    
    验证即使指定了其他参数（如 grid_size），未指定 physical_size_mm 时
    仍然使用默认值 8 * w0_mm。
    """
    # 创建 GaussianSource，指定 grid_size 但不指定 physical_size_mm
    source = GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 验证默认物理尺寸为 8 倍束腰半径
    expected_physical_size = 8.0 * w0_mm
    
    assert_allclose(
        source.physical_size_mm,
        expected_physical_size,
        rtol=1e-10,
        err_msg=(
            f"physical_size_mm 默认值计算不正确：\n"
            f"  w0_mm = {w0_mm}\n"
            f"  期望 physical_size_mm = 8 * {w0_mm} = {expected_physical_size}\n"
            f"  实际 physical_size_mm = {source.physical_size_mm}"
        ),
    )


# ============================================================================
# 额外的正确性测试
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_z_rayleigh_calculation(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试瑞利距离计算：z_R = π * w0² / λ
    
    验证 GaussianSource 的 z_rayleigh_mm 属性计算正确。
    """
    # 创建 GaussianSource
    source = GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm)
    
    # 计算预期瑞利距离
    wavelength_mm = wavelength_um * 1e-3
    expected_z_rayleigh = np.pi * w0_mm**2 / wavelength_mm
    
    assert_allclose(
        source.z_rayleigh_mm,
        expected_z_rayleigh,
        rtol=1e-10,
        err_msg=(
            f"瑞利距离计算不正确：\n"
            f"  wavelength_um = {wavelength_um}\n"
            f"  w0_mm = {w0_mm}\n"
            f"  期望 z_rayleigh_mm = {expected_z_rayleigh}\n"
            f"  实际 z_rayleigh_mm = {source.z_rayleigh_mm}"
        ),
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_wavelength_mm_conversion(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试波长单位转换：wavelength_mm = wavelength_um * 1e-3
    
    验证 GaussianSource 的 wavelength_mm 属性计算正确。
    """
    # 创建 GaussianSource
    source = GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm)
    
    # 计算预期波长（mm）
    expected_wavelength_mm = wavelength_um * 1e-3
    
    assert_allclose(
        source.wavelength_mm,
        expected_wavelength_mm,
        rtol=1e-10,
        err_msg=(
            f"波长单位转换不正确：\n"
            f"  wavelength_um = {wavelength_um}\n"
            f"  期望 wavelength_mm = {expected_wavelength_mm}\n"
            f"  实际 wavelength_mm = {source.wavelength_mm}"
        ),
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_z0_mm_storage(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    测试束腰位置存储正确性。
    """
    # 创建 GaussianSource
    source = GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=z0_mm,
    )
    
    # 验证束腰位置存储正确
    assert_allclose(
        source.z0_mm,
        z0_mm,
        rtol=1e-10,
        err_msg=f"束腰位置存储不正确：期望 {z0_mm}，实际 {source.z0_mm}",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    beam_diam_fraction=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_beam_diam_fraction_storage(
    wavelength_um: float,
    w0_mm: float,
    beam_diam_fraction: float,
):
    """
    测试 beam_diam_fraction 参数存储正确性。
    """
    # 创建 GaussianSource
    source = GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        beam_diam_fraction=beam_diam_fraction,
    )
    
    # 验证 beam_diam_fraction 存储正确
    assert_allclose(
        source.beam_diam_fraction,
        beam_diam_fraction,
        rtol=1e-10,
        err_msg=f"beam_diam_fraction 存储不正确：期望 {beam_diam_fraction}，实际 {source.beam_diam_fraction}",
    )


# ============================================================================
# 参数验证测试
# ============================================================================

def test_invalid_wavelength_raises_error():
    """
    测试无效波长（非正数）应该抛出 ValueError。
    """
    with pytest.raises(ValueError, match="wavelength_um"):
        GaussianSource(wavelength_um=-1.0, w0_mm=5.0)
    
    with pytest.raises(ValueError, match="wavelength_um"):
        GaussianSource(wavelength_um=0.0, w0_mm=5.0)


def test_invalid_w0_raises_error():
    """
    测试无效束腰半径（非正数）应该抛出 ValueError。
    """
    with pytest.raises(ValueError, match="w0_mm"):
        GaussianSource(wavelength_um=0.633, w0_mm=-1.0)
    
    with pytest.raises(ValueError, match="w0_mm"):
        GaussianSource(wavelength_um=0.633, w0_mm=0.0)


def test_invalid_grid_size_raises_error():
    """
    测试无效网格大小应该抛出 ValueError。
    """
    with pytest.raises(ValueError, match="grid_size"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, grid_size=-1)
    
    with pytest.raises(ValueError, match="grid_size"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, grid_size=0)
    
    with pytest.raises(ValueError, match="grid_size"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, grid_size=256.5)


def test_invalid_physical_size_raises_error():
    """
    测试无效物理尺寸（非正数）应该抛出 ValueError。
    """
    with pytest.raises(ValueError, match="physical_size_mm"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, physical_size_mm=-1.0)
    
    with pytest.raises(ValueError, match="physical_size_mm"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, physical_size_mm=0.0)


def test_invalid_beam_diam_fraction_raises_error():
    """
    测试无效 beam_diam_fraction（非正数）应该抛出 ValueError。
    """
    with pytest.raises(ValueError, match="beam_diam_fraction"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, beam_diam_fraction=-0.5)
    
    with pytest.raises(ValueError, match="beam_diam_fraction"):
        GaussianSource(wavelength_um=0.633, w0_mm=5.0, beam_diam_fraction=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
