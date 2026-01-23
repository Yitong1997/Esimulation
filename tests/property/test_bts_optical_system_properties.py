"""
OpticalSystem 属性基测试

使用 hypothesis 库验证 OpticalSystem 类的正确性属性。

**Feature: matlab-style-api**
**Validates: Requirements 4.3**
"""

import io
import sys
from contextlib import redirect_stdout
from typing import List, Tuple

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, 'src')

from bts.optical_system import OpticalSystem, SurfaceDefinition


# ============================================================================
# 测试策略定义
# ============================================================================

# Z 位置策略（单位 mm）
z_position_strategy = st.floats(
    min_value=1.0, max_value=1000.0, 
    allow_nan=False, allow_infinity=False
)

# 曲率半径策略（单位 mm）
# 包含正值、负值和无穷大
radius_strategy = st.one_of(
    st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10000.0, max_value=-10.0, allow_nan=False, allow_infinity=False),
    st.just(float('inf')),
)

# 半口径策略（单位 mm）
semi_aperture_strategy = st.floats(
    min_value=1.0, max_value=100.0,
    allow_nan=False, allow_infinity=False
)

# 倾斜角度策略（单位：度）
tilt_angle_strategy = st.floats(
    min_value=-89.0, max_value=89.0,
    allow_nan=False, allow_infinity=False
)

# 焦距策略（单位 mm）
focal_length_strategy = st.floats(
    min_value=10.0, max_value=10000.0,
    allow_nan=False, allow_infinity=False
)


# ============================================================================
# 表面生成策略
# ============================================================================

@st.composite
def flat_mirror_strategy(draw) -> Tuple[float, float, float, float]:
    """生成平面镜参数：(z, tilt_x, tilt_y, semi_aperture)"""
    z = draw(z_position_strategy)
    tilt_x = draw(tilt_angle_strategy)
    tilt_y = draw(tilt_angle_strategy)
    semi_aperture = draw(semi_aperture_strategy)
    return (z, tilt_x, tilt_y, semi_aperture)


@st.composite
def spherical_mirror_strategy(draw) -> Tuple[float, float, float, float, float]:
    """生成球面镜参数：(z, radius, tilt_x, tilt_y, semi_aperture)"""
    z = draw(z_position_strategy)
    radius = draw(st.floats(
        min_value=10.0, max_value=10000.0,
        allow_nan=False, allow_infinity=False
    ))
    tilt_x = draw(tilt_angle_strategy)
    tilt_y = draw(tilt_angle_strategy)
    semi_aperture = draw(semi_aperture_strategy)
    return (z, radius, tilt_x, tilt_y, semi_aperture)


@st.composite
def paraxial_lens_strategy(draw) -> Tuple[float, float, float]:
    """生成薄透镜参数：(z, focal_length, semi_aperture)"""
    z = draw(z_position_strategy)
    focal_length = draw(focal_length_strategy)
    semi_aperture = draw(semi_aperture_strategy)
    return (z, focal_length, semi_aperture)


@st.composite
def surface_type_strategy(draw) -> str:
    """生成表面类型"""
    return draw(st.sampled_from(['flat_mirror', 'spherical_mirror', 'paraxial_lens']))


@st.composite
def optical_system_with_surfaces_strategy(draw, min_surfaces: int = 1, max_surfaces: int = 5):
    """生成包含随机表面的 OpticalSystem"""
    num_surfaces = draw(st.integers(min_value=min_surfaces, max_value=max_surfaces))
    system = OpticalSystem("Test System")
    
    for _ in range(num_surfaces):
        surface_type = draw(surface_type_strategy())
        
        if surface_type == 'flat_mirror':
            z, tilt_x, tilt_y, semi_aperture = draw(flat_mirror_strategy())
            system.add_flat_mirror(z=z, tilt_x=tilt_x, tilt_y=tilt_y, semi_aperture=semi_aperture)
        elif surface_type == 'spherical_mirror':
            z, radius, tilt_x, tilt_y, semi_aperture = draw(spherical_mirror_strategy())
            system.add_spherical_mirror(z=z, radius=radius, tilt_x=tilt_x, tilt_y=tilt_y, semi_aperture=semi_aperture)
        else:  # paraxial_lens
            z, focal_length, semi_aperture = draw(paraxial_lens_strategy())
            system.add_paraxial_lens(z=z, focal_length=focal_length, semi_aperture=semi_aperture)
    
    return system


# ============================================================================
# Property 5: print_info 输出包含必要信息
# ============================================================================

@settings(max_examples=100)
@given(system=optical_system_with_surfaces_strategy(min_surfaces=1, max_surfaces=5))
def test_property_5_print_info_contains_surface_types(system: OpticalSystem):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    *For any* 包含至少一个表面的 `OpticalSystem`，调用 `print_info()` 的输出
    应该包含每个表面的类型信息（如 'standard', 'paraxial'）。
    """
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证输出包含每个表面的类型
    for surface in system._surfaces:
        # 检查表面类型是否出现在输出中
        assert surface.surface_type in output_text, (
            f"print_info 输出缺少表面类型信息：\n"
            f"  表面索引: {surface.index}\n"
            f"  期望类型: {surface.surface_type}\n"
            f"  输出内容:\n{output_text}"
        )


@settings(max_examples=100)
@given(system=optical_system_with_surfaces_strategy(min_surfaces=1, max_surfaces=5))
def test_property_5_print_info_contains_z_positions(system: OpticalSystem):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    *For any* 包含至少一个表面的 `OpticalSystem`，调用 `print_info()` 的输出
    应该包含每个表面的位置信息（z 坐标）。
    """
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证输出包含每个表面的 z 位置
    for surface in system._surfaces:
        # 格式化 z 值（与 print_info 中的格式一致）
        z_formatted = f"{surface.z:.3f}"
        
        assert z_formatted in output_text, (
            f"print_info 输出缺少表面位置信息：\n"
            f"  表面索引: {surface.index}\n"
            f"  期望 z 位置: {z_formatted} mm\n"
            f"  输出内容:\n{output_text}"
        )


@settings(max_examples=100)
@given(system=optical_system_with_surfaces_strategy(min_surfaces=1, max_surfaces=5))
def test_property_5_print_info_contains_surface_count(system: OpticalSystem):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    验证 print_info 输出包含正确的表面数量信息。
    """
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证输出包含表面数量
    expected_count = len(system._surfaces)
    assert f"表面数量: {expected_count}" in output_text, (
        f"print_info 输出缺少表面数量信息：\n"
        f"  期望表面数量: {expected_count}\n"
        f"  输出内容:\n{output_text}"
    )


# ============================================================================
# 平面镜特征信息测试
# ============================================================================

@settings(max_examples=100)
@given(
    z=z_position_strategy,
    tilt_x=tilt_angle_strategy,
    tilt_y=tilt_angle_strategy,
    semi_aperture=semi_aperture_strategy,
)
def test_property_5_flat_mirror_info(
    z: float,
    tilt_x: float,
    tilt_y: float,
    semi_aperture: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    对于平面镜，print_info 输出应该包含：
    - 表面类型 'standard'
    - z 位置
    - 曲率半径为无穷大（平面）的标识
    - 反射镜标识
    """
    system = OpticalSystem("Flat Mirror Test")
    system.add_flat_mirror(z=z, tilt_x=tilt_x, tilt_y=tilt_y, semi_aperture=semi_aperture)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证表面类型
    assert 'standard' in output_text, (
        f"平面镜输出缺少表面类型 'standard'：\n{output_text}"
    )
    
    # 验证 z 位置
    z_formatted = f"{z:.3f}"
    assert z_formatted in output_text, (
        f"平面镜输出缺少 z 位置 {z_formatted}：\n{output_text}"
    )
    
    # 验证平面标识（无穷大曲率半径）
    assert '平面' in output_text or '无穷大' in output_text, (
        f"平面镜输出缺少平面标识：\n{output_text}"
    )
    
    # 验证反射镜标识
    assert '反射镜' in output_text, (
        f"平面镜输出缺少反射镜标识：\n{output_text}"
    )


# ============================================================================
# 球面镜特征信息测试
# ============================================================================

@settings(max_examples=100)
@given(
    z=z_position_strategy,
    radius=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    tilt_x=tilt_angle_strategy,
    tilt_y=tilt_angle_strategy,
    semi_aperture=semi_aperture_strategy,
)
def test_property_5_spherical_mirror_info(
    z: float,
    radius: float,
    tilt_x: float,
    tilt_y: float,
    semi_aperture: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    对于球面镜，print_info 输出应该包含：
    - 表面类型 'standard'
    - z 位置
    - 曲率半径值
    - 反射镜标识
    """
    system = OpticalSystem("Spherical Mirror Test")
    system.add_spherical_mirror(z=z, radius=radius, tilt_x=tilt_x, tilt_y=tilt_y, semi_aperture=semi_aperture)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证表面类型
    assert 'standard' in output_text, (
        f"球面镜输出缺少表面类型 'standard'：\n{output_text}"
    )
    
    # 验证 z 位置
    z_formatted = f"{z:.3f}"
    assert z_formatted in output_text, (
        f"球面镜输出缺少 z 位置 {z_formatted}：\n{output_text}"
    )
    
    # 验证曲率半径值
    radius_formatted = f"{radius:.3f}"
    assert radius_formatted in output_text, (
        f"球面镜输出缺少曲率半径 {radius_formatted}：\n{output_text}"
    )
    
    # 验证反射镜标识
    assert '反射镜' in output_text, (
        f"球面镜输出缺少反射镜标识：\n{output_text}"
    )


# ============================================================================
# 薄透镜特征信息测试
# ============================================================================

@settings(max_examples=100)
@given(
    z=z_position_strategy,
    focal_length=focal_length_strategy,
    semi_aperture=semi_aperture_strategy,
)
def test_property_5_paraxial_lens_info(
    z: float,
    focal_length: float,
    semi_aperture: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    对于薄透镜，print_info 输出应该包含：
    - 表面类型 'paraxial'
    - z 位置
    - 焦距值
    """
    system = OpticalSystem("Paraxial Lens Test")
    system.add_paraxial_lens(z=z, focal_length=focal_length, semi_aperture=semi_aperture)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证表面类型
    assert 'paraxial' in output_text, (
        f"薄透镜输出缺少表面类型 'paraxial'：\n{output_text}"
    )
    
    # 验证 z 位置
    z_formatted = f"{z:.3f}"
    assert z_formatted in output_text, (
        f"薄透镜输出缺少 z 位置 {z_formatted}：\n{output_text}"
    )
    
    # 验证焦距值
    focal_length_formatted = f"{focal_length:.3f}"
    assert focal_length_formatted in output_text, (
        f"薄透镜输出缺少焦距 {focal_length_formatted}：\n{output_text}"
    )


# ============================================================================
# 倾斜角度信息测试
# ============================================================================

@settings(max_examples=100)
@given(
    z=z_position_strategy,
    tilt_x=st.floats(min_value=1.0, max_value=89.0, allow_nan=False, allow_infinity=False),
    tilt_y=st.floats(min_value=1.0, max_value=89.0, allow_nan=False, allow_infinity=False),
)
def test_property_5_tilt_angles_displayed_when_nonzero(
    z: float,
    tilt_x: float,
    tilt_y: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    当表面有非零倾斜角度时，print_info 输出应该包含倾斜信息。
    """
    system = OpticalSystem("Tilted Mirror Test")
    system.add_flat_mirror(z=z, tilt_x=tilt_x, tilt_y=tilt_y)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证倾斜信息存在
    assert '倾斜' in output_text or 'tilt' in output_text.lower(), (
        f"倾斜表面输出缺少倾斜信息：\n{output_text}"
    )
    
    # 验证 tilt_x 值
    tilt_x_formatted = f"{tilt_x:.2f}"
    assert tilt_x_formatted in output_text, (
        f"输出缺少 tilt_x 值 {tilt_x_formatted}：\n{output_text}"
    )
    
    # 验证 tilt_y 值
    tilt_y_formatted = f"{tilt_y:.2f}"
    assert tilt_y_formatted in output_text, (
        f"输出缺少 tilt_y 值 {tilt_y_formatted}：\n{output_text}"
    )


# ============================================================================
# 多表面系统测试
# ============================================================================

@settings(max_examples=100)
@given(
    z1=st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    z2=st.floats(min_value=110.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    z3=st.floats(min_value=210.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    radius=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    focal_length=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_property_5_mixed_surface_types(
    z1: float,
    z2: float,
    z3: float,
    radius: float,
    focal_length: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    对于包含多种类型表面的系统，print_info 应该正确显示所有表面的信息。
    """
    system = OpticalSystem("Mixed System")
    system.add_flat_mirror(z=z1, tilt_x=45.0)
    system.add_spherical_mirror(z=z2, radius=radius)
    system.add_paraxial_lens(z=z3, focal_length=focal_length)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证表面数量
    assert "表面数量: 3" in output_text, (
        f"输出缺少正确的表面数量：\n{output_text}"
    )
    
    # 验证所有表面类型都出现
    assert 'standard' in output_text, (
        f"输出缺少 'standard' 类型：\n{output_text}"
    )
    assert 'paraxial' in output_text, (
        f"输出缺少 'paraxial' 类型：\n{output_text}"
    )
    
    # 验证所有 z 位置都出现
    for z in [z1, z2, z3]:
        z_formatted = f"{z:.3f}"
        assert z_formatted in output_text, (
            f"输出缺少 z 位置 {z_formatted}：\n{output_text}"
        )


# ============================================================================
# 空系统测试
# ============================================================================

def test_print_info_empty_system():
    """
    测试空系统的 print_info 输出。
    """
    system = OpticalSystem("Empty System")
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证系统名称
    assert "Empty System" in output_text, (
        f"输出缺少系统名称：\n{output_text}"
    )
    
    # 验证表面数量为 0
    assert "表面数量: 0" in output_text, (
        f"输出缺少表面数量信息：\n{output_text}"
    )


# ============================================================================
# 系统名称测试
# ============================================================================

@settings(max_examples=100)
@given(
    name=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S'),
        whitelist_characters=' _-'
    )),
)
def test_print_info_contains_system_name(name: str):
    """
    验证 print_info 输出包含系统名称。
    """
    # 过滤掉只包含空白字符的名称
    assume(name.strip())
    
    system = OpticalSystem(name)
    system.add_flat_mirror(z=50.0)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证系统名称出现在输出中
    assert name in output_text, (
        f"输出缺少系统名称 '{name}'：\n{output_text}"
    )


# ============================================================================
# 半口径信息测试
# ============================================================================

@settings(max_examples=100)
@given(
    z=z_position_strategy,
    semi_aperture=semi_aperture_strategy,
)
def test_property_5_semi_aperture_displayed(
    z: float,
    semi_aperture: float,
):
    """
    **Feature: matlab-style-api, Property 5: print_info 输出包含必要信息**
    **Validates: Requirements 4.3**
    
    验证 print_info 输出包含半口径信息。
    """
    system = OpticalSystem("Semi Aperture Test")
    system.add_flat_mirror(z=z, semi_aperture=semi_aperture)
    
    # 捕获 print_info 输出
    output = io.StringIO()
    with redirect_stdout(output):
        system.print_info()
    
    output_text = output.getvalue()
    
    # 验证半口径值
    semi_aperture_formatted = f"{semi_aperture:.3f}"
    assert semi_aperture_formatted in output_text, (
        f"输出缺少半口径 {semi_aperture_formatted}：\n{output_text}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
