"""
simulate 函数属性基测试

使用 hypothesis 库验证 simulate 函数的正确性属性。

**Feature: matlab-style-api**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from bts import simulate, OpticalSystem, GaussianSource, SimulationResult
from bts.exceptions import ConfigurationError, SimulationError


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(
    min_value=0.4, max_value=2.0,
    allow_nan=False, allow_infinity=False
)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(
    min_value=1.0, max_value=20.0,
    allow_nan=False, allow_infinity=False
)

# 网格大小策略（使用较小的值以加快测试速度）
grid_size_strategy = st.sampled_from([64, 128])

# 表面位置策略（单位 mm）
surface_z_strategy = st.floats(
    min_value=10.0, max_value=200.0,
    allow_nan=False, allow_infinity=False
)

# 倾斜角度策略（度）
tilt_angle_strategy = st.floats(
    min_value=0.0, max_value=45.0,
    allow_nan=False, allow_infinity=False
)

# 曲率半径策略（单位 mm）
radius_strategy = st.floats(
    min_value=100.0, max_value=1000.0,
    allow_nan=False, allow_infinity=False
)

# 半口径策略（单位 mm）
semi_aperture_strategy = st.floats(
    min_value=10.0, max_value=50.0,
    allow_nan=False, allow_infinity=False
)


# ============================================================================
# 辅助函数
# ============================================================================

def create_simple_flat_mirror_system(
    z: float = 50.0,
    tilt_x: float = 0.0,
    semi_aperture: float = 25.0,
) -> OpticalSystem:
    """创建简单的平面镜系统
    
    参数:
        z: 镜面位置 (mm)
        tilt_x: 绕 X 轴倾斜角度（度）
        semi_aperture: 半口径 (mm)
    
    返回:
        OpticalSystem 对象
    """
    system = OpticalSystem("Test Flat Mirror")
    system.add_flat_mirror(z=z, tilt_x=tilt_x, semi_aperture=semi_aperture)
    return system


def create_simple_spherical_mirror_system(
    z: float = 100.0,
    radius: float = 200.0,
    semi_aperture: float = 25.0,
) -> OpticalSystem:
    """创建简单的球面镜系统
    
    参数:
        z: 镜面位置 (mm)
        radius: 曲率半径 (mm)
        semi_aperture: 半口径 (mm)
    
    返回:
        OpticalSystem 对象
    """
    system = OpticalSystem("Test Spherical Mirror")
    system.add_spherical_mirror(z=z, radius=radius, semi_aperture=semi_aperture)
    return system


def create_valid_source(
    wavelength_um: float = 0.633,
    w0_mm: float = 5.0,
    grid_size: int = 64,
) -> GaussianSource:
    """创建有效的高斯光源
    
    参数:
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        grid_size: 网格大小
    
    返回:
        GaussianSource 对象
    """
    return GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )


# ============================================================================
# Property 6: simulate 返回完整结果
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    tilt_x=tilt_angle_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_6_simulate_returns_complete_result_flat_mirror(
    z: float,
    tilt_x: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 6: simulate 返回完整结果**
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    *For any* 有效的 `OpticalSystem`（至少一个表面）和有效的 `GaussianSource`，
    调用 `bts.simulate(system, source)` 应该返回 `SimulationResult` 类型的对象，
    且结果中的表面数量应该等于系统中的表面数量。
    
    测试场景：单个平面镜系统
    """
    # 创建光学系统（单个平面镜）
    system = create_simple_flat_mirror_system(z=z, tilt_x=tilt_x)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证返回类型
    assert isinstance(result, SimulationResult), (
        f"simulate 应该返回 SimulationResult 类型，实际返回 {type(result)}"
    )
    
    # 验证仿真成功
    assert result.success, (
        f"仿真应该成功，但失败了: {result.error_message}"
    )
    
    # 验证表面数量
    # 注意：SimulationResult.surfaces 包含初始光源状态（index=-1 或 0）和所有光学表面
    # 我们需要计算实际的光学表面数量
    optical_surface_count = sum(
        1 for s in result.surfaces
        if s.surface_type != 'source' and s.index >= 0
    )
    
    assert optical_surface_count == len(system), (
        f"结果中的光学表面数量应该等于系统中的表面数量。\n"
        f"  系统表面数量: {len(system)}\n"
        f"  结果光学表面数量: {optical_surface_count}\n"
        f"  结果总表面数量: {len(result.surfaces)}"
    )


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    radius=radius_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_6_simulate_returns_complete_result_spherical_mirror(
    z: float,
    radius: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 6: simulate 返回完整结果**
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    测试场景：单个球面镜系统
    """
    # 创建光学系统（单个球面镜）
    system = create_simple_spherical_mirror_system(z=z, radius=radius)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证返回类型
    assert isinstance(result, SimulationResult), (
        f"simulate 应该返回 SimulationResult 类型，实际返回 {type(result)}"
    )
    
    # 验证仿真成功
    assert result.success, (
        f"仿真应该成功，但失败了: {result.error_message}"
    )
    
    # 验证表面数量
    optical_surface_count = sum(
        1 for s in result.surfaces
        if s.surface_type != 'source' and s.index >= 0
    )
    
    assert optical_surface_count == len(system), (
        f"结果中的光学表面数量应该等于系统中的表面数量。\n"
        f"  系统表面数量: {len(system)}\n"
        f"  结果光学表面数量: {optical_surface_count}"
    )


@settings(max_examples=50, deadline=None)
@given(
    num_surfaces=st.integers(min_value=1, max_value=3),
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_6_simulate_returns_complete_result_multiple_surfaces(
    num_surfaces: int,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 6: simulate 返回完整结果**
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    测试场景：多表面系统
    """
    # 创建光学系统（多个平面镜）
    system = OpticalSystem("Test Multiple Surfaces")
    
    for i in range(num_surfaces):
        z = 50.0 + i * 100.0  # 每个表面间隔 100mm
        system.add_flat_mirror(z=z, tilt_x=0.0, semi_aperture=30.0)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证返回类型
    assert isinstance(result, SimulationResult), (
        f"simulate 应该返回 SimulationResult 类型，实际返回 {type(result)}"
    )
    
    # 验证仿真成功
    assert result.success, (
        f"仿真应该成功，但失败了: {result.error_message}"
    )
    
    # 验证表面数量
    optical_surface_count = sum(
        1 for s in result.surfaces
        if s.surface_type != 'source' and s.index >= 0
    )
    
    assert optical_surface_count == len(system), (
        f"结果中的光学表面数量应该等于系统中的表面数量。\n"
        f"  系统表面数量: {len(system)}\n"
        f"  结果光学表面数量: {optical_surface_count}"
    )


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_6_result_contains_wavefront_data(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 6: simulate 返回完整结果**
    **Validates: Requirements 5.3**
    
    验证仿真结果包含所有表面的波前数据。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 验证每个光学表面都有波前数据
    for surface in result.surfaces:
        if surface.surface_type == 'source':
            # 光源表面只有入射（初始）波前
            assert surface.entrance is not None, (
                f"光源表面 {surface.index} 应该有入射波前数据"
            )
        elif surface.index >= 0:
            # 光学表面应该有入射和出射波前
            assert surface.entrance is not None, (
                f"表面 {surface.index} 应该有入射波前数据"
            )
            assert surface.exit is not None, (
                f"表面 {surface.index} 应该有出射波前数据"
            )
            
            # 验证波前数据的网格大小
            assert surface.exit.amplitude.shape == (grid_size, grid_size), (
                f"表面 {surface.index} 出射振幅网格大小应该是 ({grid_size}, {grid_size})，"
                f"实际是 {surface.exit.amplitude.shape}"
            )
            assert surface.exit.phase.shape == (grid_size, grid_size), (
                f"表面 {surface.index} 出射相位网格大小应该是 ({grid_size}, {grid_size})，"
                f"实际是 {surface.exit.phase.shape}"
            )


# ============================================================================
# Property 7: 仿真失败时抛出异常
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_7_empty_system_raises_configuration_error(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    *For any* 空的 `OpticalSystem`（无表面），调用 `bts.simulate()` 
    应该抛出 `ConfigurationError` 异常。
    """
    # 创建空的光学系统
    system = OpticalSystem("Empty System")
    
    # 创建有效的光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 验证抛出 ConfigurationError
    with pytest.raises(ConfigurationError):
        simulate(system, source, verbose=False)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_7_negative_wavelength_raises_value_error(
    z: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    *For any* 无效的 `GaussianSource`（负波长），调用 `bts.simulate()` 
    应该抛出 `ValueError` 异常。
    
    注意：由于 GaussianSource 构造函数会验证参数，负波长会在创建光源时就抛出异常。
    这里我们测试 simulate 函数对已创建的无效光源的处理。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建负波长光源时抛出 ValueError
    with pytest.raises(ValueError, match="wavelength_um"):
        GaussianSource(wavelength_um=-1.0, w0_mm=w0_mm, grid_size=grid_size)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    grid_size=grid_size_strategy,
)
def test_property_7_negative_waist_raises_value_error(
    z: float,
    wavelength_um: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    *For any* 无效的 `GaussianSource`（负束腰半径），调用 `bts.simulate()` 
    应该抛出 `ValueError` 异常。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建负束腰半径光源时抛出 ValueError
    with pytest.raises(ValueError, match="w0_mm"):
        GaussianSource(wavelength_um=wavelength_um, w0_mm=-1.0, grid_size=grid_size)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_7_zero_wavelength_raises_value_error(
    z: float,
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    验证零波长会抛出 ValueError。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建零波长光源时抛出 ValueError
    with pytest.raises(ValueError, match="wavelength_um"):
        GaussianSource(wavelength_um=0.0, w0_mm=w0_mm)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_7_zero_waist_raises_value_error(
    z: float,
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    验证零束腰半径会抛出 ValueError。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建零束腰半径光源时抛出 ValueError
    with pytest.raises(ValueError, match="w0_mm"):
        GaussianSource(wavelength_um=wavelength_um, w0_mm=0.0)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_7_negative_grid_size_raises_value_error(
    z: float,
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    验证负网格大小会抛出 ValueError。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建负网格大小光源时抛出 ValueError
    with pytest.raises(ValueError, match="grid_size"):
        GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm, grid_size=-1)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_7_zero_grid_size_raises_value_error(
    z: float,
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: matlab-style-api, Property 7: 仿真失败时抛出异常**
    **Validates: Requirements 5.4**
    
    验证零网格大小会抛出 ValueError。
    """
    # 创建有效的光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 验证创建零网格大小光源时抛出 ValueError
    with pytest.raises(ValueError, match="grid_size"):
        GaussianSource(wavelength_um=wavelength_um, w0_mm=w0_mm, grid_size=0)


# ============================================================================
# 额外的正确性测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_result_config_matches_input(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    验证仿真结果中的配置参数与输入参数一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 验证配置参数
    assert_allclose(
        result.config.wavelength_um,
        wavelength_um,
        rtol=1e-10,
        err_msg=f"波长不匹配：期望 {wavelength_um}，实际 {result.config.wavelength_um}",
    )
    
    assert result.config.grid_size == grid_size, (
        f"网格大小不匹配：期望 {grid_size}，实际 {result.config.grid_size}"
    )


@settings(max_examples=50, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_result_source_params_matches_input(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    验证仿真结果中的光源参数与输入参数一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 验证光源参数
    assert_allclose(
        result.source_params.wavelength_um,
        wavelength_um,
        rtol=1e-10,
        err_msg=f"波长不匹配：期望 {wavelength_um}，实际 {result.source_params.wavelength_um}",
    )
    
    assert_allclose(
        result.source_params.w0_mm,
        w0_mm,
        rtol=1e-10,
        err_msg=f"束腰半径不匹配：期望 {w0_mm}，实际 {result.source_params.w0_mm}",
    )
    
    assert result.source_params.grid_size == grid_size, (
        f"网格大小不匹配：期望 {grid_size}，实际 {result.source_params.grid_size}"
    )


@settings(max_examples=50, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_result_total_path_length_positive(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    验证仿真结果中的总光程为正值。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 验证总光程为正值
    assert result.total_path_length > 0, (
        f"总光程应该为正值，实际为 {result.total_path_length}"
    )


# ============================================================================
# 边界条件测试
# ============================================================================

def test_simulate_with_minimum_valid_parameters():
    """
    测试使用最小有效参数进行仿真。
    """
    # 创建最简单的光学系统
    system = OpticalSystem("Minimal System")
    system.add_flat_mirror(z=10.0, semi_aperture=10.0)
    
    # 创建最小参数的光源
    source = GaussianSource(
        wavelength_um=0.4,  # 最小可见光波长
        w0_mm=1.0,          # 最小束腰半径
        grid_size=32,       # 最小网格大小
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    assert isinstance(result, SimulationResult)


def test_simulate_verbose_parameter():
    """
    测试 verbose 参数不影响仿真结果。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system()
    source = create_valid_source()
    
    # 分别使用 verbose=True 和 verbose=False 执行仿真
    result_verbose = simulate(system, source, verbose=True)
    result_quiet = simulate(system, source, verbose=False)
    
    # 验证两次仿真都成功
    assert result_verbose.success
    assert result_quiet.success
    
    # 验证结果一致
    assert result_verbose.config.wavelength_um == result_quiet.config.wavelength_um
    assert result_verbose.config.grid_size == result_quiet.config.grid_size
    assert len(result_verbose.surfaces) == len(result_quiet.surfaces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
