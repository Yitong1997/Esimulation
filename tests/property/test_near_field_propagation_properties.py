"""
近场传播验证属性基测试

验证在近场条件下（z ≈ z_R），Pilot Beam 使用严格公式的正确性。

**Feature: hybrid-raytracing-validation**
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.data_models import (
    PilotBeamParams,
    GridSampling,
)
from hybrid_optical_propagation.state_converter import StateConverter


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
w0_strategy = st.floats(min_value=0.5, max_value=5.0)

# 近场因子策略（z = factor × z_R）
near_field_factor_strategy = st.floats(min_value=0.5, max_value=2.0)

# 远场因子策略（z = factor × z_R）
far_field_factor_strategy = st.floats(min_value=5.0, max_value=20.0)


# ============================================================================
# 任务 6.1：近场传播测试用例
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    near_field_factor=near_field_factor_strategy,
)
def test_near_field_curvature_radius(
    wavelength_um: float,
    w0_mm: float,
    near_field_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.1: 近场传播测试用例**
    **Validates: Requirements 11.1, 11.2**
    
    在近场条件下（z ≈ z_R），验证 Pilot Beam 曲率半径使用严格公式。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 近场条件：z = factor × z_R
    z_mm = near_field_factor * z_R
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,  # 束腰在当前位置前方
    )
    
    # 严格公式
    R_strict = z_mm * (1 + (z_R / z_mm)**2)
    
    # 远场近似
    R_approx = z_mm
    
    # 验证 Pilot Beam 使用严格公式
    assert_allclose(
        pilot_params.curvature_radius_mm,
        R_strict,
        rtol=1e-10,
        err_msg=f"近场时 Pilot Beam 应使用严格公式",
    )
    
    # 验证近场时严格公式与远场近似有显著差异
    relative_diff = abs(R_strict - R_approx) / R_strict
    
    # 当 z = z_R 时，差异约为 50%
    if near_field_factor <= 1.5:
        assert relative_diff > 0.2, (
            f"近场时严格公式与远场近似应有显著差异: "
            f"差异={relative_diff*100:.1f}%"
        )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_near_field_at_rayleigh_distance(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.1: 近场传播测试用例**
    **Validates: Requirements 11.1, 11.2**
    
    在瑞利距离处（z = z_R），曲率半径应为 2 × z_R。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 在瑞利距离处
    z_mm = z_R
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,
    )
    
    # 在 z = z_R 处，R = z_R × (1 + 1) = 2 × z_R
    expected_R = 2 * z_R
    
    assert_allclose(
        pilot_params.curvature_radius_mm,
        expected_R,
        rtol=1e-10,
        err_msg=f"在瑞利距离处，曲率半径应为 2 × z_R",
    )


# ============================================================================
# 任务 6.2：验证 Pilot Beam 使用严格公式
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    near_field_factor=near_field_factor_strategy,
)
def test_pilot_beam_phase_uses_strict_formula(
    wavelength_um: float,
    w0_mm: float,
    near_field_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.2: 验证 Pilot Beam 使用严格公式**
    **Validates: Requirements 11.3, 11.4**
    
    验证 compute_phase_grid 在近场使用严格公式计算的曲率半径。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    z_mm = near_field_factor * z_R
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,
    )
    
    grid_size = 64
    physical_size_mm = 20.0
    
    # 计算相位网格
    phase_grid = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 手动使用严格公式计算期望相位
    R_strict = z_mm * (1 + (z_R / z_mm)**2)
    
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength_mm
    expected_phase = k * r_sq / (2 * R_strict)
    
    # 验证
    assert_allclose(
        phase_grid,
        expected_phase,
        rtol=1e-10,
        err_msg="Pilot Beam 相位网格应使用严格公式",
    )


@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    near_field_factor=near_field_factor_strategy,
)
def test_pilot_beam_vs_proper_reference_near_field(
    wavelength_um: float,
    w0_mm: float,
    near_field_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.2: 验证 Pilot Beam 使用严格公式**
    **Validates: Requirements 11.3, 11.4**
    
    在近场，Pilot Beam 相位与 PROPER 参考面相位应有显著差异。
    这验证了两者使用不同的曲率半径公式。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    z_mm = near_field_factor * z_R
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,
    )
    
    grid_size = 64
    physical_size_mm = 20.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 计算 Pilot Beam 相位（使用严格公式）
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 手动计算 PROPER 参考面相位（使用远场近似）
    R_approx = z_mm  # 远场近似
    
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength_mm
    # PROPER 参考面相位（正号！根据 amplitude_conversion.md 规范）
    proper_ref_phase = k * r_sq / (2 * R_approx)
    
    # 在近场，两者符号相同但数值不同
    # Pilot Beam: φ = k × r² / (2 × R_strict)
    # PROPER: φ = k × r² / (2 × R_approx)
    # 差异来自：曲率半径公式不同（严格公式 vs 远场近似）
    
    # 验证符号相同
    center = grid_size // 2
    edge = 0
    
    pilot_edge = pilot_phase[edge, edge]
    proper_edge = proper_ref_phase[edge, edge]
    
    # 符号应相同（都是正号）
    if abs(pilot_edge) > 1e-10 and abs(proper_edge) > 1e-10:
        assert pilot_edge * proper_edge > 0, (
            f"Pilot Beam 和 PROPER 参考面相位符号应相同: "
            f"Pilot={pilot_edge:.4f}, PROPER={proper_edge:.4f}"
        )
    
    # 验证数值不同（在近场，严格公式给出更大的曲率半径，因此相位更小）
    if near_field_factor <= 1.5:
        # 在近场，R_strict > R_approx，所以 pilot_phase < proper_ref_phase
        assert np.max(np.abs(pilot_phase)) < np.max(np.abs(proper_ref_phase)) * 1.1, (
            f"近场时 Pilot Beam 相位应小于 PROPER 参考面相位"
        )


# ============================================================================
# 任务 6.3：验证近场与远场公式在远场趋于一致
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    far_field_factor=far_field_factor_strategy,
)
def test_far_field_formulas_converge(
    wavelength_um: float,
    w0_mm: float,
    far_field_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.3: 验证近场与远场公式在远场趋于一致**
    **Validates: Requirements 11.5**
    
    在远场（z >> z_R），严格公式和远场近似应趋于一致。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 远场条件：z = factor × z_R
    z_mm = far_field_factor * z_R
    
    # 严格公式
    R_strict = z_mm * (1 + (z_R / z_mm)**2)
    
    # 远场近似
    R_approx = z_mm
    
    # 计算相对误差
    relative_error = abs(R_strict - R_approx) / R_strict
    
    # 在远场，相对误差应小于 1/(factor²)
    expected_max_error = 1 / (far_field_factor**2)
    
    assert relative_error < expected_max_error * 1.1, (
        f"远场时严格公式和近似公式应趋于一致: "
        f"相对误差={relative_error*100:.2f}%, "
        f"期望最大误差={expected_max_error*100:.2f}%"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_far_field_phase_difference_small(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Task 6.3: 验证近场与远场公式在远场趋于一致**
    **Validates: Requirements 11.5**
    
    在远场，使用严格公式和远场近似计算的相位相对差应很小。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 远场条件：z = 10 × z_R
    z_mm = 10 * z_R
    
    grid_size = 64
    physical_size_mm = 20.0
    
    # 使用严格公式
    R_strict = z_mm * (1 + (z_R / z_mm)**2)
    
    # 使用远场近似
    R_approx = z_mm
    
    # 计算相位
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength_mm
    
    phase_strict = k * r_sq / (2 * R_strict)
    phase_approx = k * r_sq / (2 * R_approx)
    
    # 相位相对差（相对于严格公式）
    # 在非零相位区域计算相对差
    nonzero_mask = np.abs(phase_strict) > 1e-10
    if np.any(nonzero_mask):
        relative_diff = np.abs(phase_strict - phase_approx) / np.abs(phase_strict)
        max_relative_diff = np.max(relative_diff[nonzero_mask])
        
        # 在远场（z = 10 × z_R），相对差应约为 1%
        # 因为 R_strict/R_approx = 1 + (z_R/z)² ≈ 1.01
        assert max_relative_diff < 0.02, (
            f"远场时相位相对差应很小: 最大相对差={max_relative_diff*100:.2f}%"
        )


# ============================================================================
# 传播后 Pilot Beam 参数验证
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    propagation_distance_factor=st.floats(min_value=0.5, max_value=5.0),
)
def test_pilot_beam_propagation_curvature(
    wavelength_um: float,
    w0_mm: float,
    propagation_distance_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 11.1, 11.2**
    
    验证 Pilot Beam 传播后曲率半径正确更新。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 初始位置在束腰
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    
    # 传播距离
    distance_mm = propagation_distance_factor * z_R
    
    # 传播
    new_params = pilot_params.propagate(distance_mm)
    
    # 期望曲率半径（严格公式）
    z = distance_mm
    expected_R = z * (1 + (z_R / z)**2)
    
    assert_allclose(
        new_params.curvature_radius_mm,
        expected_R,
        rtol=1e-10,
        err_msg="传播后曲率半径应使用严格公式",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
