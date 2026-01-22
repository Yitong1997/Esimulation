"""
曲率半径公式属性基测试

验证 Pilot Beam 和 PROPER 参考面使用正确的曲率半径公式。

**Feature: hybrid-raytracing-validation**
**Validates: Requirements 9.1, 9.2, 10.1, 10.4, 11.5**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

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
w0_strategy = st.floats(min_value=0.5, max_value=10.0)

# 传播距离策略（单位 mm）
z_strategy = st.floats(min_value=1.0, max_value=1000.0)


# ============================================================================
# 属性 1：Pilot Beam 曲率半径公式正确性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    z_mm=z_strategy,
)
def test_property_1_pilot_beam_curvature_strict_formula(
    wavelength_um: float,
    w0_mm: float,
    z_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 1: Pilot Beam 曲率半径公式正确性**
    **Validates: Requirements 9.1, 10.1**
    
    验证 PilotBeamParams.from_gaussian_source() 使用严格公式：
    R = z × (1 + (z_R/z)²)
    """
    # 计算瑞利长度
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 创建 Pilot Beam 参数（z0_mm 是束腰相对于当前位置，负值表示束腰在前）
    # 当 z0_mm = -z_mm 时，当前位置在束腰后方 z_mm 处
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,  # 束腰在当前位置前方 z_mm 处
    )
    
    # 使用严格公式计算期望曲率半径
    expected_R = z_mm * (1 + (z_R / z_mm)**2)
    
    # 验证
    assert_allclose(
        pilot_params.curvature_radius_mm,
        expected_R,
        rtol=1e-10,
        err_msg=f"Pilot Beam 曲率半径不符合严格公式: "
                f"实际={pilot_params.curvature_radius_mm:.6f}, "
                f"期望={expected_R:.6f}",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_property_1_pilot_beam_curvature_at_waist(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 1: Pilot Beam 曲率半径公式正确性**
    **Validates: Requirements 9.1**
    
    在束腰处（z=0），曲率半径应为无穷大。
    """
    # 创建 Pilot Beam 参数，束腰在当前位置
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,  # 束腰在当前位置
    )
    
    # 验证曲率半径为无穷大
    assert np.isinf(pilot_params.curvature_radius_mm), (
        f"束腰处曲率半径应为无穷大，实际为 {pilot_params.curvature_radius_mm}"
    )


# ============================================================================
# 属性 2：from_q_parameter 正确提取曲率半径
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    z_mm=z_strategy,
)
def test_property_2_from_q_parameter_curvature(
    wavelength_um: float,
    w0_mm: float,
    z_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 2: from_q_parameter 正确提取曲率半径**
    **Validates: Requirements 9.1, 10.1**
    
    验证 from_q_parameter 从 1/Re(1/q) 正确提取曲率半径。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 构造 q 参数
    q = z_mm + 1j * z_R
    
    # 从 q 参数创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_q_parameter(
        q=q,
        wavelength_um=wavelength_um,
    )
    
    # 从 1/q 提取期望曲率半径
    inv_q = 1.0 / q
    expected_R = 1.0 / np.real(inv_q)
    
    # 验证
    assert_allclose(
        pilot_params.curvature_radius_mm,
        expected_R,
        rtol=1e-10,
        err_msg=f"from_q_parameter 曲率半径提取错误: "
                f"实际={pilot_params.curvature_radius_mm:.6f}, "
                f"期望={expected_R:.6f}",
    )


# ============================================================================
# 属性 6：近场与远场公式一致性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_property_6_far_field_consistency(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 6: 近场与远场公式一致性**
    **Validates: Requirements 11.5**
    
    当 z >> z_R（远场）时，严格公式和远场近似应趋于一致。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 远场条件：z = 10 × z_R
    z_mm = 10 * z_R
    
    # 严格公式
    R_strict = z_mm * (1 + (z_R / z_mm)**2)
    
    # 远场近似
    R_approx = z_mm
    
    # 计算相对误差
    relative_error = abs(R_strict - R_approx) / R_strict
    
    # 在远场，相对误差应小于 1%
    assert relative_error < 0.01, (
        f"远场时严格公式和近似公式差异过大: "
        f"严格={R_strict:.6f}, 近似={R_approx:.6f}, "
        f"相对误差={relative_error*100:.2f}%"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_property_6_near_field_difference(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 6: 近场与远场公式一致性**
    **Validates: Requirements 11.5**
    
    当 z ≈ z_R（近场）时，严格公式和远场近似应有显著差异。
    这验证了为什么 Pilot Beam 必须使用严格公式。
    """
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 近场条件：z = z_R
    z_mm = z_R
    
    # 严格公式
    R_strict = z_mm * (1 + (z_R / z_mm)**2)  # = 2 × z_R
    
    # 远场近似
    R_approx = z_mm  # = z_R
    
    # 计算相对误差
    relative_error = abs(R_strict - R_approx) / R_strict
    
    # 在近场，相对误差应显著（约 50%）
    assert relative_error > 0.4, (
        f"近场时严格公式和近似公式差异应该显著: "
        f"严格={R_strict:.6f}, 近似={R_approx:.6f}, "
        f"相对误差={relative_error*100:.2f}%"
    )


# ============================================================================
# 属性 2：PROPER 参考面相位公式正确性
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=st.floats(min_value=0.1, max_value=1.0),  # 小束腰确保远场条件
    z_factor=st.floats(min_value=3.0, max_value=20.0),  # z = z_factor × z_R
)
def test_property_2_proper_reference_phase_formula(
    wavelength_um: float,
    w0_mm: float,
    z_factor: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 2: PROPER 参考面相位公式正确性**
    **Validates: Requirements 9.2, 10.4**
    
    验证 StateConverter.compute_proper_reference_phase() 使用正确的公式。
    """
    import proper
    
    wavelength_mm = wavelength_um * 1e-3
    z_R = np.pi * w0_mm**2 / wavelength_mm
    
    # 确保在远场（PROPER 使用球面参考）
    z_mm = z_factor * z_R  # 远场条件
    
    # 创建 PROPER 波前对象
    beam_diameter_m = 0.05
    wavelength_m = wavelength_um * 1e-6
    grid_size = 64
    
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
    
    # 设置高斯光束参数
    wfo.w0 = w0_mm * 1e-3
    wfo.z_Rayleigh = z_R * 1e-3
    wfo.z = z_mm * 1e-3
    wfo.z_w0 = 0.0  # 束腰在原点
    wfo.reference_surface = "SPHERI"
    
    # 创建网格采样信息
    sampling_m = proper.prop_get_sampling(wfo)
    physical_size_mm = sampling_m * 1e3 * grid_size
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 计算 PROPER 参考面相位
    converter = StateConverter(wavelength_um)
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    # 手动计算期望值
    R_ref_m = wfo.z - wfo.z_w0  # 远场近似
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_m = (X_mm * 1e-3)**2 + (Y_mm * 1e-3)**2
    k = 2 * np.pi / wfo.lamda
    expected_phase = -k * r_sq_m / (2 * R_ref_m)
    
    # 验证
    assert_allclose(
        proper_ref_phase,
        expected_phase,
        rtol=1e-10,
        err_msg="PROPER 参考面相位公式不正确",
    )


@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
)
def test_property_2_proper_planar_reference(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 2: PROPER 参考面相位公式正确性**
    **Validates: Requirements 9.2**
    
    验证平面参考面时返回零相位。
    """
    import proper
    
    # 创建 PROPER 波前对象
    beam_diameter_m = 0.05
    wavelength_m = wavelength_um * 1e-6
    grid_size = 64
    
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
    
    # 设置为平面参考
    wfo.reference_surface = "PLANAR"
    
    # 创建网格采样信息
    sampling_m = proper.prop_get_sampling(wfo)
    physical_size_mm = sampling_m * 1e3 * grid_size
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 计算 PROPER 参考面相位
    converter = StateConverter(wavelength_um)
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    # 验证为零
    assert_allclose(
        proper_ref_phase,
        np.zeros((grid_size, grid_size)),
        atol=1e-15,
        err_msg="平面参考面相位应为零",
    )


# ============================================================================
# compute_phase_grid 使用 curvature_radius_mm 验证
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=w0_strategy,
    z_mm=z_strategy,
)
def test_compute_phase_grid_uses_curvature_radius_mm(
    wavelength_um: float,
    w0_mm: float,
    z_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 10.1**
    
    验证 compute_phase_grid 使用 self.curvature_radius_mm（严格公式）。
    """
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=-z_mm,
    )
    
    grid_size = 64
    physical_size_mm = 30.0
    
    # 计算相位网格
    phase_grid = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 手动计算期望值（使用 curvature_radius_mm）
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    expected_phase = k * r_sq / (2 * pilot_params.curvature_radius_mm)
    
    # 验证
    assert_allclose(
        phase_grid,
        expected_phase,
        rtol=1e-10,
        err_msg="compute_phase_grid 未使用 curvature_radius_mm",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
