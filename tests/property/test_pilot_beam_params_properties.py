"""
PilotBeamParams 属性基测试

使用 hypothesis 库验证 PilotBeamParams 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 8.1, 8.3, 8.5, 8.6**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.data_models import PilotBeamParams


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(min_value=0.1, max_value=50.0)

# 束腰位置策略（单位 mm）
waist_position_strategy = st.floats(min_value=-500.0, max_value=500.0)

# 传播距离策略（单位 mm）
propagation_distance_strategy = st.floats(min_value=-1000.0, max_value=1000.0)

# 焦距策略（单位 mm，排除接近零的值）
focal_length_strategy = st.one_of(
    st.floats(min_value=10.0, max_value=10000.0),
    st.floats(min_value=-10000.0, max_value=-10.0),
)


# ============================================================================
# Property 7: Pilot Beam ABCD 追踪正确性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    distance_mm=propagation_distance_strategy,
)
def test_property_7_free_space_propagation(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    distance_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    自由空间传播：q_out = q_in + d
    """
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 传播
    params_after = params.propagate(distance_mm)
    
    # 验证 q 参数变换
    expected_q = params.q_parameter + distance_mm
    assert_allclose(
        params_after.q_parameter,
        expected_q,
        rtol=1e-10,
        err_msg="自由空间传播 q 参数变换不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    focal_length_mm=focal_length_strategy,
)
def test_property_7_thin_lens_transformation(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    focal_length_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    薄透镜变换：1/q_out = 1/q_in - 1/f
    """
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 应用薄透镜
    params_after = params.apply_lens(focal_length_mm)
    
    # 验证 q 参数变换
    # 1/q_out = 1/q_in - 1/f
    inv_q_in = 1.0 / params.q_parameter
    inv_q_out_expected = inv_q_in - 1.0 / focal_length_mm
    q_out_expected = 1.0 / inv_q_out_expected
    
    assert_allclose(
        params_after.q_parameter,
        q_out_expected,
        rtol=1e-10,
        err_msg="薄透镜 q 参数变换不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    radius_mm=focal_length_strategy,  # 复用焦距策略
)
def test_property_7_spherical_mirror_transformation(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    radius_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    球面镜变换：1/q_out = 1/q_in - 2/R
    """
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 应用球面镜
    params_after = params.apply_mirror(radius_mm)
    
    # 验证 q 参数变换
    # 1/q_out = 1/q_in - 2/R
    inv_q_in = 1.0 / params.q_parameter
    inv_q_out_expected = inv_q_in - 2.0 / radius_mm
    q_out_expected = 1.0 / inv_q_out_expected
    
    assert_allclose(
        params_after.q_parameter,
        q_out_expected,
        rtol=1e-10,
        err_msg="球面镜 q 参数变换不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_7_curvature_radius_from_q(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    曲率半径计算：R = 1 / Re(1/q)
    """
    # 跳过束腰位置（曲率半径无穷大）
    assume(abs(z0_mm) > 1.0)
    
    # 创建 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 验证曲率半径
    inv_q = 1.0 / params.q_parameter
    expected_R = 1.0 / np.real(inv_q) if np.real(inv_q) != 0 else np.inf
    
    if not np.isinf(expected_R):
        assert_allclose(
            params.curvature_radius_mm,
            expected_R,
            rtol=1e-6,
            err_msg="曲率半径计算不正确",
        )


# ============================================================================
# Property 12: Pilot Beam 相位在主光线处为零
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_12_phase_at_center_is_zero(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 12: Pilot Beam 相位在主光线处为零**
    **Validates: Requirements 8.5**
    
    Pilot Beam 相位定义为相对于主光线的相位延迟，主光线处相位为 0。
    验证方法：使用奇数网格大小，确保中心点在 (0, 0)。
    """
    # 创建 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 使用奇数网格大小，确保中心点在 (0, 0)
    grid_size = 65  # 奇数
    physical_size_mm = 20.0
    phase_grid = params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 中心点相位应为零
    center = grid_size // 2
    center_phase = phase_grid[center, center]
    
    # 验证中心点坐标确实是 (0, 0)
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    assert abs(coords[center]) < 1e-10, f"中心坐标不为零：{coords[center]}"
    
    assert abs(center_phase) < 1e-10, (
        f"主光线处相位不为零：{center_phase}"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_12_phase_at_radius_zero(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 12: Pilot Beam 相位在主光线处为零**
    **Validates: Requirements 8.5, 8.6**
    
    compute_phase_at_radius(0) 应返回 0。
    """
    # 创建 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 计算 r=0 处的相位
    phase_at_zero = params.compute_phase_at_radius(0.0)
    
    assert abs(phase_at_zero) < 1e-10, (
        f"r=0 处相位不为零：{phase_at_zero}"
    )


# ============================================================================
# 额外的正确性测试
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_q_parameter_roundtrip(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    测试 q 参数往返一致性：
    from_gaussian_source -> from_q_parameter 应该得到相同的参数
    """
    # 创建初始 Pilot Beam
    params1 = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 从 q 参数重建
    params2 = PilotBeamParams.from_q_parameter(params1.q_parameter, wavelength_um)
    
    # 验证参数一致性
    assert_allclose(
        params2.waist_radius_mm,
        params1.waist_radius_mm,
        rtol=1e-6,
        err_msg="束腰半径往返不一致",
    )
    
    assert_allclose(
        params2.q_parameter,
        params1.q_parameter,
        rtol=1e-10,
        err_msg="q 参数往返不一致",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    d1=propagation_distance_strategy,
    d2=propagation_distance_strategy,
)
def test_propagation_additivity(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    d1: float,
    d2: float,
):
    """
    测试传播的可加性：propagate(d1).propagate(d2) == propagate(d1 + d2)
    """
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 分两步传播
    params_two_steps = params.propagate(d1).propagate(d2)
    
    # 一步传播
    params_one_step = params.propagate(d1 + d2)
    
    # 验证 q 参数一致
    assert_allclose(
        params_two_steps.q_parameter,
        params_one_step.q_parameter,
        rtol=1e-10,
        err_msg="传播可加性不满足",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_rayleigh_length_formula(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试瑞利长度公式：z_R = π * w0² / λ
    """
    # 创建 Pilot Beam（在束腰处）
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, 0.0)
    
    # 计算预期瑞利长度
    wavelength_mm = wavelength_um * 1e-3
    expected_z_R = np.pi * w0_mm**2 / wavelength_mm
    
    assert_allclose(
        params.rayleigh_length_mm,
        expected_z_R,
        rtol=1e-10,
        err_msg="瑞利长度计算不正确",
    )


# ============================================================================
# 折射面变换测试
# ============================================================================

# 折射率策略（典型光学材料范围）
refractive_index_strategy = st.floats(min_value=1.0, max_value=2.5)


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    radius_mm=focal_length_strategy,  # 复用焦距策略
    n1=refractive_index_strategy,
    n2=refractive_index_strategy,
)
def test_property_7_refraction_transformation(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    radius_mm: float,
    n1: float,
    n2: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    折射面变换：
    ABCD 矩阵: |     1           0      |
              | (n1-n2)/(n2*R)  n1/n2  |
    
    q 参数变换: q_out = (A*q_in + B) / (C*q_in + D)
    """
    # 避免 n1 == n2（无折射效果）
    assume(abs(n1 - n2) > 0.01)
    
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 应用折射面
    params_after = params.apply_refraction(radius_mm, n1, n2)
    
    # 验证 q 参数变换
    # ABCD 矩阵元素
    A = 1
    B = 0
    C = (n1 - n2) / (n2 * radius_mm)
    D = n1 / n2
    
    q_out_expected = (A * params.q_parameter + B) / (C * params.q_parameter + D)
    
    assert_allclose(
        params_after.q_parameter,
        q_out_expected,
        rtol=1e-10,
        err_msg="折射面 q 参数变换不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    n1=refractive_index_strategy,
    n2=refractive_index_strategy,
)
def test_property_7_flat_refraction_transformation(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    n1: float,
    n2: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 7: Pilot Beam ABCD 追踪正确性**
    **Validates: Requirements 8.1, 8.3**
    
    平面折射面变换（R = ∞）：
    ABCD 矩阵: | 1  0     |
              | 0  n1/n2 |
    
    q 参数变换: q_out = q_in * n2 / n1
    """
    # 避免 n1 == n2（无折射效果）
    assume(abs(n1 - n2) > 0.01)
    
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 应用平面折射面
    params_after = params.apply_refraction(np.inf, n1, n2)
    
    # 验证 q 参数变换
    # 对于平面折射面: q_out = q_in / (n1/n2) = q_in * n2 / n1
    q_out_expected = params.q_parameter * n2 / n1
    
    assert_allclose(
        params_after.q_parameter,
        q_out_expected,
        rtol=1e-10,
        err_msg="平面折射面 q 参数变换不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
    radius_mm=focal_length_strategy,
    n=refractive_index_strategy,
)
def test_refraction_same_index_no_change(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
    radius_mm: float,
    n: float,
):
    """
    测试相同折射率时无变化：apply_refraction(R, n, n) 应该返回相同的 q 参数
    """
    # 创建初始 Pilot Beam
    params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 应用相同折射率的折射面
    params_after = params.apply_refraction(radius_mm, n, n)
    
    # q 参数应该不变
    assert_allclose(
        params_after.q_parameter,
        params.q_parameter,
        rtol=1e-10,
        err_msg="相同折射率时 q 参数应该不变",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
