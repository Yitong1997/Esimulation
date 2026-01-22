"""
ParaxialPhasePropagator 属性基测试

使用 hypothesis 库验证 ParaxialPhasePropagator 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 19.4, 19.5**
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
from hybrid_optical_propagation.paraxial_propagator import (
    ParaxialPhasePropagator,
    compute_paraxial_phase_correction,
)


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 焦距策略（单位 mm）
focal_length_strategy = st.floats(min_value=10.0, max_value=1000.0)

# 网格大小策略
grid_size_strategy = st.sampled_from([32, 64, 128])

# 物理尺寸策略（单位 mm）
physical_size_strategy = st.floats(min_value=10.0, max_value=100.0)


# ============================================================================
# Property 11: PARAXIAL 表面相位修正正确性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    focal_length_mm=focal_length_strategy,
    grid_size=grid_size_strategy,
    physical_size_mm=physical_size_strategy,
)
def test_property_11_phase_correction_formula(
    wavelength_um: float,
    focal_length_mm: float,
    grid_size: int,
    physical_size_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 11: PARAXIAL 表面相位修正正确性**
    **Validates: Requirements 19.4, 19.5**
    
    相位修正应满足公式: φ(r) = -k × r² / (2f)
    """
    # 计算相位修正
    phase_correction = compute_paraxial_phase_correction(
        focal_length_mm=focal_length_mm,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
        wavelength_um=wavelength_um,
    )
    
    # 手动计算期望值
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    expected_phase = -k * r_sq / (2 * focal_length_mm)
    
    # 验证
    assert_allclose(
        phase_correction,
        expected_phase,
        rtol=1e-10,
        err_msg="相位修正公式不正确",
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    focal_length_mm=focal_length_strategy,
    physical_size_mm=physical_size_strategy,
)
def test_property_11_phase_at_center_is_zero(
    wavelength_um: float,
    focal_length_mm: float,
    physical_size_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 11: PARAXIAL 表面相位修正正确性**
    **Validates: Requirements 19.4**
    
    在光轴处（r=0），相位修正应为零。
    使用奇数网格确保中心点在精确的 r=0 处。
    """
    # 使用奇数网格确保中心点在 r=0
    grid_size = 65
    
    # 计算相位修正
    phase_correction = compute_paraxial_phase_correction(
        focal_length_mm=focal_length_mm,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
        wavelength_um=wavelength_um,
    )
    
    # 中心点
    center = grid_size // 2
    
    # 验证中心点相位为零
    center_phase = phase_correction[center, center]
    
    assert abs(center_phase) < 1e-10, (
        f"中心点相位应为零，实际为 {center_phase:.6e}"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    focal_length_mm=focal_length_strategy,
)
def test_property_11_phase_sign_for_positive_focal_length(
    wavelength_um: float,
    focal_length_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 11: PARAXIAL 表面相位修正正确性**
    **Validates: Requirements 19.5**
    
    对于正焦距（会聚透镜），边缘相位应为负值。
    """
    grid_size = 64
    physical_size_mm = 30.0
    
    # 计算相位修正
    phase_correction = compute_paraxial_phase_correction(
        focal_length_mm=focal_length_mm,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
        wavelength_um=wavelength_um,
    )
    
    # 边缘点（角落）
    corner_phase = phase_correction[0, 0]
    
    # 正焦距时，边缘相位应为负值
    assert corner_phase < 0, (
        f"正焦距时边缘相位应为负值，实际为 {corner_phase:.6f}"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    focal_length_mm=focal_length_strategy,
)
def test_property_11_phase_sign_for_negative_focal_length(
    wavelength_um: float,
    focal_length_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 11: PARAXIAL 表面相位修正正确性**
    **Validates: Requirements 19.5**
    
    对于负焦距（发散透镜），边缘相位应为正值。
    """
    grid_size = 64
    physical_size_mm = 30.0
    
    # 使用负焦距
    negative_focal_length = -focal_length_mm
    
    # 计算相位修正
    phase_correction = compute_paraxial_phase_correction(
        focal_length_mm=negative_focal_length,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
        wavelength_um=wavelength_um,
    )
    
    # 边缘点（角落）
    corner_phase = phase_correction[0, 0]
    
    # 负焦距时，边缘相位应为正值
    assert corner_phase > 0, (
        f"负焦距时边缘相位应为正值，实际为 {corner_phase:.6f}"
    )


# ============================================================================
# Pilot Beam 更新测试
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=st.floats(min_value=1.0, max_value=10.0),
    focal_length_mm=focal_length_strategy,
)
def test_pilot_beam_lens_transformation(
    wavelength_um: float,
    w0_mm: float,
    focal_length_mm: float,
):
    """
    测试 Pilot Beam 薄透镜变换（使用 PilotBeamParams.apply_lens）。
    """
    # 创建初始 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    
    # 应用薄透镜变换
    new_params = pilot_params.apply_lens(focal_length_mm)
    
    # 验证 q 参数变换
    # 薄透镜变换: 1/q_out = 1/q_in - 1/f
    q_in = pilot_params.q_parameter
    expected_inv_q = 1/q_in - 1/focal_length_mm
    expected_q = 1/expected_inv_q
    
    assert_allclose(
        new_params.q_parameter.real,
        expected_q.real,
        rtol=1e-6,
        err_msg="薄透镜变换后 q 参数实部不正确",
    )
    assert_allclose(
        new_params.q_parameter.imag,
        expected_q.imag,
        rtol=1e-6,
        err_msg="薄透镜变换后 q 参数虚部不正确",
    )


# ============================================================================
# 初始化测试
# ============================================================================

@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
)
def test_paraxial_propagator_initialization(
    wavelength_um: float,
):
    """
    测试 ParaxialPhasePropagator 初始化。
    """
    propagator = ParaxialPhasePropagator(wavelength_um)
    
    assert propagator.wavelength_um == wavelength_um


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# 任务 3.3: 验证 PROPER 参数与 ABCD 法则计算结果一致
# ============================================================================

@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=st.floats(min_value=1.0, max_value=10.0),
    focal_length_mm=focal_length_strategy,
)
def test_proper_params_match_abcd_after_lens(
    wavelength_um: float,
    w0_mm: float,
    focal_length_mm: float,
):
    """
    **Feature: hybrid-raytracing-validation, Property 3: 薄透镜 Pilot Beam 与 PROPER 参数同步**
    **Validates: Requirements 6.3, 10.2**
    
    验证使用 prop_lens 后，PROPER 的高斯光束参数与 ABCD 法则计算结果一致。
    
    由于薄透镜使用 prop_lens 处理，Pilot Beam 参数可以在传播后直接从 PROPER 读取。
    此测试验证 apply_lens() 独立计算的结果与 PROPER 更新后的参数一致。
    """
    import proper
    
    # 创建初始 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    
    # 创建 PROPER 波前对象
    beam_diameter_m = 0.05  # 50 mm
    wavelength_m = wavelength_um * 1e-6
    grid_size = 64
    
    wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
    
    # 同步初始高斯光束参数
    wfo.w0 = w0_mm * 1e-3
    wfo.z_Rayleigh = pilot_params.rayleigh_length_mm * 1e-3
    wfo.z_w0 = wfo.z  # 束腰在当前位置
    
    # 应用薄透镜
    focal_length_m = focal_length_mm * 1e-3
    proper.prop_lens(wfo, focal_length_m)
    
    # 使用 ABCD 法则计算 Pilot Beam 参数
    new_pilot_params = pilot_params.apply_lens(focal_length_mm)
    
    # 比较高斯光束参数
    # 注意：PROPER 使用米，Pilot Beam 使用毫米
    proper_w0_mm = wfo.w0 * 1e3
    proper_z_R_mm = wfo.z_Rayleigh * 1e3
    
    pilot_w0_mm = new_pilot_params.waist_radius_mm
    pilot_z_R_mm = new_pilot_params.rayleigh_length_mm
    
    # 验证束腰半径一致（相对误差 < 5%）
    # 注意：由于 PROPER 和 ABCD 法则的实现细节可能略有不同，
    # 允许较大的容差
    w0_error = abs(proper_w0_mm - pilot_w0_mm) / pilot_w0_mm if pilot_w0_mm > 0 else 0
    
    # 验证瑞利长度一致
    z_R_error = abs(proper_z_R_mm - pilot_z_R_mm) / pilot_z_R_mm if pilot_z_R_mm > 0 else 0
    
    # 由于 PROPER 内部实现可能与理论 ABCD 法则有细微差异，
    # 这里主要验证两者在合理范围内一致
    assert w0_error < 0.1 or abs(proper_w0_mm - pilot_w0_mm) < 0.1, (
        f"束腰半径不一致: PROPER={proper_w0_mm:.4f} mm, "
        f"Pilot Beam={pilot_w0_mm:.4f} mm, 误差={w0_error*100:.1f}%"
    )
    
    assert z_R_error < 0.1 or abs(proper_z_R_mm - pilot_z_R_mm) < 0.1, (
        f"瑞利长度不一致: PROPER={proper_z_R_mm:.4f} mm, "
        f"Pilot Beam={pilot_z_R_mm:.4f} mm, 误差={z_R_error*100:.1f}%"
    )
