"""
FreeSpacePropagator 属性基测试

使用 hypothesis 库验证 FreeSpacePropagator 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 4.3, 4.4, 4.5, 4.6, 4.7**
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
    PropagationState,
    SourceDefinition,
)
from hybrid_optical_propagation.free_space_propagator import (
    FreeSpacePropagator,
    compute_propagation_distance,
)
from sequential_system.coordinate_tracking import (
    OpticalAxisState,
    Position3D,
    RayDirection,
)


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(min_value=1.0, max_value=10.0)

# 传播距离策略（单位 mm）
distance_strategy = st.floats(min_value=-500.0, max_value=500.0)

# 位置策略（单位 mm）
position_strategy = st.floats(min_value=-1000.0, max_value=1000.0)

# 方向分量策略
direction_component_strategy = st.floats(min_value=-1.0, max_value=1.0)


# ============================================================================
# Property 3: 传播距离计算正确性
# ============================================================================

@settings(max_examples=100)
@given(
    x1=position_strategy,
    y1=position_strategy,
    z1=position_strategy,
    x2=position_strategy,
    y2=position_strategy,
    z2=position_strategy,
)
def test_property_3_distance_magnitude(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 3: 传播距离计算正确性**
    **Validates: Requirements 4.5**
    
    距离绝对值等于两点间的欧几里得距离。
    """
    current_pos = np.array([x1, y1, z1])
    target_pos = np.array([x2, y2, z2])
    direction = np.array([0.0, 0.0, 1.0])  # 沿 +Z 方向
    
    expected_magnitude = np.linalg.norm(target_pos - current_pos)
    
    # 跳过极小距离（函数有 1e-10 的阈值）
    assume(expected_magnitude > 1e-9 or expected_magnitude < 1e-15)
    
    distance = compute_propagation_distance(current_pos, target_pos, direction)
    
    if expected_magnitude < 1e-10:
        # 极小距离应返回 0
        assert distance == 0.0, f"极小距离应返回 0，实际得到 {distance}"
    else:
        assert_allclose(
            abs(distance),
            expected_magnitude,
            rtol=1e-10,
            err_msg="距离绝对值应等于欧几里得距离",
        )


@settings(max_examples=100)
@given(
    x1=position_strategy,
    y1=position_strategy,
    z1=position_strategy,
    dz=st.floats(min_value=1.0, max_value=500.0),  # 正向位移
)
def test_property_3_positive_direction(
    x1: float, y1: float, z1: float,
    dz: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 3: 传播距离计算正确性**
    **Validates: Requirements 4.6**
    
    当位移向量与光轴方向同向时，距离为正。
    """
    current_pos = np.array([x1, y1, z1])
    target_pos = np.array([x1, y1, z1 + dz])  # 沿 +Z 方向移动
    direction = np.array([0.0, 0.0, 1.0])  # 光轴沿 +Z
    
    distance = compute_propagation_distance(current_pos, target_pos, direction)
    
    assert distance > 0, f"同向位移应产生正距离，实际得到 {distance}"


@settings(max_examples=100)
@given(
    x1=position_strategy,
    y1=position_strategy,
    z1=position_strategy,
    dz=st.floats(min_value=1.0, max_value=500.0),  # 正向位移量
)
def test_property_3_negative_direction(
    x1: float, y1: float, z1: float,
    dz: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 3: 传播距离计算正确性**
    **Validates: Requirements 4.7**
    
    当位移向量与光轴方向反向时，距离为负。
    """
    current_pos = np.array([x1, y1, z1])
    target_pos = np.array([x1, y1, z1 - dz])  # 沿 -Z 方向移动
    direction = np.array([0.0, 0.0, 1.0])  # 光轴沿 +Z
    
    distance = compute_propagation_distance(current_pos, target_pos, direction)
    
    assert distance < 0, f"反向位移应产生负距离，实际得到 {distance}"


@settings(max_examples=100)
@given(
    dx=direction_component_strategy,
    dy=direction_component_strategy,
    dz=direction_component_strategy,
    dist=st.floats(min_value=10.0, max_value=500.0),
)
def test_property_3_arbitrary_direction(
    dx: float, dy: float, dz: float,
    dist: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 3: 传播距离计算正确性**
    **Validates: Requirements 4.5, 4.6, 4.7**
    
    对于任意方向，距离计算应正确。
    """
    # 确保方向向量非零
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    assume(norm > 0.1)
    
    # 归一化方向
    direction = np.array([dx, dy, dz]) / norm
    
    # 当前位置
    current_pos = np.array([0.0, 0.0, 0.0])
    
    # 沿方向移动
    target_pos = current_pos + dist * direction
    
    distance = compute_propagation_distance(current_pos, target_pos, direction)
    
    # 应该得到正距离
    assert_allclose(
        distance,
        dist,
        rtol=1e-10,
        err_msg="沿光轴方向移动应得到正距离",
    )
    
    # 反向移动
    target_pos_neg = current_pos - dist * direction
    distance_neg = compute_propagation_distance(current_pos, target_pos_neg, direction)
    
    # 应该得到负距离
    assert_allclose(
        distance_neg,
        -dist,
        rtol=1e-10,
        err_msg="逆光轴方向移动应得到负距离",
    )


# ============================================================================
# Property 4: 自由空间传播往返一致性
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    distance_mm=st.floats(min_value=10.0, max_value=200.0),
)
def test_property_4_roundtrip_consistency(
    wavelength_um: float,
    w0_mm: float,
    distance_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 4: 自由空间传播往返一致性**
    **Validates: Requirements 4.3, 4.4**
    
    先正向传播 d 再逆向传播 -d 后，波前应恢复到原始状态。
    """
    import proper
    
    # 创建初始波前
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 使用新的振幅/相位分离接口
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    grid_sampling = source.get_grid_sampling()
    
    # 创建初始光轴状态
    initial_axis_state = OpticalAxisState(
        position=Position3D(0.0, 0.0, 0.0),
        direction=RayDirection(0.0, 0.0, 1.0),
        path_length=0.0,
    )
    
    # 创建初始传播状态（使用新的振幅/相位分离接口）
    initial_state = PropagationState(
        surface_index=0,
        position='entrance',
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=pilot_params,
        proper_wfo=wfo,
        optical_axis_state=initial_axis_state,
        grid_sampling=grid_sampling,
    )
    
    # 保存原始振幅和相位
    original_amplitude = amplitude.copy()
    original_phase = phase.copy()
    
    # 创建传播器
    propagator = FreeSpacePropagator(wavelength_um)
    
    # 正向传播
    forward_axis = initial_axis_state.propagate(distance_mm)
    forward_state = propagator.propagate_distance(
        initial_state,
        distance_mm,
        target_surface_index=1,
        target_position='entrance',
        target_axis_state=forward_axis,
    )
    
    # 逆向传播
    backward_axis = forward_axis.propagate(-distance_mm)
    backward_state = propagator.propagate_distance(
        forward_state,
        -distance_mm,
        target_surface_index=0,
        target_position='entrance',
        target_axis_state=backward_axis,
    )
    
    # 验证振幅恢复（使用新的 amplitude 属性）
    recovered_amplitude = backward_state.amplitude
    
    # 在有效区域比较
    valid_mask = original_amplitude > 0.01 * np.max(original_amplitude)
    
    if np.sum(valid_mask) > 0:
        amp_diff = np.abs(original_amplitude[valid_mask] - recovered_amplitude[valid_mask])
        max_amp_diff = np.max(amp_diff) / np.max(original_amplitude)
        
        # 允许 5% 的数值误差（衍射传播有数值误差）
        assert max_amp_diff < 0.05, (
            f"振幅往返不一致：最大相对差异 {max_amp_diff:.4f}"
        )
    
    # 验证 Pilot Beam 参数恢复
    # q 参数应该恢复（往返后 q_out = q_in + d - d = q_in）
    q_original = pilot_params.q_parameter
    q_recovered = backward_state.pilot_beam_params.q_parameter
    
    assert_allclose(
        q_recovered.real,
        q_original.real,
        rtol=1e-6,
        err_msg="Pilot Beam q 参数实部应恢复",
    )
    assert_allclose(
        q_recovered.imag,
        q_original.imag,
        rtol=1e-6,
        err_msg="Pilot Beam q 参数虚部应恢复",
    )


@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_4_zero_distance_identity(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 4: 自由空间传播往返一致性**
    **Validates: Requirements 4.3**
    
    零距离传播应该是恒等变换。
    """
    import proper
    
    # 创建初始波前
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 使用新的振幅/相位分离接口
    amplitude, phase, pilot_params, wfo = source.create_initial_wavefront()
    grid_sampling = source.get_grid_sampling()
    
    # 创建初始光轴状态
    initial_axis_state = OpticalAxisState(
        position=Position3D(0.0, 0.0, 0.0),
        direction=RayDirection(0.0, 0.0, 1.0),
        path_length=0.0,
    )
    
    # 创建初始传播状态（使用新的振幅/相位分离接口）
    initial_state = PropagationState(
        surface_index=0,
        position='entrance',
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=pilot_params,
        proper_wfo=wfo,
        optical_axis_state=initial_axis_state,
        grid_sampling=grid_sampling,
    )
    
    # 创建传播器
    propagator = FreeSpacePropagator(wavelength_um)
    
    # 零距离传播
    result_state = propagator.propagate_distance(
        initial_state,
        0.0,
        target_surface_index=0,
        target_position='entrance',
        target_axis_state=initial_axis_state,
    )
    
    # 验证振幅不变（使用新的 amplitude 属性）
    assert_allclose(
        result_state.amplitude,
        amplitude,
        rtol=1e-10,
        err_msg="零距离传播振幅应不变",
    )
    
    # 验证 Pilot Beam 参数不变
    assert_allclose(
        result_state.pilot_beam_params.q_parameter.real,
        pilot_params.q_parameter.real,
        rtol=1e-10,
        err_msg="零距离传播 q 参数实部应不变",
    )
    assert_allclose(
        result_state.pilot_beam_params.q_parameter.imag,
        pilot_params.q_parameter.imag,
        rtol=1e-10,
        err_msg="零距离传播 q 参数虚部应不变",
    )


# ============================================================================
# 额外的正确性测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    distance_mm=st.floats(min_value=10.0, max_value=100.0),
)
def test_pilot_beam_propagation_consistency(
    wavelength_um: float,
    w0_mm: float,
    distance_mm: float,
):
    """
    测试：Pilot Beam 参数传播应与 ABCD 法则一致。
    """
    # 创建初始 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, 0.0)
    
    # 使用 propagate 方法
    propagated_params = pilot_params.propagate(distance_mm)
    
    # 验证 q 参数变换: q_out = q_in + d
    expected_q = pilot_params.q_parameter + distance_mm
    
    assert_allclose(
        propagated_params.q_parameter.real,
        expected_q.real,
        rtol=1e-10,
        err_msg="q 参数实部应满足 q_out = q_in + d",
    )
    assert_allclose(
        propagated_params.q_parameter.imag,
        expected_q.imag,
        rtol=1e-10,
        err_msg="q 参数虚部应保持不变",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
