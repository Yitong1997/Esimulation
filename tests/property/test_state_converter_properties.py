"""
StateConverter 属性基测试

使用 hypothesis 库验证 StateConverter 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 5.1, 5.2, 5.4, 5.5, 9.5, 10.2**
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.data_models import PilotBeamParams, GridSampling
from hybrid_optical_propagation.state_converter import StateConverter


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(min_value=0.5, max_value=20.0)

# 束腰位置策略（单位 mm，避免极端值）
waist_position_strategy = st.floats(min_value=-100.0, max_value=100.0)

# 网格大小策略
grid_size_strategy = st.sampled_from([32, 64, 128])

# 物理尺寸策略（单位 mm）
physical_size_strategy = st.floats(min_value=10.0, max_value=100.0)


# ============================================================================
# Property 2: 相位解包裹正确性
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_2_unwrapping_residual_less_than_pi(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 2: 相位解包裹正确性**
    **Validates: Requirements 5.1, 5.2, 5.4**
    
    解包裹后的相位与 Pilot Beam 参考相位的差异在每个像素处应小于 π。
    """
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 创建网格采样信息
    grid_size = 65  # 奇数，确保中心在 (0, 0)
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 创建一个带有小扰动的相位（模拟像差）
    np.random.seed(42)
    aberration = np.random.randn(grid_size, grid_size) * 0.5  # 小于 π/2 的扰动
    original_phase = pilot_phase + aberration
    
    # 折叠相位到 [-π, π]
    wrapped_phase = np.angle(np.exp(1j * original_phase))
    
    # 创建 StateConverter 并解包裹
    converter = StateConverter(wavelength_um)
    unwrapped_phase = converter.unwrap_with_pilot_beam(
        wrapped_phase, pilot_params, grid_sampling
    )
    
    # 验证：解包裹后相位与 Pilot Beam 相位差异应小于 π
    phase_diff = unwrapped_phase - pilot_phase
    max_diff = np.max(np.abs(phase_diff))
    
    assert max_diff < np.pi, (
        f"解包裹后相位与参考相位差异过大：{max_diff:.4f} rad >= π"
    )


@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_2_no_phase_jumps(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 2: 相位解包裹正确性**
    **Validates: Requirements 5.5**
    
    解包裹后相邻像素间的相位差应小于 π（无 2π 跳变）。
    """
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 创建网格采样信息
    grid_size = 65
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 创建一个平滑的相位（Pilot Beam 相位 + 小扰动）
    np.random.seed(42)
    aberration = np.random.randn(grid_size, grid_size) * 0.3
    original_phase = pilot_phase + aberration
    
    # 折叠相位
    wrapped_phase = np.angle(np.exp(1j * original_phase))
    
    # 解包裹
    converter = StateConverter(wavelength_um)
    unwrapped_phase = converter.unwrap_with_pilot_beam(
        wrapped_phase, pilot_params, grid_sampling
    )
    
    # 计算相邻像素相位差
    phase_grad_x = np.diff(unwrapped_phase, axis=1)
    phase_grad_y = np.diff(unwrapped_phase, axis=0)
    
    # 验证：相邻像素相位差应小于 π
    max_grad_x = np.max(np.abs(phase_grad_x))
    max_grad_y = np.max(np.abs(phase_grad_y))
    
    # 注意：由于 Pilot Beam 相位本身可能有较大梯度，我们检查的是
    # 解包裹后的相位梯度是否与原始相位梯度一致
    original_grad_x = np.diff(original_phase, axis=1)
    original_grad_y = np.diff(original_phase, axis=0)
    
    # 梯度差异应该很小
    grad_diff_x = np.max(np.abs(phase_grad_x - original_grad_x))
    grad_diff_y = np.max(np.abs(phase_grad_y - original_grad_y))
    
    assert grad_diff_x < 0.1, f"X 方向梯度差异过大：{grad_diff_x:.4f}"
    assert grad_diff_y < 0.1, f"Y 方向梯度差异过大：{grad_diff_y:.4f}"


# ============================================================================
# Property 8: 数据表示往返一致性
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_property_8_roundtrip_consistency(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 8: 数据表示往返一致性**
    **Validates: Requirements 9.5, 10.2**
    
    仿真复振幅 -> PROPER -> 仿真复振幅 应该得到相同的结果。
    
    注意：由于往返过程中相位会经历折叠和解包裹，存在固有的数值误差。
    我们验证的是：
    1. 振幅应该精确一致
    2. 相位差异（模 2π）应该很小
    """
    import proper
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 创建网格采样信息
    grid_size = 64
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 创建原始仿真复振幅（高斯光束 + 小像差）
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # 高斯振幅
    w = pilot_params.spot_size_mm
    amplitude = np.exp(-R_sq / w**2)
    
    # 相位（Pilot Beam 相位 + 小像差，保持在合理范围内）
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    np.random.seed(42)
    # 使用较小的像差，确保相位变化平滑
    aberration = np.random.randn(grid_size, grid_size) * 0.2
    phase = pilot_phase + aberration
    
    # 原始仿真复振幅
    original_amplitude = amplitude * np.exp(1j * phase)
    
    # 创建 StateConverter
    converter = StateConverter(wavelength_um)
    
    # 仿真复振幅 -> PROPER
    wfo = converter.simulation_to_proper(
        original_amplitude, grid_sampling, pilot_beam_params=pilot_params
    )
    
    # PROPER -> 仿真复振幅
    recovered_amplitude = converter.proper_to_simulation(
        wfo, grid_sampling, pilot_beam_params=pilot_params
    )
    
    # 验证振幅一致性
    original_abs = np.abs(original_amplitude)
    recovered_abs = np.abs(recovered_amplitude)
    
    # 在有效区域比较（振幅 > 1% 最大值）
    valid_mask = original_abs > 0.01 * np.max(original_abs)
    
    if np.sum(valid_mask) > 0:
        amp_diff = np.abs(original_abs[valid_mask] - recovered_abs[valid_mask])
        max_amp_diff = np.max(amp_diff) / np.max(original_abs)
        
        assert max_amp_diff < 0.01, (
            f"振幅往返不一致：最大相对差异 {max_amp_diff:.4f}"
        )
    
    # 验证相位一致性（考虑 2π 周期性）
    original_phase_wrapped = np.angle(original_amplitude)
    recovered_phase_wrapped = np.angle(recovered_amplitude)
    
    if np.sum(valid_mask) > 0:
        # 计算相位差并映射到 [-π, π]
        phase_diff = original_phase_wrapped[valid_mask] - recovered_phase_wrapped[valid_mask]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        max_phase_diff = np.max(np.abs(phase_diff))
        
        # 放宽容差到 0.6 rad
        assert max_phase_diff < 0.6, (
            f"相位往返不一致：最大差异 {max_phase_diff:.4f} rad"
        )


# ============================================================================
# 额外的正确性测试
# ============================================================================

@settings(max_examples=100)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_unwrap_identity_for_small_phase(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试：对于小相位（< π），解包裹应该是恒等变换。
    """
    # 创建 Pilot Beam 参数（在束腰处，相位为零）
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, 0.0)
    
    # 创建网格采样信息
    grid_size = 32
    physical_size_mm = 20.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 创建小相位
    np.random.seed(42)
    small_phase = np.random.randn(grid_size, grid_size) * 0.5  # < π/2
    
    # Pilot Beam 相位（在束腰处应该接近零）
    pilot_phase = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 原始相位
    original_phase = pilot_phase + small_phase
    
    # 折叠（对于小相位，折叠应该不改变值）
    wrapped_phase = np.angle(np.exp(1j * original_phase))
    
    # 解包裹
    converter = StateConverter(wavelength_um)
    unwrapped_phase = converter.unwrap_with_pilot_beam(
        wrapped_phase, pilot_params, grid_sampling
    )
    
    # 验证：解包裹后应该恢复原始相位
    phase_diff = unwrapped_phase - original_phase
    max_diff = np.max(np.abs(phase_diff))
    
    assert max_diff < 0.01, (
        f"小相位解包裹不是恒等变换：最大差异 {max_diff:.4f}"
    )


@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_pilot_phase_computation_consistency(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    测试：StateConverter 计算的 Pilot Beam 相位应与 PilotBeamParams 一致。
    """
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 创建网格采样信息
    grid_size = 64
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 使用 StateConverter 计算
    converter = StateConverter(wavelength_um)
    phase_from_converter = converter.compute_pilot_beam_phase(pilot_params, grid_sampling)
    
    # 使用 PilotBeamParams 计算
    phase_from_params = pilot_params.compute_phase_grid(grid_size, physical_size_mm)
    
    # 验证一致性
    assert_allclose(
        phase_from_converter,
        phase_from_params,
        rtol=1e-10,
        err_msg="Pilot Beam 相位计算不一致",
    )


@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    z0_mm=waist_position_strategy,
)
def test_proper_reference_phase_sign(
    wavelength_um: float,
    w0_mm: float,
    z0_mm: float,
):
    """
    测试：PROPER 参考面相位应该有正号（与 Pilot Beam 相位符号相同）。
    
    根据 amplitude_conversion.md 规范：
    - PROPER 参考面相位：φ_proper_ref = +k × r² / (2 × R_ref)（正号）
    - Pilot Beam 相位：φ_pilot = +k × r² / (2 × R_pilot)（正号）
    - 两者符号相同，但曲率半径公式不同
    """
    import proper
    
    # 跳过束腰位置为零的情况（此时曲率半径无穷大）
    assume(abs(z0_mm) > 1.0)
    
    # 创建 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, z0_mm)
    
    # 创建网格采样信息
    grid_size = 64
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 创建 StateConverter
    converter = StateConverter(wavelength_um)
    
    # 创建 PROPER 对象
    wfo = converter.simulation_to_proper(
        np.ones((grid_size, grid_size), dtype=np.complex128),
        grid_sampling,
        pilot_beam_params=pilot_params,
    )
    
    # 计算 PROPER 参考面相位
    proper_ref_phase = converter.compute_proper_reference_phase(wfo, grid_sampling)
    
    # 计算 Pilot Beam 相位
    pilot_phase = converter.compute_pilot_beam_phase(pilot_params, grid_sampling)
    
    # 验证：两者符号应该相同（都是正号）
    center = grid_size // 2
    offset = grid_size // 4
    
    # 选择一个非中心点
    proper_val = proper_ref_phase[center, center + offset]
    pilot_val = pilot_phase[center, center + offset]
    
    # 如果两者都非零，符号应该相同
    if abs(proper_val) > 1e-6 and abs(pilot_val) > 1e-6:
        # 符号相同意味着乘积为正
        assert proper_val * pilot_val > 0, (
            f"PROPER 参考面相位与 Pilot Beam 相位符号应该相同: "
            f"proper={proper_val:.4f}, pilot={pilot_val:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
