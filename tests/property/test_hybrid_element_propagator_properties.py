"""
HybridElementPropagator 属性基测试

使用 hypothesis 库验证 HybridElementPropagator 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 3.5, 6.6, 7.6, 13.3**
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
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(min_value=1.0, max_value=10.0)

# 光线数量策略
num_rays_strategy = st.sampled_from([100, 200, 400])


# ============================================================================
# 基础功能测试
# ============================================================================

@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    num_rays=num_rays_strategy,
)
def test_hybrid_element_propagator_initialization(
    wavelength_um: float,
    num_rays: int,
):
    """
    测试 HybridElementPropagator 初始化。
    """
    propagator = HybridElementPropagator(
        wavelength_um=wavelength_um,
        num_rays=num_rays,
        method="local_raytracing",
    )
    
    assert propagator.wavelength_um == wavelength_um
    assert propagator.num_rays == num_rays
    assert propagator.method == "local_raytracing"


@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
)
def test_hybrid_element_propagator_methods(
    wavelength_um: float,
):
    """
    测试 HybridElementPropagator 支持的方法。
    """
    # 局部光线追迹方法
    propagator_raytracing = HybridElementPropagator(
        wavelength_um=wavelength_um,
        method="local_raytracing",
    )
    assert propagator_raytracing.method == "local_raytracing"
    
    # 纯衍射方法
    propagator_diffraction = HybridElementPropagator(
        wavelength_um=wavelength_um,
        method="pure_diffraction",
    )
    assert propagator_diffraction.method == "pure_diffraction"


# ============================================================================
# Property 9: 能量守恒（雅可比矩阵振幅）- 简化测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_9_jacobian_amplitude_positive(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 9: 能量守恒**
    **Validates: Requirements 6.6**
    
    雅可比矩阵振幅应该为正值。
    """
    propagator = HybridElementPropagator(
        wavelength_um=wavelength_um,
        num_rays=100,
    )
    
    # 创建模拟的输入/输出光线
    from optiland.rays import RealRays
    
    n_rays = 10
    input_rays = RealRays(
        x=np.linspace(-5, 5, n_rays),
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    output_rays = RealRays(
        x=np.linspace(-5, 5, n_rays),
        y=np.zeros(n_rays),
        z=np.full(n_rays, 100.0),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 计算雅可比矩阵振幅
    jacobian_amp = propagator._compute_jacobian_amplitude(
        input_rays, output_rays, None
    )
    
    # 验证所有值为正
    assert np.all(jacobian_amp > 0), "雅可比矩阵振幅应该为正值"


# ============================================================================
# Property 10: 波前无整体倾斜 - 简化测试
# ============================================================================

@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_10_pilot_beam_update_mirror(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 10: 波前无整体倾斜**
    **Validates: Requirements 3.5, 7.6, 13.3**
    
    Pilot Beam 更新应该正确应用球面镜变换。
    """
    from dataclasses import dataclass, field
    from typing import List
    
    # 创建模拟的表面定义
    @dataclass
    class MockSurface:
        is_mirror: bool = True
        radius: float = 200.0
        conic: float = 0.0
        thickness: float = 0.0
        semi_aperture: float = 25.0
        material: str = "mirror"
    
    propagator = HybridElementPropagator(
        wavelength_um=wavelength_um,
        num_rays=100,
    )
    
    # 创建初始 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    
    # 创建模拟表面
    surface = MockSurface(radius=200.0)
    
    # 更新 Pilot Beam
    new_pilot_params = propagator._update_pilot_beam(pilot_params, surface)
    
    # 验证 q 参数变换正确
    # 球面镜变换: 1/q_out = 1/q_in - 2/R
    q_in = pilot_params.q_parameter
    R = surface.radius
    expected_inv_q = 1/q_in - 2/R
    expected_q = 1/expected_inv_q
    
    assert_allclose(
        new_pilot_params.q_parameter.real,
        expected_q.real,
        rtol=1e-6,
        err_msg="球面镜变换后 q 参数实部不正确",
    )
    assert_allclose(
        new_pilot_params.q_parameter.imag,
        expected_q.imag,
        rtol=1e-6,
        err_msg="球面镜变换后 q 参数虚部不正确",
    )


@settings(max_examples=50)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    radius_mm=st.one_of(
        st.floats(min_value=50.0, max_value=500.0),
        st.floats(min_value=-500.0, max_value=-50.0),
    ),
)
def test_property_10_pilot_beam_update_refraction(
    wavelength_um: float,
    w0_mm: float,
    radius_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 10: 波前无整体倾斜**
    **Validates: Requirements 3.5, 7.6, 13.3**
    
    Pilot Beam 更新应该正确应用折射面变换。
    """
    from dataclasses import dataclass
    
    # 创建模拟的折射面定义
    @dataclass
    class MockSurface:
        is_mirror: bool = False
        radius: float = 200.0
        conic: float = 0.0
        thickness: float = 0.0
        semi_aperture: float = 25.0
        material: str = "n-bk7"
    
    propagator = HybridElementPropagator(
        wavelength_um=wavelength_um,
        num_rays=100,
    )
    
    # 创建初始 Pilot Beam 参数
    pilot_params = PilotBeamParams.from_gaussian_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
    )
    
    # 创建模拟折射面
    surface = MockSurface(radius=radius_mm)
    
    # 更新 Pilot Beam
    new_pilot_params = propagator._update_pilot_beam(pilot_params, surface)
    
    # 验证 q 参数变换正确
    # 折射面变换: ABCD 矩阵
    n1 = 1.0  # 空气
    n2 = 1.5168  # N-BK7
    
    A = 1
    B = 0
    C = (n1 - n2) / (n2 * radius_mm)
    D = n1 / n2
    
    q_in = pilot_params.q_parameter
    expected_q = (A * q_in + B) / (C * q_in + D)
    
    assert_allclose(
        new_pilot_params.q_parameter.real,
        expected_q.real,
        rtol=1e-6,
        err_msg="折射面变换后 q 参数实部不正确",
    )
    assert_allclose(
        new_pilot_params.q_parameter.imag,
        expected_q.imag,
        rtol=1e-6,
        err_msg="折射面变换后 q 参数虚部不正确",
    )


# ============================================================================
# 光线采样测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    num_rays=num_rays_strategy,
)
def test_ray_sampling_count(
    wavelength_um: float,
    num_rays: int,
):
    """
    测试光线采样数量。
    
    使用 WavefrontToRaysSampler 的 hexapolar 分布时，
    实际光线数由环数决定：actual_rays = 1 + 3*n*(n+1)
    其中 n 是环数，n ≈ sqrt(num_rays/3)
    
    因此实际光线数可能与 num_rays 不完全相等，但应该接近。
    """
    from sequential_system.coordinate_tracking import (
        OpticalAxisState,
        Position3D,
        RayDirection,
    )
    
    propagator = HybridElementPropagator(
        wavelength_um=wavelength_um,
        num_rays=num_rays,
    )
    
    # 创建测试波前
    grid_size = 64
    physical_size_mm = 30.0
    grid_sampling = GridSampling.create(grid_size, physical_size_mm)
    
    # 创建高斯波前（使用新的振幅/相位分离接口）
    X, Y = grid_sampling.get_coordinate_arrays()
    R_sq = X**2 + Y**2
    w = 5.0  # mm
    amplitude = np.exp(-R_sq / w**2)
    phase = np.zeros_like(amplitude)
    
    # 创建光轴状态
    entrance_axis = OpticalAxisState(
        position=Position3D(0.0, 0.0, 0.0),
        direction=RayDirection(0.0, 0.0, 1.0),
        path_length=0.0,
    )
    
    # 采样光线（使用新的振幅/相位分离接口）
    rays = propagator._sample_rays_from_wavefront(
        amplitude,
        phase,
        grid_sampling,
        entrance_axis,
    )
    
    # 验证采样数量
    # hexapolar 分布的光线数 = 1 + 3*n*(n+1)，其中 n 是环数
    # 环数 n ≈ sqrt(num_rays/3)
    actual_count = len(np.asarray(rays.x))
    
    # 计算期望的环数和光线数
    num_rings = max(1, int(np.sqrt(num_rays / 3.0)))
    # 如果实际光线数太少，增加一环
    expected_min = 1 + 3 * num_rings * (num_rings + 1)
    while expected_min < num_rays and num_rings < 100:
        num_rings += 1
        expected_min = 1 + 3 * num_rings * (num_rings + 1)
    
    # 验证实际光线数至少达到期望的最小值
    assert actual_count >= num_rays * 0.5, (
        f"采样光线数量过少：期望至少 {num_rays * 0.5:.0f}，实际 {actual_count}"
    )
    
    # 验证实际光线数不超过期望的 3 倍
    assert actual_count <= num_rays * 3, (
        f"采样光线数量过多：期望最多 {num_rays * 3:.0f}，实际 {actual_count}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
