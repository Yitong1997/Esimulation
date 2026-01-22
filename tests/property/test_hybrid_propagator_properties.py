"""
HybridOpticalPropagator 属性基测试

使用 hypothesis 库验证 HybridOpticalPropagator 的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 2.1-2.7, 3.1-3.6, 13.2, 16.1-16.4**
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
from hybrid_optical_propagation.hybrid_propagator import (
    HybridOpticalPropagator,
    PropagationResult,
)


# ============================================================================
# 辅助函数：创建测试用光学系统
# ============================================================================

def create_mock_surface(
    index: int,
    vertex_position: np.ndarray,
    orientation: np.ndarray = None,
    radius: float = np.inf,
    is_mirror: bool = False,
    surface_type: str = 'standard',
    material: str = 'air',
    focal_length: float = np.inf,
):
    """创建模拟的 GlobalSurfaceDefinition 对象"""
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class MockSurface:
        index: int
        surface_type: str
        vertex_position: np.ndarray
        orientation: np.ndarray
        radius: float = np.inf
        conic: float = 0.0
        is_mirror: bool = False
        semi_aperture: float = 25.0
        material: str = "air"
        asphere_coeffs: List[float] = field(default_factory=list)
        comment: str = ""
        thickness: float = 0.0
        radius_x: float = np.inf
        conic_x: float = 0.0
        focal_length: float = np.inf
        
        @property
        def surface_normal(self) -> np.ndarray:
            return -self.orientation[:, 2]
    
    if orientation is None:
        orientation = np.eye(3)
    
    return MockSurface(
        index=index,
        surface_type=surface_type,
        vertex_position=np.asarray(vertex_position),
        orientation=np.asarray(orientation),
        radius=radius,
        is_mirror=is_mirror,
        material=material,
        focal_length=focal_length,
    )


def create_simple_optical_system():
    """创建简单的光学系统（单个平面镜）
    
    平面镜在 z=100mm 处，绕 X 轴旋转 -45 度（使法向量指向入射光来的方向）。
    入射光沿 +z 方向，反射后应该沿 -y 方向。
    """
    # 绕 X 轴旋转 -45 度，使镜面法向量指向 [0, 0.707, 0.707]
    # 这样入射光 [0, 0, 1] 反射后变成 [0, -1, 0]
    angle = -np.pi / 4
    c, s = np.cos(angle), np.sin(angle)
    orientation = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])
    
    mirror = create_mock_surface(
        index=0,
        vertex_position=np.array([0.0, 0.0, 100.0]),
        orientation=orientation,
        is_mirror=True,
        material='mirror',
    )
    
    return [mirror]


def create_two_mirror_system():
    """创建双镜系统
    
    两个 45 度镜子使光线从 +z 变为 -y 再变回 +z。
    """
    # 第一个镜子在 z=100mm，绕 X 轴旋转 -45 度
    # 入射 +z，出射 -y
    angle1 = -np.pi / 4
    c1, s1 = np.cos(angle1), np.sin(angle1)
    orientation1 = np.array([
        [1, 0, 0],
        [0, c1, -s1],
        [0, s1, c1],
    ])
    
    mirror1 = create_mock_surface(
        index=0,
        vertex_position=np.array([0.0, 0.0, 100.0]),
        orientation=orientation1,
        is_mirror=True,
        material='mirror',
    )
    
    # 第二个镜子在 y=-100mm
    # 入射方向是 -y，需要镜面法向量使得反射后变成 +z
    # 镜面法向量应该是 [0, -0.707, 0.707]（指向入射光来的方向）
    # 这意味着 Z 轴是 [0, 0.707, -0.707]
    # 绕 X 轴旋转 -45 度可以得到这个
    angle2 = -np.pi / 4
    c2, s2 = np.cos(angle2), np.sin(angle2)
    orientation2 = np.array([
        [1, 0, 0],
        [0, c2, -s2],
        [0, s2, c2],
    ])
    
    mirror2 = create_mock_surface(
        index=1,
        vertex_position=np.array([0.0, -100.0, 100.0]),
        orientation=orientation2,
        is_mirror=True,
        material='mirror',
    )
    
    return [mirror1, mirror2]


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(min_value=0.4, max_value=2.0)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(min_value=1.0, max_value=10.0)

# 网格大小策略
grid_size_strategy = st.sampled_from([64, 128])



# ============================================================================
# Property 1: 入射面/出射面垂直于光轴
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_1_entrance_plane_perpendicular_to_axis(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 1: 入射面/出射面垂直于光轴**
    **Validates: Requirements 3.1, 3.3, 13.2**
    
    入射面的法向量应与入射光轴方向平行（点积绝对值为 1）。
    """
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=64,
        num_rays=50,
    )
    
    # 获取入射面定义
    entrance_plane = propagator._create_entrance_plane(0)
    
    # 获取入射光轴方向
    entrance_axis = propagator.get_optical_axis_at_surface(0, 'entrance')
    axis_direction = entrance_axis.direction.to_array()
    
    # 入射面法向量应与光轴方向平行
    plane_normal = entrance_plane['normal']
    dot_product = np.abs(np.dot(plane_normal, axis_direction))
    
    assert_allclose(
        dot_product, 1.0, atol=1e-10,
        err_msg="入射面法向量与光轴方向不平行"
    )


@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_property_1_exit_plane_perpendicular_to_axis(
    wavelength_um: float,
    w0_mm: float,
):
    """
    **Feature: hybrid-optical-propagation, Property 1: 入射面/出射面垂直于光轴**
    **Validates: Requirements 3.1, 3.3, 13.2**
    
    出射面的法向量应与出射光轴方向平行（点积绝对值为 1）。
    """
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=64,
        num_rays=50,
    )
    
    # 获取出射面定义
    exit_plane = propagator._create_exit_plane(0)
    
    # 获取出射光轴方向
    exit_axis = propagator.get_optical_axis_at_surface(0, 'exit')
    axis_direction = exit_axis.direction.to_array()
    
    # 出射面法向量应与光轴方向平行
    plane_normal = exit_plane['normal']
    dot_product = np.abs(np.dot(plane_normal, axis_direction))
    
    assert_allclose(
        dot_product, 1.0, atol=1e-10,
        err_msg="出射面法向量与光轴方向不平行"
    )


# ============================================================================
# 光轴追踪测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_optical_axis_reflection_direction(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试反射镜处光轴方向的正确性。
    
    对于 45 度倾斜的平面镜，入射光沿 +z 方向，
    反射后应该沿 -y 方向。
    """
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=64,
        num_rays=50,
    )
    
    # 获取入射和出射光轴方向
    entrance_axis = propagator.get_optical_axis_at_surface(0, 'entrance')
    exit_axis = propagator.get_optical_axis_at_surface(0, 'exit')
    
    entrance_dir = entrance_axis.direction.to_array()
    exit_dir = exit_axis.direction.to_array()
    
    # 入射方向应该是 +z
    assert_allclose(
        entrance_dir, [0, 0, 1], atol=1e-10,
        err_msg="入射方向不正确"
    )
    
    # 出射方向应该是 -y（45度反射）
    expected_exit = np.array([0, -1, 0])
    assert_allclose(
        exit_dir, expected_exit, atol=1e-10,
        err_msg="出射方向不正确"
    )


@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_optical_axis_two_mirror_system(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试双镜系统的光轴追踪。
    
    两个 45 度镜子应该使光线从 +z 变为 -y 再变回 +z。
    """
    # 创建光学系统
    optical_system = create_two_mirror_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=64,
        num_rays=50,
    )
    
    # 第一个镜子：入射 +z，出射 -y
    exit_axis_1 = propagator.get_optical_axis_at_surface(0, 'exit')
    exit_dir_1 = exit_axis_1.direction.to_array()
    assert_allclose(
        exit_dir_1, [0, -1, 0], atol=1e-10,
        err_msg="第一个镜子出射方向不正确"
    )
    
    # 第二个镜子：入射 -y，出射 +z
    exit_axis_2 = propagator.get_optical_axis_at_surface(1, 'exit')
    exit_dir_2 = exit_axis_2.direction.to_array()
    assert_allclose(
        exit_dir_2, [0, 0, 1], atol=1e-10,
        err_msg="第二个镜子出射方向不正确"
    )


# ============================================================================
# 传播器初始化测试
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_propagator_initialization(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    测试传播器初始化的正确性。
    """
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        num_rays=50,
    )
    
    # 验证属性
    assert propagator.wavelength_um == wavelength_um
    assert propagator.grid_size == grid_size
    assert propagator.num_rays == 50
    assert len(propagator.optical_system) == 1


@settings(max_examples=20, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_initial_state_creation(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试初始传播状态的创建。
    """
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=64,
        physical_size_mm=30.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=64,
        num_rays=50,
    )
    
    # 初始化传播状态
    initial_state = propagator._initialize_propagation()
    
    # 验证初始状态（使用新的振幅/相位分离接口）
    assert initial_state.surface_index == -1
    assert initial_state.position == 'source'
    assert initial_state.amplitude.shape == (64, 64)
    assert initial_state.phase.shape == (64, 64)
    assert initial_state.pilot_beam_params.wavelength_um == wavelength_um
    assert initial_state.grid_sampling.grid_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# ============================================================================
# Property 13: 网格采样信息一致性
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_13_grid_sampling_consistency(
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: hybrid-optical-propagation, Property 13: 网格采样信息一致性**
    **Validates: Requirements 17.1, 17.3, 17.4**
    
    从 PROPER 对象提取的网格采样信息应与 GridSampling 对象中存储的信息一致。
    """
    import proper
    
    # 创建光学系统
    optical_system = create_simple_optical_system()
    
    # 创建入射波面定义
    physical_size_mm = 30.0
    source = SourceDefinition(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        num_rays=50,
    )
    
    # 初始化传播状态
    initial_state = propagator._initialize_propagation()
    
    # 从 GridSampling 获取信息
    grid_sampling = initial_state.grid_sampling
    
    # 从 PROPER 对象提取信息
    proper_grid_size = proper.prop_get_gridsize(initial_state.proper_wfo)
    proper_sampling_m = proper.prop_get_sampling(initial_state.proper_wfo)
    proper_sampling_mm = proper_sampling_m * 1e3
    
    # 验证网格大小一致
    assert grid_sampling.grid_size == proper_grid_size, (
        f"网格大小不一致: GridSampling={grid_sampling.grid_size}, "
        f"PROPER={proper_grid_size}"
    )
    
    # 从 PROPER 创建 GridSampling 并比较
    grid_sampling_from_proper = GridSampling.from_proper(initial_state.proper_wfo)
    
    # 验证从 PROPER 提取的采样信息与存储的一致
    assert_allclose(
        grid_sampling_from_proper.sampling_mm,
        proper_sampling_mm,
        rtol=0.01,
        err_msg="从 PROPER 提取的采样间隔不一致"
    )


@settings(max_examples=50, deadline=None)
@given(
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
)
def test_grid_sampling_from_proper(
    wavelength_um: float,
    w0_mm: float,
):
    """
    测试从 PROPER 对象创建 GridSampling 的正确性。
    
    注意：PROPER 的采样间隔由 beam_diameter / (grid_size * beam_ratio) 决定，
    而不是简单的 physical_size / grid_size。
    """
    import proper
    
    # 创建 PROPER 波前
    grid_size = 64
    physical_size_mm = 30.0
    beam_diameter_m = physical_size_mm * 1e-3
    wavelength_m = wavelength_um * 1e-6
    beam_ratio = 0.5
    
    wfo = proper.prop_begin(
        beam_diameter_m,
        wavelength_m,
        grid_size,
        beam_ratio,
    )
    
    # 从 PROPER 创建 GridSampling
    grid_sampling = GridSampling.from_proper(wfo)
    
    # 验证网格大小
    assert grid_sampling.grid_size == grid_size
    
    # PROPER 的采样间隔
    proper_sampling_m = proper.prop_get_sampling(wfo)
    proper_sampling_mm = proper_sampling_m * 1e3
    
    # 验证采样间隔与 PROPER 一致
    assert_allclose(
        grid_sampling.sampling_mm,
        proper_sampling_mm,
        rtol=0.01,
        err_msg="采样间隔与 PROPER 不一致"
    )


@settings(max_examples=50, deadline=None)
@given(
    grid_size=grid_size_strategy,
    physical_size=st.floats(min_value=10.0, max_value=100.0),
)
def test_grid_sampling_compatibility(
    grid_size: int,
    physical_size: float,
):
    """
    测试 GridSampling 兼容性检查。
    """
    # 创建两个相同的 GridSampling
    gs1 = GridSampling.create(grid_size, physical_size)
    gs2 = GridSampling.create(grid_size, physical_size)
    
    # 应该兼容
    assert gs1.is_compatible(gs2), "相同参数的 GridSampling 应该兼容"
    
    # 创建不同网格大小的 GridSampling
    gs3 = GridSampling.create(grid_size * 2, physical_size)
    
    # 应该不兼容
    assert not gs1.is_compatible(gs3), "不同网格大小的 GridSampling 不应该兼容"
    
    # 创建物理尺寸差异较大的 GridSampling
    gs4 = GridSampling.create(grid_size, physical_size * 1.5)
    
    # 应该不兼容
    assert not gs1.is_compatible(gs4), "物理尺寸差异较大的 GridSampling 不应该兼容"

